import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphTransformer
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class Mol2VecEncoder(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.proj(x)
        return x


class ESM1vEncoder(nn.Module):
    def __init__(self, embed_dim=128, context_window=1024):
        super().__init__()
        self.context_window = context_window
        self.proj = nn.Linear(1280, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=context_window)

    def forward(self, x):
        x = self.proj(x)
        x = self.pos_enc(x)
        return x


class GraphEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=128, num_layers=3, heads=8):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.gt = GraphTransformer(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            node_attn_out_dim=hidden_dim,
        )

    def forward(self, x, edge_index, edge_attr, batch):
        node_feat = self.node_proj(x)
        edge_feat = self.edge_proj(edge_attr)
        node_feat = self.gt(node_feat, edge_index, edge_feat)
        return node_feat


class InterpretableBindingRegion(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.fc_drug = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.fc_prot = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.att = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, X_D, X_P, threshold=0.5):
        batch, ND, F = X_D.shape
        batch, NP, F = X_P.shape

        X_D_exp = X_D.unsqueeze(2).expand(-1, -1, NP, -1)
        X_P_exp = X_P.unsqueeze(1).expand(-1, ND, -1, -1)
        X_D_proj = self.fc_drug(X_D_exp)
        X_P_proj = self.fc_prot(X_P_exp)
        concat = torch.cat([X_D_proj, X_P_proj], dim=-1)
        A = self.att(concat)
        A = A.squeeze(-1)

        w = A.mean(dim=1)

        X_D_pooled = X_D.mean(dim=1)
        X_D_expanded = X_D_pooled.unsqueeze(1).expand(-1, NP, -1)
        w_exp = w.unsqueeze(-1)
        X_B = X_D_expanded * w_exp + X_P * (1 - w_exp)

        Mask_B_init = (w >= threshold).float()

        return X_B, w, Mask_B_init


def expand_mask_to_2d(Mask_B_init, NP):
    Mask_B = Mask_B_init.unsqueeze(2)
    Mask_B = Mask_B.expand(-1, -1, NP)
    Mask_B_2d = torch.where(Mask_B == 1, 0.0, -float('inf'))
    return Mask_B_2d


class CrossScaleAttention(nn.Module):
    def __init__(self, feature_dim=128, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = feature_dim // num_heads
        self.d_v = self.d_k
        self.W_o = nn.Linear(feature_dim, feature_dim)

    def mmha(self, Q, K, V, mask=None):
        batch, seq_q, _ = Q.shape
        _, seq_k, _ = K.shape

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = self.W_o(out)
        return out

    def forward(self, X_M, X_E, X_B, mask_2d=None):
        out1 = self.mmha(X_M, X_B, X_B)
        out2 = self.mmha(X_B, X_M, X_M)
        D_star = (out1 + out2) / 2
        D_star = D_star.mean(dim=1)

        out3 = self.mmha(X_E, X_B, X_B)
        out4 = self.mmha(X_B, X_E, X_E, mask_2d)
        P_star = (out3 + out4) / 2
        P_star = P_star.mean(dim=1)

        return D_star, P_star


class PredictionMLP(nn.Module):
    def __init__(self, input_dim=256, hidden_dims=[1024, 512, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


class MAIDTA(nn.Module):
    def __init__(self, config):
        super().__init__()
        F = config['feature_dim']
        num_heads = config['num_heads']

        self.drug_seq_encoder = Mol2VecEncoder(
            vocab_size=config.get('vocab_size', 1000),
            embed_dim=F
        )

        self.prot_seq_encoder = ESM1vEncoder(embed_dim=F)

        self.drug_graph_encoder = GraphEncoder(
            node_dim=config.get('drug_node_dim', 48),
            edge_dim=config.get('drug_edge_dim', 18),
            hidden_dim=F,
            num_layers=config.get('gtrans_layers', 3),
            heads=num_heads
        )
        self.prot_graph_encoder = GraphEncoder(
            node_dim=config.get('prot_node_dim', 256),
            edge_dim=config.get('prot_edge_dim', 1),
            hidden_dim=F,
            num_layers=config.get('gtrans_layers', 3),
            heads=num_heads
        )

        self.binding_module = InterpretableBindingRegion(feature_dim=F)

        self.cross_attn = CrossScaleAttention(feature_dim=F, num_heads=num_heads)

        self.mlp = PredictionMLP(input_dim=2*F)

        self.threshold = config.get('threshold', 0.5)

    def forward(self, data_drug, data_prot):
        X_M = self.drug_seq_encoder(data_drug['smiles_tokens'])

        X_E = self.prot_seq_encoder(data_prot['seq_tokens'])

        X_D_nodes = self.drug_graph_encoder(
            data_drug['graph_x'],
            data_drug['graph_edge_index'],
            data_drug['graph_edge_attr'],
            data_drug['batch']
        )
        X_D, mask_D = to_dense_batch(X_D_nodes, data_drug['batch'])

        X_P_nodes = self.prot_graph_encoder(
            data_prot['graph_x'],
            data_prot['graph_edge_index'],
            data_prot['graph_edge_attr'],
            data_prot['batch']
        )
        X_P, mask_P = to_dense_batch(X_P_nodes, data_prot['batch'])

        X_B, w, Mask_B_init = self.binding_module(X_D, X_P, self.threshold)

        mask_2d = expand_mask_to_2d(Mask_B_init, X_P.size(1))

        D_star, P_star = self.cross_attn(X_M, X_E, X_B, mask_2d)

        joint_feat = torch.cat([D_star, P_star], dim=-1)
        pred = self.mlp(joint_feat)

        return pred, w, Mask_B_init
