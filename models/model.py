import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_transformer_pytorch import GraphTransformer, Mol2VecEncoder, ESM1vEncoder, CrossAttention

class MAIDTA(nn.Module):
    def __init__(self,
                 drug_dim=128,
                 protein_dim=128,
                 gtrans_layers=3,
                 num_heads=4,
                 dropout=0.1):
        super().__init__()


        self.drug_seq_encoder = Mol2VecEncoder(embed_dim=drug_dim)
        self.prot_seq_encoder = ESM1vEncoder(embed_dim=protein_dim)


        self.drug_graph_encoder = GraphTransformer(
            node_in_dim=56, edge_in_dim=18, hidden_dim=drug_dim, num_layers=gtrans_layers
        )
        self.prot_graph_encoder = GraphTransformer(
            node_in_dim=256, edge_in_dim=1, hidden_dim=protein_dim, num_layers=gtrans_layers
        )


        self.binding_fc = nn.Sequential(
            nn.Linear(drug_dim + protein_dim, drug_dim),
            nn.ReLU(),
            nn.Linear(drug_dim, 1)
        )
        self.binding_attn = nn.MultiheadAttention(embed_dim=drug_dim, num_heads=num_heads, batch_first=True)

        self.cross_attn = CrossAttention(dim=drug_dim, num_heads=num_heads)
        self.fusion = nn.Sequential(
            nn.Linear(drug_dim * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def get_binding_mask(self, drug_feat, prot_feat, threshold=0.5):
        # 原子-残基注意力权重
        attn_map = torch.bmm(drug_feat, prot_feat.transpose(1, 2))
        attn_score = torch.sigmoid(attn_map.mean(dim=-1))
        mask = (attn_score >= threshold).float()
        mask[mask == 0] = -float('inf')
        mask[mask == 1] = 0.0
        return mask, attn_score

    def forward(self,
                drug_smiles, drug_graph_node, drug_graph_edge,
                prot_seq, prot_graph_node, prot_graph_edge):


        feat_drug_seq = self.drug_seq_encoder(drug_smiles)
        feat_prot_seq = self.prot_seq_encoder(prot_seq)

        feat_drug_graph = self.drug_graph_encoder(drug_graph_node, drug_graph_edge)
        feat_prot_graph = self.prot_graph_encoder(prot_graph_node, prot_graph_edge)

        binding_mask, attn_weight = self.get_binding_mask(feat_drug_graph, feat_prot_graph)

        drug_fused = self.cross_attn(feat_drug_seq, feat_drug_graph, mask=binding_mask)
        prot_fused = self.cross_attn(feat_prot_seq, feat_prot_graph, mask=binding_mask)

        drug_pool = drug_fused.mean(dim=1)
        prot_pool = prot_fused.mean(dim=1)
        joint_feat = torch.cat([drug_pool, prot_pool], dim=-1)

        output = self.fusion(joint_feat)
        return output.squeeze(), attn_weight, binding_mask