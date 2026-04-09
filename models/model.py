import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_feat=None):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class GraphTransformer(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, num_layers, num_heads=4):
        super().__init__()
        self.node_emb = nn.Linear(node_in_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, node_feat, edge_feat):
        x = self.node_emb(node_feat)
        for layer in self.layers:
            x = layer(x)
        return x

class Mol2VecEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(1000, embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return self.fc(x)


class ESM1vEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(33, embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return self.proj(x)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, q, kv, mask=None):
        out, _ = self.mha(q, kv, kv, attn_mask=mask)
        return q + out