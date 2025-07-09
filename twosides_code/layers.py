import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class IntraGraphAttention(nn.Module):
    def __init__(self, input_dim, out_dim, heads=1):
        super().__init__()
        self.gat = GATConv(input_dim, out_dim // heads, heads=heads, concat=True)

    def forward(self, data):
        x = F.elu(data.x)
        x = self.gat(x, data.edge_index)
        return x


class InterGraphAttention(nn.Module):
    def __init__(self, input_dim, out_dim, heads=1):
        super().__init__()
        self.gat = GATConv((input_dim, input_dim), out_dim // heads, heads=heads, concat=True)

    def forward(self, h_data, t_data, b_graph):
        h_x = F.elu(h_data.x)
        t_x = F.elu(t_data.x)

        t_rep = self.gat((h_x, t_x), b_graph.edge_index)
        h_rep = self.gat((t_x, h_x), b_graph.edge_index[[1, 0]])
        return h_rep, t_rep


class CoAttentionLayer(nn.Module):
    def __init__(self, n_features, hidden_dim=None):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim if hidden_dim else n_features  # 默认保持维度不变

        self.w_q = nn.Linear(n_features, self.hidden_dim)
        self.w_k = nn.Linear(n_features, self.hidden_dim)
        self.bias = nn.Parameter(torch.zeros(self.hidden_dim))
        self.a = nn.Parameter(torch.zeros(self.hidden_dim, 1))

        # 初始化
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.zeros_(self.w_q.bias)
        nn.init.zeros_(self.w_k.bias)
        nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.a)

    def forward(self, receiver, attendant):
        if receiver.dim() == 2:
            receiver = receiver.unsqueeze(1)
        if attendant.dim() == 2:
            attendant = attendant.unsqueeze(1)

        B, N, D = receiver.shape
        assert D == self.n_features, f"Expected input dim {self.n_features}, got {D}"

        receiver_flat = receiver.reshape(B * N, D)
        attendant_flat = attendant.reshape(B * N, D)

        keys = self.w_k(receiver_flat).reshape(B, N, -1)
        queries = self.w_q(attendant_flat).reshape(B, N, -1)

        e_activations = torch.tanh(queries + keys + self.bias)
        e_scores = torch.matmul(e_activations, self.a).squeeze(-1)  # (B, N)
        return e_scores


class RESCAL(nn.Module):
    def __init__(self, n_rels, n_features):
        super().__init__()
        self.rel_emb = nn.Embedding(n_rels, n_features * 2)
        self.rel_proj = nn.Sequential(
            nn.ELU(),
            nn.Linear(n_features * 2, n_features * 2),
            nn.ELU(),
            nn.Linear(n_features * 2, n_features),
        )
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, heads, tails, rels, alpha_scores=None):
        rels = self.rel_proj(self.rel_emb(rels))  # (B, D)
        rels = F.normalize(rels, dim=-1)
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)

        pair = (heads.unsqueeze(-3) * tails.unsqueeze(-2)).unsqueeze(-2)  # [B, 1, 1, D, D]
        rels = rels.view(-1, 1, 1, rels.size(-1), 1)                      # [B, 1, 1, D, 1]
        scores = torch.matmul(pair, rels).squeeze(-1).squeeze(-1)        # [B]

        if alpha_scores is not None:
            scores = alpha_scores * scores

        return scores.sum(dim=(-2, -1))  # Return scalar scores
