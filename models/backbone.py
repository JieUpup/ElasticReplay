import torch
import torch.nn as nn

class SimpleDynamicBrainNet(nn.Module):
    def __init__(self, node_dim=16, hidden_dim=64, num_classes=2):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.cls = nn.Linear(hidden_dim, num_classes)

    def encode_graph(self, x_t, adj_t):
        h = self.node_proj(x_t)          # [B, N, H]
        h = torch.matmul(adj_t, h)       # [B, N, H]
        h = h.mean(dim=1)                # [B, H]
        return h

    def forward(self, x_seq, adj_seq):
        feats = []
        T = x_seq.size(1)
        for t in range(T):
            feats.append(self.encode_graph(x_seq[:, t], adj_seq[:, t]))
        feats = torch.stack(feats, dim=1)   # [B, T, H]
        out, _ = self.temporal(feats)
        z = out[:, -1]
        logits = self.cls(z)
        return logits
