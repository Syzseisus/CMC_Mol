import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.attn_mlp = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        y, _ = self.attn(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        res = x + self.attn_mlp(y.squeeze(0))
        return self.norm(res)
