import torch
import torch.nn as nn


class RBFEncoder(nn.Module):
    def __init__(self, num_rbf: int = 300, cutoff: float = 30.0, d_out: int = 128, dropout: float = 0.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / (centers[1] - centers[0]).item() ** 2
        self.mlp = nn.Sequential(
            nn.Linear(num_rbf, num_rbf),  # in_dim == hidden_dim
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(num_rbf, d_out),
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, dist):
        diff = dist.unsqueeze(-1) - self.centers
        exp = torch.exp(-self.gamma * diff**2)
        return self.mlp(exp)
