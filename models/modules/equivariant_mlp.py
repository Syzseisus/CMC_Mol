from typing import Tuple

import torch
import torch.nn as nn


class EquivariantMLP(nn.Module):
    def __init__(self, d_s: int, d_v: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(3 * d_s, d_s), nn.SiLU(), nn.Linear(d_s, d_s))
        self.scalar_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_s, d_s))
        self.vector_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_s, d_v))
        self.norm = nn.LayerNorm(d_s)
        self.d_v = d_v

    def forward(
        self,
        s: torch.Tensor,  # (N, d_s)
        v: torch.Tensor,  # (N, d_v, 3)
        edge_index: torch.LongTensor,  # (2, E)
        edge_attr: torch.Tensor,  # (E, d_s) — RBF 임베딩
        edge_vec_unit: torch.Tensor,  # (E, 3) — 방향 단위벡터
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index

        # edge, vector message
        edge_msg = self.edge_mlp(torch.cat([s[row], s[col], edge_attr], dim=-1))  # (E, d_s)
        v_msg = self.vector_mlp(edge_msg)  # (E, d_v)
        v_msg = v_msg.view(-1, self.d_v, 1) * edge_vec_unit.unsqueeze(1)  # (E, d_v, 3)

        # aggregation
        v_out = torch.zeros_like(v)
        s_out = torch.zeros_like(s)
        v_out.index_add_(0, row, v_msg)
        s_out.index_add_(0, row, edge_msg)

        # update
        s = self.norm(s + self.scalar_mlp(s_out))
        v = v + v_out
        return s, v
