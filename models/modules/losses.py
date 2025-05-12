import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorToAtomLogitsModule(nn.Module):
    """
    Predict atom type logits from vector features v_i ∈ R^{d_v x 3}

        - d_v : 벡터 특성의 차원
        - num_classes : 원소 종류의 수
    """

    def __init__(self, d_v, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_v, d_v), nn.SiLU(), nn.Linear(d_v, num_classes))

    def forward(self, v):  # v: [N, d_v, 3]
        v_norm = torch.norm(v, dim=-1)  # [N, d_v]
        return self.mlp(v_norm)  # [N, num_classes]


class ScalarToDistanceModule(nn.Module):
    """
    Predict a single scalar distance for each edge
    from the concatenated node scalars [s_i || s_j].

        - d_s     : 노드 특성의 차원
        - 2 * d_s : 엣지 단위로 concat된 노드 특성의 차원
        - 1       : distance에 해당하는 scalar
    """

    def __init__(self, d_s: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * d_s, d_s), nn.SiLU(), nn.Linear(d_s, 1))

    def forward(self, s: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:  # [N, d_s]  # [2, E]  # [E]
        row, col = edge_index
        h_edge = torch.cat([s[row], s[col]], dim=-1)  # [E, 2*d_s]
        return self.mlp(h_edge).squeeze(-1)  # [E]
