from argparse import Namespace

import torch
import torch.nn as nn
from torch_geometric.nn import GraphNorm
from torch_geometric.utils import softmax
from torch_scatter import scatter_sum, scatter_mean


def mlp(in_dim, hidden_dim, out_dim, dropout=0.15):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class AllInOneLayer(nn.Module):
    """
    1. 노드 / 엣지 피쳐
        - Scalar
            - s_node: 원소의 화학적 특성 (N, 9) -> (N, d_s) - via ogb.AtomEncoder
            - s_edge: 엣지 종류 임베딩 (E, 1) -> (E, d_s) - via ogb.BondEncoder
        - Vector
            - v_node: SH(원자간 단위벡터)의 평균 (E, 3) -> (E, d_v)
            - v_edge: RBF(원자간 거리) (E, 3) -> (E, d_v)
            - d_v = (l_max + 1) ** 2 - 1
        - Unified (for EGNN-style Message Passing)
            - u_node = MLP(concat([s_node, v_node])) - d_s + d_v -> d_u
            - u_edge = MLP(concat([s_edge, v_edge])) - d_s + d_v -> d_u
        - 전역정보: CLS 토큰 활용
            - `s_node_cls = nn.Parameter(torch.zeros(d_s))`
            - `v_node_cls = torch.zeros(d_v, 3)`
    2. 메시지 패싱
        1. 메시지 계산
            - Molecule 엣지 메시지: `msg_init_ij = MLP(concat[u_node_i, u_node_j, u_edge_ij])` -> (E, d_s)
            - Scalar 엣지 메시지: `s_msg_ij = msg_init_ij`
            - Vector 엣지 메시지 (EGNN-style):
                1. `coeff = MLP(msg_init_ij)` -> (E, 1)
                2. `v_msg_ij = v_edge_ij * coeff` -> (E, d_v)
        2. 노드 별 메시지 가중치 계산
            - `a = MLP(concat[s_node_i, s_node_j, s_edge_ij, v_node_i, v_node_j, v_edge_ij])` -> (E,)
        3. 메시지 집계: 엣지 메시지의 합 - `a_score`를 통한 가중합
            - `s_msg = a_score.unsqueeze(-1) * s_msg_ij` -> (E, d_s)
            - `s_agg = scatter_sum(s_msg, row, dim=0, dim_size=N)` -> (N, d_s)
            - `v_msg = a_score.unsqueeze(-1) * v_msg_ij` -> (E, d_v)
            - `v_agg = scatter_sum(v_msg, row, dim=0, dim_size=N)` -> (N, d_v)
        4. 노드 업데이트
            - `s_delta = MLP(concat[s; s_agg; s_cls])` -> (N, d_s)
            - `s_out = GraphNorm(s + s_delta)` -> (N, d_s)
            - `v_delta = MLP(concat[v; v_agg; v_cls])` -> (N, d_v)
            - `v_out = v + v_delta` -> (N, d_v)
        5. 전역정보 업데이트
            - `s_cls_delta = MLP(concat[s_cls; s_node_mean; s_edge_mean])` -> (B, d_s)
            - `s_cls_out = LayerNorm(s_cls + s_cls_delta)` -> (B, d_s)
            - `v_cls_delta = MLP(concat[v_cls; v_node_mean; v_edge_mean])` -> (B, d_v)
            - `v_cls_out = v_cls + v_cls_delta` -> (B, d_v)
    """

    def __init__(self, args: Namespace):
        super().__init__()
        self.d_s = args.d_s  # 스칼라 흐름 임베딩 차원, 64
        self.d_v = (args.l_max + 1) ** 2 - 1  # 벡터 흐름 임베딩 차원, 8
        self.d_u = args.d_u  # 공통 흐름 임베딩 차원, 64
        self.d_msg = args.d_msg  # 메시지 임베딩 차원, 64
        self.dropout = args.dropout  # 드롭아웃 비율, 0.15
        self.n_heads = args.n_heads  # 멀티헤드 어텐션 헤드 수, 4

        # 1. Unified Flow
        # Node feature extractor
        self.node_mlp = mlp(self.d_s + self.d_v, 2 * self.d_u, self.d_u, self.dropout)
        # Edge feature extractor
        self.edge_mlp = mlp(self.d_s + self.d_v, 2 * self.d_u, self.d_u, self.dropout)

        # 2. Edge Message
        # Message MLP
        self.msg_mlp = mlp(3 * self.d_u, 2 * self.d_u, self.d_s, self.dropout)
        self.s2v_proj = mlp(self.d_s, self.d_s // 2, 1, self.dropout)
        # Attention (== message weight)
        self.multi_head_attn_mlp = nn.ModuleList(
            [mlp(3 * self.d_s + 3 * self.d_v, self.d_s + self.d_v, 1) for _ in range(self.n_heads)]
        )

        # 3. Update
        # Each Node
        self.s_delta_mlp = mlp(self.d_s * 3, self.d_s, self.d_s, self.dropout)
        self.v_delta_mlp = mlp(self.d_v * 3, self.d_v, self.d_v, self.dropout)
        self.norm_node = GraphNorm(self.d_s)
        self.v_delta_gate = mlp(self.d_s + 3, self.d_s, 1, self.dropout)
        # CLS token
        self.s_cls_delta_mlp = mlp(self.d_s * 3, self.d_s, self.d_s, self.dropout)
        self.v_cls_delta_mlp = mlp(self.d_v * 3, self.d_v, self.d_v, self.dropout)
        self.norm_cls = nn.LayerNorm(self.d_s)

    def forward(self, s, v, s_edge, v_edge, edge_index, s_cls, v_cls, batch):
        """
        Args:
            s          (N, d_s) - atomic chemical feature
            v          (N, d_v) - SH(mean of edge unit vector)
            s_edge     (E, d_s) - bond type embedding
            v_edge     (E, d_v) - RBF(edge_len)
            edge_index (2, E)   - edge index
            s_cls      (B, d_s) - initialized to learnable zeros
            v_cls      (B, d_v) - initialized to fixed zeros
            batch      (N,)     - batch index
        """
        # ===== Pre-processing
        row, col = edge_index
        N = s.shape[0]  # 노드 수
        B = s_cls.shape[0]  # 배치 수

        # node feature를 엣지 쌍 단위로 준비 및 Pre-compute unsqueezed unit vector
        # Scalar Flow
        s_node_i, s_node_j = s[row], s[col]  # (E, d_s)
        s_edge_ij = s_edge  # (E, d_s)

        # Vector Flow
        v_node_i, v_node_j = v[row], v[col]  # (E, d_v)
        v_edge_ij = v_edge  # (E, d_v)

        # ===== EGNN-style Message Passing x Attention
        # 0) Unified Flow: 노드와 엣지 피쳐를 통합하여 메시지 계산에 사용
        u_node_i = self.node_mlp(torch.cat([s_node_i, v_node_i], dim=-1))  # (E, d_u)
        u_node_j = self.node_mlp(torch.cat([s_node_j, v_node_j], dim=-1))  # (E, d_u)
        u_edge_ij = self.edge_mlp(torch.cat([s_edge_ij, v_edge_ij], dim=-1))  # (E, d_u)
        # 1) Compute message in Unified Flow
        msg_init_ij = self.msg_mlp(torch.cat([u_node_i, u_node_j, u_edge_ij], dim=-1))  # (E, d_s)
        # 2) Scalar edge message
        s_msg_ij = msg_init_ij  # (E, d_s)
        # 3) Vector edge message: project to 1D then give direction
        coeff = self.s2v_proj(msg_init_ij)  # (E, 1)
        v_msg_ij = v_edge_ij * coeff  # (E, d_v)
        # 4) Compute attention
        attn_ij_head_scores = []
        for attn_head in self.multi_head_attn_mlp:
            raw_attn_ij = attn_head(
                torch.cat([s_node_i, s_node_j, s_edge_ij, v_node_i, v_node_j, v_edge_ij], dim=-1)
            )  # (E, 1)  # TODO: 불변성분넣기
            attn_ij = softmax(raw_attn_ij, index=row, num_nodes=N)  # (E,)
            attn_ij_head_scores.append(attn_ij)
        attn_ij = torch.cat(attn_ij_head_scores, dim=-1).mean(dim=-1)  # (E,)
        # 5) Weighted message aggregation
        s_w_msg_ij = attn_ij.unsqueeze(-1) * s_msg_ij  # (E, d_s)
        s_agg_i = scatter_sum(s_w_msg_ij, row, dim=0, dim_size=N)  # (N, d_s)
        v_w_msg_ij = attn_ij.unsqueeze(-1) * v_msg_ij  # (E, d_v)
        v_agg_i = scatter_sum(v_w_msg_ij, row, dim=0, dim_size=N)  # (N, d_v)

        # ===== Update
        #  - Scalar: Residual + GraphNorm / Vector: Residual with Clipping (for SE(3) equivariant)
        # each node
        s_delta = self.s_delta_mlp(torch.cat([s, s_agg_i, s_cls[batch]], dim=-1))  # (N, d_s)
        v_delta = self.v_delta_mlp(torch.cat([v, v_agg_i, v_cls[batch]], dim=-1))  # (N, d_v)
        # scalar - Residual + GraphNorm
        s_out = self.norm_node(s + s_delta)  # (N, d_s)
        # vector - Residual with Gating
        v_norm = torch.norm(v, dim=-1, keepdim=True)  # 원본 크기, (N, 1)
        delta_norm = torch.norm(v_delta, dim=-1, keepdim=True)  # 델타 크기, (N, 1)
        cos_sim = (v * v_delta).sum(dim=-1, keepdim=True) / (v_norm * delta_norm + 1e-6)  # 원본과 델타 각도, (N, 1)
        gate_input = torch.cat([v_norm, delta_norm, cos_sim, s], dim=-1)  # (N, d_s + 3)
        v_delta_gate = torch.sigmoid(self.v_delta_gate(gate_input))  # (N, 1)
        v_out = v + v_delta * v_delta_gate  # (N, d_v)

        # CLS token
        s_node_mean = scatter_mean(s_out, batch, dim=0, dim_size=B)  # (B, d_s)
        s_edge_mean = scatter_mean(s_agg_i, batch, dim=0, dim_size=B)  # (B, d_s)
        s_cls_delta = self.s_cls_delta_mlp(torch.cat([s_cls, s_node_mean, s_edge_mean], dim=-1))  # (B, d_s)
        s_cls_out = self.norm_cls(s_cls + s_cls_delta)  # (B, d_s)
        v_node_mean = scatter_mean(v_out, batch, dim=0, dim_size=B)  # (B, d_v)
        v_edge_mean = scatter_mean(v_agg_i, batch, dim=0, dim_size=B)  # (B, d_v)
        v_cls_delta = self.v_cls_delta_mlp(torch.cat([v_cls, v_node_mean, v_edge_mean], dim=-1))  # (B, d_v)
        v_cls_out = v_cls + v_cls_delta  # (B, d_v)

        return s_out, v_out, s_cls_out, v_cls_out
