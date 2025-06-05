import torch
import torch.nn as nn


class UnifiedEquivariantGNN(nn.Module):
    def __init__(self, d_s: int, dropout: float = 0.0):
        super().__init__()
        # Node feature extractor: 입력 [s, directional projection]
        self.node_gate = nn.Sequential(nn.Linear(d_s + 1, d_s), nn.SiLU(), nn.Linear(d_s, d_s))
        # Edge feature extractor: 입력 [edge_attr, cross_mag]
        self.edge_gate = nn.Sequential(nn.Linear(d_s + 1, d_s), nn.SiLU(), nn.Linear(d_s, d_s))
        # Message MLP: 입력 concatenated latent node & edge features
        # 3d_s → 2d_s → d_s
        self.message_gate = nn.Sequential(
            nn.Linear(3 * d_s, 2 * d_s),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(2 * d_s, d_s),
        )
        self.proj_edge_msg = nn.Linear(d_s, 1)
        self.update_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_s, d_s))
        self.norm = nn.LayerNorm(d_s)

    def forward(self, s, v, edge_index, edge_attr, edge_vec_unit):
        """
        Args:
            s             (N, d_s)  - atomic chemical feature
            v             (N, 3)    - mean of edge unit vector
            edge_index    (2, E)    - edge index
            edge_attr     (E, d_s)  - RBF(edge_len)
            edge_vec_unit (E, 3)    - edge unit vector

        Message Passing:
            1. Scalar
                - node feature : s         - atomic chemical feature
                - edge_feature : edge_attr - RBF(edge_len)
            2. Vector
                - node feature : dot_*     - residual cosine similarity between node vectors and edge directions
                - edge_feature : cross_mag - cross product magnitude between node vectors
            3. Message computation:
                - scalar message: output of MLP using concatenated features
                - vector message: scalar coefficient projected from MLP, scaled along edge direction (EGNN-style)
            4. Aggregation:
                - sum messages from neighbors and update node features
                - re-normalize v to maintain unit vector constraint
        """
        # ===== Pre-processing
        # node feature를 엣지 쌍 단위로 준비 및 Pre-compute unsqueezed unit vector
        row, col = edge_index
        s_i, s_j = s[row], s[col]  # (E, d_s)
        v_i, v_j = v[row], v[col]  # (E, 3)

        # ===== Scalar, Vector 각각에서의 node / edge feature 계산
        # 1. Scalar node feature == `s_i`, `s_j`  (E, d_s)
        # 2. Scalar edge feature == `edge_attr`  (E, d_s)

        # 3. Vector node feature
        dot_ij = (v_i * edge_vec_unit).sum(dim=-1, keepdim=True)  # cos
        dot_ji = (v_j * (-edge_vec_unit)).sum(dim=-1, keepdim=True)  # cos
        res_ij = 1.0 - dot_ij  # (E, 1)
        res_ji = 1.0 - dot_ji  # (E, 1)

        # 4. Vector edge feature
        cross_prod = torch.cross(v_i, v_j)  # sin between v_i, v_j
        cross_mag = torch.linalg.vector_norm(cross_prod, dim=-1, keepdim=True)  # (E, 1)

        # ===== Concat scalar and vector
        node_feat_i = self.node_gate(torch.cat([s_i, res_ij], dim=-1))  # (E, d_s)
        node_feat_j = self.node_gate(torch.cat([s_j, res_ji], dim=-1))  # (E, d_s)
        edge_feat = self.edge_gate(torch.cat([edge_attr, cross_mag], dim=-1))  # (E, d_s)

        # ===== Message Passing
        # 1. Compute message - EGNN-style
        # 1-1. Concat node and edge feature
        msg_input = self.message_gate(torch.cat([node_feat_i, node_feat_j, edge_feat], dim=-1))
        # 1-2. Scalar edge message : use just as is
        s_msg = msg_input
        # 1-3. Vector edge message: project to 1D then give direction
        coeff = self.proj_edge_msg(msg_input)  # (E, 1)
        v_msg = edge_vec_unit * coeff  # (E, 3)

        # 2. Aggregate messages for vector and scalar updates
        s_out = torch.zeros_like(s)
        v_out = torch.zeros_like(v)
        s_out.index_add_(0, row, s_msg)
        v_out.index_add_(0, row, v_msg)

        # 3. Update: apply non-linear transformation and normalization only to scalar
        s = self.norm(s + self.update_mlp(s_out))
        v = v + v_out
        denom = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=1e-6)
        v = v / denom  # keep unit vector

        return s, v
