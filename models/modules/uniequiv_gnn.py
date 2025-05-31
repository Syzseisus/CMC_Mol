import torch
import torch.nn as nn


class UnifiedEquivariantGNN(nn.Module):
    def __init__(self, d_s: int, d_v: int):
        super().__init__()
        # Node feature extractor: 입력 [s, directional projection]
        self.node_gate = nn.Sequential(nn.Linear(d_s + d_v, d_s), nn.SiLU(), nn.Linear(d_s, d_s))
        # Edge feature extractor: 입력 [edge_attr, cross_mag]
        self.edge_gate = nn.Sequential(nn.Linear(d_s + d_v, d_s), nn.SiLU(), nn.Linear(d_s, d_s))
        # Message MLP: 입력 concatenated latent node & edge features
        self.message_gate = nn.Sequential(nn.Linear(3 * d_s, d_s), nn.SiLU(), nn.Linear(d_s, d_v))
        self.update_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_s, d_s))
        self.norm = nn.LayerNorm(d_s)

        self.blocks = nn.ModuleList(
            [
                self.node_gate,
                self.edge_gate,
                self.message_gate,
                self.update_mlp,
                self.norm,
            ]
        )

    @property
    def num_blocks(self):
        return len(self.blocks)

    def freeze_block(self, indices, freeze: bool = True):
        for i in indices:
            for p in self.blocks[i].parameters():
                p.requires_grad_(not freeze)

    def forward(self, s, v, edge_index, edge_attr, edge_vec_unit):
        """
        Args:
            s             (N, d_s)    - atomic chemical feature
            v             (N, d_v, 3) - random unit vector, where 3D information is encoded in the direction
            edge_index    (2, E)      - edge index
            edge_attr     (E, d_s)    - RBF(edge_len)
            edge_vec_unit (E, 3)      - edge unit vector

        Message Passing:
            1. Scalar
                - node feature : s         - atomic chemical feature
                - edge_feature : edge_attr - RBF(edge_len)
            2. Vector
                - node feature : dot_*     - dot product of v and edge_vec_unit
                                             -> how much v is "aligned" with edge direction
                - edge_feature : cross_mag - cross product of v and edge_vec_unit
                                             -> how much v is "perpendicular" to edge direction
        """
        # ===== Pre-processing
        # node feature를 엣지 쌍 단위로 준비 및 Pre-compute unsqueezed unit vector
        row, col = edge_index
        s_i, s_j = s[row], s[col]  # Scalar node feature  (E, d_s)
        v_i, v_j = (
            v[row],
            v[col],
        )  # Vector node feature  (E, d_v, 3)
        evu = edge_vec_unit.unsqueeze(1)  # (E, 1, 3)

        # ===== Scalar, Vector 각각에서의 node / edge feature 계산
        # 1. Scalar node feature == `s_i`, `s_j`  (E, d_s)
        # 2. Scalar edge feature == `edge_attr`  (E, d_s)

        # 3. Vector node feature
        dot_ij = torch.sum(v_j * evu, dim=-1)  # (E, d_v)
        dot_ji = torch.sum(v_i * evu, dim=-1)  # (E, d_v)

        # 4. Vector edge feature
        cross_prod = torch.cross(v_j, evu, dim=-1)  # (E, d_v, 3)
        cross_mag = torch.norm(cross_prod, dim=-1)  # (E, d_v)

        # ===== Concat scalar and vector
        node_feat_i = self.node_gate(torch.cat([s_i, dot_ij], dim=-1))  # (E, d_s)
        node_feat_j = self.node_gate(torch.cat([s_j, dot_ji], dim=-1))  # (E, d_s)
        edge_feat = self.edge_gate(torch.cat([edge_attr, cross_mag], dim=-1))  # (E, d_s)

        # ===== Message Passing
        # 1. Compute message
        # 1-1. Concat node and edge feature
        msg_input = self.message_gate(torch.cat([node_feat_i, node_feat_j, edge_feat], dim=-1))
        # 1-2. Scalar edge message : just passing MLP
        s_msg = msg_input
        # 1-3. Vector edge message: project along the edge unit vector to **ensure equivariance**
        v_msg = msg_input.view(-1, self.d_v, 1) * evu

        # 2. Aggregate messages for vector and scalar updates
        s_out = torch.zeros_like(s)
        v_out = torch.zeros_like(v)
        s_out.index_add_(0, row, s_msg)
        v_out.index_add_(0, row, v_msg)

        # 3. Update: apply non-linear transformation and normalization only to scalar
        s = self.norm(s + self.update_mlp(s_out))
        v = v + v_out

        return s, v
