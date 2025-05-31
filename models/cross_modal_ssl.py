import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter_mean
from ogb.graphproppred.mol_encoder import AtomEncoder

from models.modules import RBFEncoder, SelfAttention, UnifiedEquivariantGNN


class CrossModalSSL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_rbf = self.args.num_rbf
        self.cutoff = self.args.cutoff
        self.d_s = self.args.d_scalar
        self.layers = self.args.num_layers
        self.n_heads = self.args.num_attn_heads
        self.alpha = self.args.alpha
        self.dropout = self.args.dropout

        self.embed_x = AtomEncoder(self.d_s)
        self.rbf = RBFEncoder(self.num_rbf, self.cutoff, self.d_s, self.dropout)
        self.gnn_layers = nn.ModuleList([UnifiedEquivariantGNN(self.d_s, self.dropout) for _ in range(self.layers)])
        self.sa_layers = nn.ModuleList([SelfAttention(self.d_s, self.n_heads) for _ in range(self.layers)])

        self.s_mask_token = nn.Parameter(torch.zeros(self.d_s))
        self.e_mask_token = nn.Parameter(torch.zeros(self.d_s))
        self.v_mask_token = nn.Parameter(torch.zeros(3))
        nn.init.normal_(self.s_mask_token, mean=0.0, std=self.d_s ** (-0.5))  # similar to `AtomEncoder`
        nn.init.normal_(self.e_mask_token, mean=0.0, std=self.d_s ** (-0.5))  # similar to `RBFEncoder`

    def forward(self, data: Data):
        """
        data has
            x               : Tensor of node scalar features, shape (N, d_s)
            edge_vec        : Tensor of relative position vectors (Δx, Δy, Δz), shape (E, 3)
            edge_len        : Tensor of Euclidean distances for each edge, shape (E,)
            edge_index        : LongTensor of edge indices, shape (2, E)
            target_atom     : LongTensor of masked node indices for which to predict scalar features, shape (N,)
            target_edge_len : Tensor of ground-truth lengths for masked edges, shape (E,)
            num_nodes       : Integer count of nodes N in the graph (int)
            mask_atom       : BoolTensor mask for nodes to be masked, shape (N,)
            mask_edge       : BoolTensor mask for edges to be masked, shape (E,)
        where N == num_nodes, E == num_edges
        """

        # 초기 임베딩
        s = self.embed_x(data.x)
        edge_attr = self.rbf(data.edge_len)
        edge_vec_unit = data.edge_vec / (data.edge_len.unsqueeze(-1) + 1e-8)

        # 마스킹
        s[data.mask_atom] = self.s_mask_token
        edge_attr[data.mask_edge] = self.e_mask_token
        edge_vec_unit[data.mask_edge] = self.v_mask_token

        # 마스킹 반영한 초기화
        v = scatter_mean(edge_vec_unit, data.edge_index[0], dim=0)  # (N, 3)

        # 모델 forward
        for gnn, sa in zip(self.gnn_layers, self.sa_layers):
            s, v = gnn(s, v, data.edge_index, edge_attr, edge_vec_unit)
            s = sa(s)

        return s, v
