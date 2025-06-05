import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter_mean
from ogb.graphproppred.mol_encoder import AtomEncoder

from models.modules import RBFEncoder, SelfAttention, UnifiedEquivariantGNN


class CrossModalFT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_rbf = self.args.num_rbf
        self.cutoff = self.args.cutoff
        self.aggr = self.args.aggr
        self.d_s = self.args.d_scalar
        self.layers = self.args.num_layers
        self.n_heads = self.args.num_attn_heads
        self.dropout = self.args.dropout

        self.embed_x = AtomEncoder(self.d_s)
        self.rbf = RBFEncoder(self.num_rbf, self.cutoff, self.d_s)
        self.gnn_layers = nn.ModuleList([UnifiedEquivariantGNN(self.d_s, self.dropout) for _ in range(self.layers)])
        self.sa_layers = nn.ModuleList([SelfAttention(self.d_s, self.n_heads) for _ in range(self.layers)])

    def forward(self, data: Data):
        """
        data has
            x           : Tensor of node scalar features, shape (N, d_s)
            edge_vec    : Tensor of relative position vectors (Δx, Δy, Δz), shape (E, 3)
            edge_len    : Tensor of Euclidean distances for each edge, shape (E,)
            edge_index  : LongTensor of edge indices, shape (2, E)
            num_nodes   : Integer count of nodes N in the graph (int)
        where N == num_nodes, E == num_edges
        """

        # 초기 임베딩
        s = self.embed_x(data.x)
        edge_attr = self.rbf(data.edge_len)
        edge_vec_unit = data.edge_vec / (data.edge_len.unsqueeze(-1) + 1e-6)
        v = scatter_mean(
            edge_vec_unit,
            data.edge_index[0],
            dim=0,
            dim_size=data.x.shape[0],  # 고립된 노드가 있을 경우 N보다 작아짐
        )

        # 고립된 노드를 랜덤 단위벡터로 초기화
        missing = (v.norm(dim=-1) == 0).nonzero(as_tuple=False).view(-1)
        if missing.numel() > 0:
            rnd = torch.randn(len(missing), 3, device=v.device)
            denom = torch.clamp(rnd.norm(dim=-1, keepdim=True), min=1e-6)
            v[missing] = rnd / denom

        # 모델 forward
        for gnn, sa in zip(self.gnn_layers, self.sa_layers):
            s, v = gnn(s, v, data.edge_index, edge_attr, edge_vec_unit)
            s = sa(s)

        return s, v
