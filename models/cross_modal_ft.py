import torch
import torch.nn as nn
from torch_geometric.data import Data
from ogb.graphproppred.mol_encoder import AtomEncoder

from models.modules import RBFEncoder, InvariantGNN, SelfAttention, EquivariantMLP


def unit_sphere_(tensor: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """
    In-place initialize the input tensor with random unit vectors on a hypersphere scaled by alpha.

    Args:
        tensor  : Tensor of shape (..., N) to initialize in-place.
        alpha   : Scale factor to apply to each unit vector (default: 0.01).

    Returns:
        The same tensor, now filled with scaled unit vectors.
    """
    rand = torch.randn_like(tensor)  # shape (..., N)
    lengths = rand.norm(dim=-1, keepdim=True)  # shape (..., 1)
    unit = rand / (lengths + 1e-8)  # shape (..., N)

    with torch.no_grad():
        tensor.copy_(unit * alpha)
    return tensor


class CrossModalFT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_rbf = self.args.num_rbf
        self.cutoff = self.args.cutoff
        self.aggr = self.args.aggr
        self.d_s = self.args.d_scalar
        self.d_v = self.args.d_vector
        self.layers = self.args.num_layers
        self.n_heads = self.args.num_attn_heads
        self.alpha = self.args.alpha

        self.embed_x = AtomEncoder(self.d_s)
        self.rbf = RBFEncoder(self.num_rbf, self.cutoff, self.d_s)
        self.s_layers = nn.ModuleList(InvariantGNN(self.d_s, self.aggr) for _ in range(self.layers))
        self.v_layers = nn.ModuleList(EquivariantMLP(self.d_s, self.d_v) for _ in range(self.layers))
        self.sa_layers = nn.ModuleList(SelfAttention(self.d_s, self.n_heads) for _ in range(self.layers))

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
        v = torch.zeros(data.num_nodes, self.d_v, 3, device=s.device)
        unit_sphere_(v, self.alpha)
        edge_attr = self.rbf(data.edge_len)
        edge_vec_unit = data.edge_vec / (data.edge_len.unsqueeze(-1) + 1e-8)

        # 모델 forward
        for sl, vl, sa in zip(self.s_layers, self.v_layers, self.sa_layers):
            s = sl(s, data.edge_index, edge_attr)
            s, v = vl(s, v, data.edge_index, edge_attr, edge_vec_unit)
            s = sa(s)

        return s, v
