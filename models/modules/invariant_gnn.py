from typing import Union

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import Aggregation

AGGR = Union[str, Aggregation]


class InvariantGNN(MessagePassing):
    def __init__(self, d: int, aggr: AGGR = "add"):
        super().__init__(aggr=aggr, node_dim=0)
        self.phi = nn.Sequential(nn.Linear(3 * d, d), nn.SiLU(), nn.Linear(d, d))
        self.psi = nn.Sequential(nn.SiLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, x, edge_index, edge_attr):
        h = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        res = x + self.psi(h)
        return self.norm(res)

    def message(self, x_i, x_j, edge_attr):
        return self.phi(torch.cat([x_i, x_j, edge_attr], dim=-1))
