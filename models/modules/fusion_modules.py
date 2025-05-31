from typing import Literal
from functools import partial
from argparse import Namespace

import torch
import torch.nn as nn
from torch_geometric.nn import GlobalAttention, global_mean_pool


# === Vector Processors ===
class NormVector(nn.Module):
    def __init__(self, d_s: int, d_v: int, d_f: int):
        super().__init__()
        self.emb_dim = d_v

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return torch.norm(v, dim=-1)


class NormLinearVector(nn.Module):
    def __init__(self, d_s: int, d_v: int, d_f: int):
        super().__init__()
        self.emb_dim = d_f
        self.linear = nn.Linear(d_v, d_f)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        v_norm = torch.norm(v, dim=-1)
        return self.linear(v_norm)


class FlattenVector(nn.Module):
    def __init__(self, d_s: int, d_v: int, d_f: int):
        super().__init__()
        self.emb_dim = 3 * d_v

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return v.view(v.size(0), -1)


class FlattenLinearVector(nn.Module):
    def __init__(self, d_s: int, d_v: int, d_f: int):
        super().__init__()
        self.emb_dim = d_f
        self.linear = nn.Linear(3 * d_v, d_f)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        v_flat = v.view(v.size(0), -1)
        return self.linear(v_flat)


class DirectionProjVector(nn.Module):
    def __init__(self, d_s: int, d_v: int, d_f: int):
        super().__init__()
        self.emb_dim = d_v
        self.w = nn.Parameter(torch.randn(d_v, 3))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # project each vector by learned directions
        return (v * self.w.unsqueeze(0)).sum(dim=-1)


class DirectionProjLinearVector(nn.Module):
    def __init__(self, d_s: int, d_v: int, d_f: int):
        super().__init__()
        self.emb_dim = d_f

        self.w = nn.Parameter(torch.randn(d_v, 3))
        self.linear = nn.Linear(d_v, d_f)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        v_proj = (v * self.w.unsqueeze(0)).sum(dim=-1)
        return self.linear(v_proj)


# === Scalar Processors ===
class IdentityScalar(nn.Module):
    def __init__(self, d_s: int, d_v: int, d_f: int):
        super().__init__()
        self.emb_dim = d_s
        self.identity = nn.Identity()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.identity(s)


class LinearScalar(nn.Module):
    def __init__(self, d_s: int, d_v: int, d_f: int):
        super().__init__()
        self.emb_dim = d_f
        self.linear = nn.Linear(d_s, d_f)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.linear(s)


# === Fusion Operations ===
class Only2D(nn.Module):
    def __init__(self, d_s_emb: int, d_v_emb: int, d_f: int):
        super().__init__()
        assert d_s_emb == d_f, f"`d_s_emb` and `d_f` must match in `Only2D` (Got s={d_s_emb}, f={d_f})"
        self.out_dim = d_f

    def forward(self, s_emb: torch.Tensor, v_emb: torch.Tensor) -> torch.Tensor:
        return s_emb


class Only3D(nn.Module):
    def __init__(self, d_s_emb: int, d_v_emb: int, d_f: int):
        super().__init__()
        assert d_v_emb == d_f, f"`d_v_emb` and `d_f` must match in `Only3D` (Got v={d_v_emb}, f={d_f})"
        self.out_dim = d_f

    def forward(self, s_emb: torch.Tensor, v_emb: torch.Tensor) -> torch.Tensor:
        return v_emb


class SumFusionOp(nn.Module):
    def __init__(self, d_s_emb: int, d_v_emb: int, d_f: int):
        super().__init__()
        self.out_dim = d_f

    def forward(self, s_emb: torch.Tensor, v_emb: torch.Tensor) -> torch.Tensor:
        return s_emb + v_emb


class ConcatFusionOp(nn.Module):
    def __init__(self, d_s_emb: int, d_v_emb: int, d_f: int):
        super().__init__()
        self.out_dim = d_s_emb + d_v_emb

    def forward(self, s_emb: torch.Tensor, v_emb: torch.Tensor) -> torch.Tensor:
        return torch.cat([s_emb, v_emb], dim=-1)


class AttnFusionOp(nn.Module):
    def __init__(self, d_s_emb: int, d_v_emb: int, d_f: int):
        super().__init__()
        assert (
            d_s_emb == d_v_emb
        ), f"Scalar and vector dimensions must match in `AttnFusion` (Got s={d_s_emb}, v={d_v_emb})"
        assert (
            d_s_emb == d_f
        ), f"Both scalar and vector must have dimension `d_f` in `AttnFusion` (Got s=v={d_s_emb}, d_f={d_f})"

        self.out_dim = d_f

        self.attn = nn.MultiheadAttention(d_f, num_heads=1, batch_first=True)

    def forward(self, s_emb: torch.Tensor, v_emb: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([s_emb, v_emb], dim=1)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        return attn_out.sum(dim=1)


class GateFusionOp(nn.Module):
    def __init__(self, d_s_emb: int, d_v_emb: int, d_f: int, dropout: float = 0.0):
        super().__init__()
        assert (
            d_s_emb == d_v_emb
        ), f"Scalar and vector dimensions must match in `GateFusion` (Got s={d_s_emb}, v={d_v_emb})"
        self.out_dim = d_f
        self.dropout = dropout

        self.gate_fn = nn.Sequential(
            nn.Linear(d_s_emb + d_v_emb, d_s_emb + d_v_emb),  # in_dim == hidden_dim
            nn.SiLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(d_s_emb + d_v_emb, d_f),
            nn.Sigmoid(),
        )

    def forward(self, s_emb: torch.Tensor, v_emb: torch.Tensor) -> torch.Tensor:
        h = torch.cat([s_emb, v_emb], dim=-1)
        g = self.gate_fn(h)
        return g * s_emb + (1 - g) * v_emb


class GMUFusionOp(nn.Module):
    """
    Gated Multimodal Units for Information Fusion (https://arxiv.org/pdf/1702.01992)
    Section 3.1
    hs = tanh(Ws · s) , hv = tanh(Wv · v)
    g  = sigmoid(Wg · [s;v])
    z  = g * hs + (1-g) * hv
    """

    def __init__(self, d_s_emb: int, d_v_emb: int, d_f: int):
        super().__init__()
        self.out_dim = d_f

        self.Ws = nn.Linear(d_s_emb, d_f, bias=False)
        self.Wv = nn.Linear(d_v_emb, d_f, bias=False)
        self.Wg = nn.Linear(d_s_emb + d_v_emb, d_f, bias=False)

    def forward(self, s_emb: torch.Tensor, v_emb: torch.Tensor) -> torch.Tensor:
        hs = torch.tanh(self.Ws(s_emb))
        hv = torch.tanh(self.Wv(v_emb))
        g = torch.sigmoid(self.Wg(torch.cat([s_emb, v_emb], dim=-1)))
        return g * hs + (1 - g) * hv


class MMHighwayFusionOp(nn.Module):
    """
    Highway Networks (https://arxiv.org/pdf/1505.00387)
    Section 2
    original
        h = tanh(Wh · [x])
        t = sigmoid(Wt · [x])
        y = t * h + (1 - t) * x

    multimodal version
        h = tanh(Wh · [s;v])
        g = sigmoid(Wg · [s;v])
        z = g * h + (1 - g) * (b * s + (1 - b) * v)

    v1. b = 0.5 (constant)
    v2. b = sigmoid(Wb · [s;v])
    """

    def __init__(self, d_s_emb: int, d_v_emb: int, d_f: int, version: Literal["v1", "v2"] = "v1"):
        super().__init__()
        assert (
            d_s_emb == d_v_emb
        ), f"Scalar and vector dimensions must match in `MMHighwayFusion` (Got s={d_s_emb}, v={d_v_emb})"
        assert (
            d_s_emb == d_f
        ), f"Both scalar and vector must have dimension `d_f` in `MMHighwayFusion` (Got s=v={d_s_emb}, d_f={d_f})"
        self.out_dim = d_f
        self.version = version

        self.Wh = nn.Linear(d_s_emb + d_v_emb, d_f)
        self.Wg = nn.Linear(d_s_emb + d_v_emb, d_f)
        if self.version == "v1":
            self.b = 0.5
        elif self.version == "v2":
            self.Wb = nn.Linear(d_s_emb + d_v_emb, d_f, bias=False)

    def forward(self, s_emb: torch.Tensor, v_emb: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([s_emb, v_emb], dim=-1)
        h = torch.tanh(self.Wh(concat))
        g = torch.sigmoid(self.Wg(concat))
        if self.version == "v1":
            z = g * h + 0.5 * (1 - g) * (s_emb + v_emb)
        elif self.version == "v2":
            b = torch.sigmoid(self.Wb(concat))
            z = g * h + (1 - g) * (b * s_emb + (1 - b) * v_emb)
        return z


# === Projection Layers ===
class IdentityProj(nn.Module):
    def __init__(self, d_in, d_f: int):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.identity(h)


class LinearProj(nn.Module):
    def __init__(self, d_in, d_f: int):
        super().__init__()
        self.linear = nn.Linear(d_in, d_f)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)


# === Fusion Head ===
class FusionHead(nn.Module):
    def __init__(self, args: Namespace, num_classes: int):
        super().__init__()
        self.args = args
        self.d_s = self.args.d_scalar
        self.d_v = self.args.d_vector
        self.d_f = self.args.d_fusion
        self.read_out = self.args.read_out
        self.num_classes = num_classes

        self.s_proc = SCALAR_MAP[self.args.s_proc_cls](self.d_s, self.d_v, self.d_f)
        self.v_proc = VECTOR_MAP[self.args.v_proc_cls](self.d_s, self.d_v, self.d_f)
        if self.args.fusion_cls in {"gate"}:
            self.fusion = FUSION_MAP[self.args.fusion_cls](
                self.s_proc.emb_dim, self.v_proc.emb_dim, self.d_f, self.args.dropout
            )
        else:
            self.fusion = FUSION_MAP[self.args.fusion_cls](self.s_proc.emb_dim, self.v_proc.emb_dim, self.d_f)
        self.proj = PROJ_MAP[self.args.proj_cls](self.fusion.out_dim, self.d_f)

        if self.read_out == "mean":
            self.pool = global_mean_pool
        elif self.read_out == "attn":
            # 논문 그대로 구현, `gate_nn`은 내부적으로 softmax 적용됨
            gate_nn = nn.Linear(self.d_f, 1)
            nn = nn.Linear(self.d_f, self.d_f)
            self.pool = GlobalAttention(gate_nn=gate_nn, nn=nn)

        self.mlp = nn.Sequential(nn.Linear(self.d_f, self.d_f), nn.SiLU(), nn.Linear(self.d_f, self.num_classes))

    def forward(self, s, v, batch):
        v_emb = self.v_proc(v)
        s_emb = self.s_proc(s)
        h = self.fusion(s_emb, v_emb)
        h = self.proj(h)
        g = self.pool(h, batch)
        return self.mlp(g)


# 1. Vector Processing - in: (N, d_v, 3)
VECTOR_MAP = {
    "norm": NormVector,  # -> (N, d_v)
    "norm_linear": NormLinearVector,  # -> (N, d_v) -> linear -> (N, d_f)
    "flatten": FlattenVector,  # -> flatten -> (N, 3 * d_v)
    "flatten_linear": FlattenLinearVector,  # -> flatten -> (N, 3*d_v) -> linear -> (N, d_f)
    "direction_proj": DirectionProjVector,  # -> direction-weighted sum -> (N, d_v)
    "direction_proj_linear": DirectionProjLinearVector,  # -> direction proj -> (N, d_v) -> linear -> (N, d_f)
}

# 2. Scalar Processing
SCALAR_MAP = {
    "none": IdentityScalar,  # (N, d_s) -> Identity -> (N, d_s)
    "linear": LinearScalar,  # (N, d_s) -> linear -> (N, d_f)
}

# 3. Fusion Operation
FUSION_MAP = {
    "sum": SumFusionOp,  # just summation. `s` and `v`'s shape should be (N, d_f)
    "concat": ConcatFusionOp,  # torch.cat([s, v], dim=-1)
    "attn": AttnFusionOp,  # g = MLP(torch.cat([s, v], dim=-1))
    "gate": GateFusionOp,  # t = torch.stack([s, v], dim=1); ATTN(t, t, t).sum(dim=1)
    "gmu": GMUFusionOp,  # hs = tanh(W x s); hv = tanh(W x v); g = sigmoid(W x [s;v]); z = g * hs + (1-g) * hv
    # === 성능 끌어올리기 위한 발악 ===
    # h = tanh(Wh · [s;v]); g = sigmoid(Wg · [s;v]); z = g * h + 0.5 * (1 - g) * (s + v)
    "mm_highway_avg": partial(MMHighwayFusionOp, version="v1"),
    # h = tanh(Wh · [s;v]); g = sigmoid(Wg · [s;v]); b = sigmoid(Wb · [s;v]); z = g * h + (1 - g) * (b * s + (1 - b) * v)
    "mm_highway_gate": partial(MMHighwayFusionOp, version="v2"),
    # === for ablation ===
    "2d_only": Only2D,  # s -> Identity -> (N, d_f). `s`'s shape should be (N, d_f)
    "3d_only": Only3D,  # v -> Identity -> (N, d_f). `v`'s shape should be (N, d_f)
    "simple_mlp": ConcatFusionOp,  # torch.cat([s, v], dim=-1) -> linear -> (N, d_f)
}

# 4. Project Operation
PROJ_MAP = {
    "none": IdentityProj,  # INPUT -> Identity -> (N, d_s)
    "linear": LinearProj,  # INPUT -> (N, d_f)
}
