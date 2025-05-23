from typing import Literal

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
        self.linear = nn.Linear(d_v, d_f)
        self.emb_dim = d_f

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
        self.linear = nn.Linear(3 * d_v, d_f)
        self.emb_dim = d_f

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        v_flat = v.view(v.size(0), -1)
        return self.linear(v_flat)


class DirectionProjVector(nn.Module):
    def __init__(self, d_s: int, d_v: int, d_f: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d_v, 3))
        self.emb_dim = d_v

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # project each vector by learned directions
        return (v * self.w.unsqueeze(0)).sum(dim=-1)


class DirectionProjLinearVector(nn.Module):
    def __init__(self, d_s: int, d_v: int, d_f: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d_v, 3))
        self.linear = nn.Linear(d_v, d_f)
        self.emb_dim = d_f

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        v_proj = (v * self.w.unsqueeze(0)).sum(dim=-1)
        return self.linear(v_proj)


# === Scalar Processors ===
class IdentityScalar(nn.Module):
    def __init__(self, d_s: int, d_v: int, d_f: int):
        super().__init__()
        self.identity = nn.Identity()
        self.emb_dim = d_s

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.identity(s)


class LinearScalar(nn.Module):
    def __init__(self, d_s: int, d_v: int, d_f: int):
        super().__init__()
        self.linear = nn.Linear(d_s, d_f)
        self.emb_dim = d_f

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


class GateFusionOp(nn.Module):
    def __init__(self, d_s_emb: int, d_v_emb: int, d_f: int):
        super().__init__()
        assert (
            d_s_emb == d_v_emb
        ), f"Scalar and vector dimensions must match in `GateFusion` (Got s={d_s_emb}, v={d_v_emb})"

        self.gate_fn = nn.Sequential(nn.Linear((d_s_emb + d_v_emb), d_f), nn.ReLU(), nn.Linear(d_f, d_f), nn.Sigmoid())
        self.out_dim = d_f

    def forward(self, s_emb: torch.Tensor, v_emb: torch.Tensor) -> torch.Tensor:
        h = torch.cat([s_emb, v_emb], dim=-1)
        g = self.gate_fn(h)
        return g * s_emb + (1 - g) * v_emb


class AttnFusionOp(nn.Module):
    def __init__(self, d_s_emb: int, d_v_emb: int, d_f: int):
        super().__init__()
        assert (
            d_s_emb == d_v_emb
        ), f"Scalar and vector dimensions must match in `AttnFusion` (Got s={d_s_emb}, v={d_v_emb})"
        assert (
            d_s_emb == d_f
        ), f"Both scalar and vector must have dimension `d_f` in `AttnFusion` (Got s=v={d_s_emb}, d_f={d_f})"

        self.attn = nn.MultiheadAttention(d_f, num_heads=1, batch_first=True)
        self.out_dim = d_f

    def forward(self, s_emb: torch.Tensor, v_emb: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([s_emb, v_emb], dim=1)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        return attn_out.sum(dim=1)


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
    def __init__(
        self,
        s_proc_cls: nn.Module,
        v_proc_cls: nn.Module,
        fusion_cls: nn.Module,
        proj_cls: nn.Module,
        d_s: int,
        d_v: int,
        d_f: int,
        read_out: Literal["mean", "attn"],
        num_classes: int,
    ):
        super().__init__()
        self.s_proc = s_proc_cls(d_s, d_v, d_f)
        self.v_proc = v_proc_cls(d_s, d_v, d_f)
        self.fusion = fusion_cls(self.s_proc.emb_dim, self.v_proc.emb_dim, d_f)
        self.proj = proj_cls(self.fusion.out_dim, d_f)

        if "only" in fusion_cls.__name__.lower():
            print(f"{f' Ablation: {fusion_cls.__name__} ':=^80}")

        if read_out == "mean":
            self.pool = global_mean_pool
        elif read_out == "attn":
            gate_nn = nn.Sequential(nn.Linear(d_f, d_f), nn.ReLU(), nn.Linear(d_f, 1))
            self.pool = GlobalAttention(gate_nn)

        self.mlp = nn.Sequential(nn.Linear(d_f, d_f), nn.SiLU(), nn.Linear(d_f, num_classes))

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
    # for ablation
    "2d_only": Only2D,  # s -> Identity -> (N, d_f). `s`'s shape should be (N, d_f)
    "3d_only": Only3D,  # v -> Identity -> (N, d_f). `v`'s shape should be (N, d_f)
    "simple_mlp": ConcatFusionOp,  # torch.cat([s, v], dim=-1) -> linear -> (N, d_f)
}

# 4. Project Operation
PROJ_MAP = {
    "none": IdentityProj,  # INPUT -> Identity -> (N, d_s)
    "linear": LinearProj,  # INPUT -> (N, d_f)
}


def build_modular_head(args, num_classes):
    # ablation 할라고 넣어둔 거.
    # 일단은 기본 조합만 사용함.
    s_proc_cls = SCALAR_MAP[args.s_proc_cls]
    v_proc_cls = VECTOR_MAP[args.v_proc_cls]
    fusion_cls = FUSION_MAP[args.fusion_cls]
    proj_cls = PROJ_MAP[args.proj_cls]

    return FusionHead(
        s_proc_cls=s_proc_cls,
        v_proc_cls=v_proc_cls,
        fusion_cls=fusion_cls,
        proj_cls=proj_cls,
        d_s=args.d_scalar,
        d_v=args.d_vector,
        d_f=args.d_fusion,
        read_out=args.read_out,
        num_classes=num_classes,
    )
