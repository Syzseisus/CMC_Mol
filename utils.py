import io
import os
import glob
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Union, Sequence
from collections import defaultdict as ddict

import yaml
import torch
import torch.distributed as dist
from torch.utils.data.dataset import Subset
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Dataset as PyG_Dataset

from models.modules import PROJ_MAP, FUSION_MAP, SCALAR_MAP, VECTOR_MAP
from data_provider.moleculenet_stats import TASK_TYPE, REGRESSION, CLASSIFICATION


def split_dataset(dataset: Union[PyG_Dataset, Dataset], split: Union[float, Sequence[float]]) -> List[Subset]:
    """
    Splits a dataset into training, validation, and optionally test subsets based on the provided split ratio.

    Args:
        dataset (Dataset): The full dataset to be split.
        split (float or list/tuple of floats):
            - If float: treated as train ratio (0 < split < 1), remaining is validation.
            - If 1-element list/tuple: same as float.
            - If 2-element list/tuple: train / validation.
            - If 3-element list/tuple: train / validation / test.
            - Total will be normalized to sum == 1.
            - Length must be between 1 and 3.

    Returns:
        list[Subset]: A list of 2 or 3 `Subset` objects.
    """

    n_total = len(dataset)

    if isinstance(split, float):
        assert 0 < split < 1, f"`args.split` as float must be between 0 and 1. (Got {split})"
        ratios = [split, 1 - split]

    elif isinstance(split, (list, tuple)):
        if len(split) > 3:
            raise ValueError(f"`args.split` list/tuple must have 1-3 elements. (Got {len(split)})")
        total = sum(split)
        ratios = [s / total for s in split]
        if len(split) == 1:
            if not (0 < split[0] < 1):
                raise ValueError(f"Single-element `args.split` must be a train ratio < 1. (Got {split[0]})")
            ratios = [split[0], 1 - split[0]]
    else:
        raise ValueError(f"`args.split` must be a float or a list/tuple of 1-3 elements. (Got {split}, {type(split)})")

    lengths = [int(r * n_total) for r in ratios]

    # Adjust rounding error
    while sum(lengths) < n_total:
        lengths[-1] += 1
    while sum(lengths) > n_total:
        lengths[-1] -= 1

    return random_split(dataset, lengths)


def broadcast_str(value: str):
    value_bytes = value.encode()
    max_len = 256
    padded = value_bytes + b" " * (max_len - len(value_bytes))
    tensor = torch.ByteTensor(list(padded)).to("cuda")

    dist.broadcast(tensor, src=0)
    return bytes(tensor.tolist()).decode().strip()


def parse_split(value):
    """
    Train/validation[/test] split ratio.
    Can be a float (e.g., "0.8") or a comma-separated list of floats (e.g., "0.7,0.2,0.1").
    """
    try:
        return float(value)
    except ValueError:
        pass

    try:
        parts = value.split(",")
        return [float(v) for v in parts]
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid --split value: {value}.\n"
            "Must be either a float (e.g., '0.8') or a comma-separated list of floats (e.g., '0.7,0.2,0.1')."
        )


def resolve_sdf_path(sdf_name: str, root: str) -> str:
    # Case 1: sdf_name looks like a path
    if os.path.sep in sdf_name or sdf_name.startswith("."):
        parent = os.path.abspath(os.path.dirname(sdf_name))
        root_abs = os.path.abspath(root)
        if parent == root_abs:
            sdf_path = sdf_name
        else:
            raise FileExistsError(
                f"SDF file and shards are in different folders:\n" f"SDF   : {parent}\n" f"Shards: {root_abs}"
            )

    # Case 2: sdf_name is a plain filename
    elif os.path.basename(sdf_name) == sdf_name:
        sdf_full_path = os.path.join(root, sdf_name)
        if os.path.isfile(sdf_full_path):
            sdf_path = sdf_full_path
        else:
            raise FileExistsError(f"SDF file not found at: {sdf_full_path}")

    # Case 3: unhandled input
    else:
        raise ValueError(f"Invalid sdf_name format: {sdf_name}")

    return sdf_path


def add_arg_group(parser, group_name, category_dict):
    def add_argument(*args, **kwargs):
        dest = kwargs.get("dest")
        if not dest:
            # try to extract --arg style to arg name
            for arg in args:
                if arg.startswith("--"):
                    dest = arg.lstrip("-").replace("-", "_")
                    break
        if dest:
            category_dict.setdefault(group_name, []).append(dest)
        return parser.add_argument(*args, **kwargs)

    return add_argument


def get_args():
    parser = argparse.ArgumentParser()
    categories = {}
    # fmt: off
    # === Project Config ===
    add = add_arg_group(parser, "Project Config", categories)
    add("--project", type=str, default="cmc_atom", help="WandB project name")
    add("--save_dir", type=str, default="../save", help="Path to save checkpoints and logs")
    add("--ckpt_folder", type=str, default="checkpoints", help="Directory name where saving checkpoints : ckpt_dir = save_dir/project/\{now\}/ckpt_folder")
    add("--log_folder", type=str, default="log", help="Directory name where saving logs : log_dir = save_dir/project/\{now\}/log_folder")
    add("--seed", type=int, default=1013, help="Random seed")

    # === Optimization / Training Hyperparameters ===
    add = add_arg_group(parser, "Optimization / Training Hyperparameters", categories)
    add("--max_epochs", type=int, default=100, help="Number of training epochs")
    add("--batch_size", type=int, default=128, help="Batch size")
    add("--num_workers", type=int, default=4, help="Number of worker processes for DataLoader")
    add("--prefetch_factor", type=int, default=2, help="Number of worker processes for DataLoader")
    add("--lr", type=float, default=1e-5, help="Learning rate")
    add("--lr_scheduler", type=str, default="cosine_warmup", choices=["cosine", "cosine_warmup"], help="Learning rate scheduler")
    add("--warmup_ratio", type=float, default=0.05, help="Warmup ratio for cosine_warmup")
    add("--wd", type=float, default=5e-5, help="Weight decay")
    add("--clipping", type=float, default=1.0, help="Gradient clipping")
    add("--lambda_atom", type=float, default=1.0, help="Weight for the full atom feature predictionloss.")
    add("--lambda_bond", type=float, default=1.0, help="Weight for the bond length and type prediction loss.")
    add("--lambda_bond_dist", type=float, default=0.2, help="Weight for the bond length prediction loss.")

    # === Data Config ===   
    add = add_arg_group(parser, "Data Config", categories)
    add("--root", type=str, default="/workspace/DATASET/PCQM4M_V2/", help="Root directory for dataset storage")
    add("--sdf_name", type=str, default="pcqm4m-v2-train.sdf", help="Name of PCQM4Mv2 SDF file")
    add("--lmdb_path", type=str, default="/workspace/DATASET/PCQM4M_V2/aug5", help="Path for LMDB folder")
    add("--mask_ratio", type=float, default=0.30, help="Ratio of masked atoms/coords for SSL")
    add("--mask_atom_strat", type=str, default="random", choices=["random", "anti_c_dominant"], help="Masking strategy : random (default) or anti-C dominant")
    add("--alpha", type=float, default=0.01, help="Scale factor to apply to each unit vector")
    add("--split", type=parse_split, default=0.8, help=parse_split.__doc__)
    add("--limit", type=int, default=None, help="Limit number of data samples")

    # === Trainer / Logging / Callback Config ===
    add = add_arg_group(parser, "Trainer / Logging / Callback Config", categories)
    add("--patience", type=int, default=3, help="Early stopping patience")
    add("--top_k", type=int, default=3, help="Top-k checkpoints to keep")
    add("--log_every_n_steps", type=int, default=1, help="Log every N steps")

    # === Model Architecture Config ===
    add = add_arg_group(parser, "Model Architecture Config", categories)
    add("--dropout", type=float, default=0.15, help="Dropout rate")
    add("--d_scalar", type=int, default=256, help="Hidden dimension for scalar features")
    add("--d_vector", type=int, default=128, help="Hidden dimension for vector features in head")
    add("--num_layers", type=int, default=6, help="Number of GNN layers")
    add("--num_attn_heads", type=int, default=4, help="Number of attention heads")
    add("--num_rbf", type=int, default=300, help="Number of RBF kernels for edge encoding")
    add("--cutoff", type=float, default=10.0, help="Distance cutoff for RBF")
    add("--aggr", type=str, default="mean", choices=["mean", "add", "max"], help="Aggregation method in GNN")
    # fmt: on

    args = parser.parse_args()

    # === Create `sdf_path` ===
    args.sdf_path = resolve_sdf_path(args.sdf_name, args.root)

    # === Generate `now` only on rank 0 and broadcast to others ===
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0
    args.now = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") if is_main else "none"
    if dist.is_available() and dist.is_initialized():
        args.now = broadcast_str(args.now)

    # === Structure output directories ===
    args.save_dir = os.path.join(args.save_dir, args.project, args.now)
    args.ckpt_dir = os.path.join(args.save_dir, args.ckpt_folder)
    args.log_dir = os.path.join(args.save_dir, args.log_folder)

    # === Runtime Generated Fields ===
    registed = {k for v in categories.values() for k in v}
    runtime = [k for k in vars(args) if k not in registed]
    categories["Runtime Generated Fields"] = runtime

    return args, categories


def get_args_ft():
    parser = argparse.ArgumentParser()
    categories = {}
    # fmt: off
    # === Project Config ===
    add = add_arg_group(parser, "Project Config", categories)
    add("--project", type=str, default="cmc_atom_down")
    add("--save_dir", type=str, default="../save")
    add("--ckpt_folder", type=str, default="checkpoints")
    add("--log_folder", type=str, default="log")
    add("--seed", type=int, default=1013, help="Random seed")
    add("--pretrain_ckpt", type=str, required=True)
    add("--k_fold", type=int, default=5, help="Number of K-Folds")
    add("--save_metrics_json", type=str, default=None, help="Path to save aggregated K-Fold metrics as JSON")
    add("--zero_shot", action="store_true", default=False, help="Zero-shot test")

    # === Optimization / Training Hyperparameters ===
    add = add_arg_group(parser, "Optimization / Training Hyperparameters", categories)
    add("--max_epochs", type=int, default=50, help="Number of training epochs")
    add("--epoch_freeze", type=int, default=3, help="Number of epochs to freeze the backbone")
    add("--unfreeze_steps", type=int, default=2, help="Number of epochs to unfreeze the backbone")
    add("--batch_size", type=int, default=256, help="Batch size")
    add("--num_workers", type=int, default=8, help="Number of worker processes for DataLoader")
    add("--prefetch_factor", type=int, default=2, help="Number of worker processes for DataLoader")
    add("--lr", type=float, default=5e-3, help="Learning rate")
    add("--lr_scheduler", type=str, default="cosine_warmup", choices=["cosine", "cosine_warmup"], help="Learning rate scheduler")
    add("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for cosine_warmup")
    add("--wd", type=float, default=5e-5, help="Weight decay")
    add("--clipping", type=float, default=1.0, help="Gradient clipping")

    # === Data Config ===
    add = add_arg_group(parser, "Data Config", categories)
    add("--dataset_name", type=str, default="bbbp", help="MoleculeNet dataset name")
    add("--root", type=str, default="/workspace/DATASET/MoleculeNet/", help="Root directory for dataset storage")
    add("--alpha", type=float, default=0.01, help="Scale factor to apply to each unit vector")
    add("--limit", type=int, default=None, help="Limit number of data samples")
    add("--target", type=str, default=None, help="Only for QM8 and QM9. Target name")
    add("--split_strat", type=str, default="force_scaffold", choices=["default", "force_scaffold", "force_random"], help="Split strategy : default (default) or force scaffold/random")

    # === Trainer / Logging / Callback Config ===
    add = add_arg_group(parser, "Trainer / Logging / Callback Config", categories)
    add("--top_k", type=int, default=-1, help="Top-k checkpoints to keep")
    add("--log_every_n_steps", type=int, default=1, help="Log every N steps")
    add("--log_preds", action="store_true", default=False, help="Log predictions and targets")

    # === Model Architecture Config ===
    add = add_arg_group(parser, "Model Architecture Config", categories)
    add("--dropout", type=float, default=0.15, help="Dropout rate")
    add("--d_scalar", type=int, default=256, help="Hidden dimension for scalar features")
    add("--d_vector", type=int, default=128, help="Hidden dimension for vector features in head")
    add("--num_layers", type=int, default=6, help="Number of GNN layers")
    add("--num_attn_heads", type=int, default=4, help="Number of attention heads")
    add("--num_rbf", type=int, default=300, help="Number of RBF kernels for edge encoding")
    add("--cutoff", type=float, default=10.0, help="Distance cutoff for RBF")
    add("--aggr", type=str, default="mean", choices=["mean", "add", "max"], help="Aggregation method in GNN")
    add("--full_ft", action="store_true", default=False, help="Unfreeze pre-trained model parameters and do Full Fine-tuning. (default: False == Linear Probing)")

    # === Fusion Model Config ===
    add = add_arg_group(parser, "Fusion Model Config", categories)
    add("--s_proc_cls", type=str, default="linear", choices=SCALAR_MAP.keys(), help="scalar processing을 위한 classs name")
    add("--v_proc_cls", type=str, default="direction_embed", choices=VECTOR_MAP.keys(), help="vector processing을 위한 classs name")
    add("--fusion_cls", type=str, default="gmu", choices=FUSION_MAP.keys(), help="fusion을 위한 classs name")
    add("--proj_cls", type=str, default="linear", choices=PROJ_MAP.keys(), help="projection을 위한 classs name")
    add("--d_fusion", type=int, default=128, help="Hidden dimension for fusing two modals.")
    add("--read_out", type=str, default="attn", choices=["mean", "attn"], help="Fusion 다음 Node -> Graph ReadOut 방법.")
    # fmt: on

    args = parser.parse_args()

    # === Dataset Config ===
    args.task_type = TASK_TYPE[args.dataset_name]
    args.num_classes = (
        CLASSIFICATION[args.dataset_name]["num_tasks"]
        if args.task_type == "classification"
        else REGRESSION[args.dataset_name]["num_tasks"]
    )
    categories["Dataset Config"] = ["task_type", "num_classes"]

    # === Generate `now` only on rank 0 and broadcast to others
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0
    args.now = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") if is_main else "none"
    if dist.is_available() and dist.is_initialized():
        args.now = broadcast_str(args.now)

    # === Structure output directories
    args.save_dir = os.path.join(args.save_dir, args.project, f"{args.now}_{args.dataset_name}")
    args.ckpt_dir = os.path.join(args.save_dir, args.ckpt_folder)
    args.log_dir = os.path.join(args.save_dir, args.log_folder)
    args.lmdb_path = os.path.join(args.root, args.dataset_name, "processed", "lmdb")

    # === Runtime Generated Fields ===
    registed = {k for v in categories.values() for k in v}
    runtime = [k for k in vars(args) if k not in registed]
    categories["Runtime Generated Fields"] = runtime

    return args, categories


def get_args_inference():
    parser = argparse.ArgumentParser()
    categories = {}
    # fmt: off
    # === Project Config ===
    add = add_arg_group(parser, "Project Config", categories)
    add("--project", type=str, default="cmc_atom_inference")
    add("--finetune_dir", type=str, required=True)
    add("--save_dir", type=str, default="../save")
    add("--log_folder", type=str, default="log")
    add("--save_metrics_json", type=str, default=None, help="Path to save aggregated K-Fold metrics as JSON")

    # === Trainer / Logging / Callback Config ===
    add = add_arg_group(parser, "Trainer / Logging / Callback Config", categories)
    add("--log_every_n_steps", type=int, default=1, help="Log every N steps")
    add("--log_preds", action="store_true", default=False, help="Log predictions and targets")
    # fmt: on

    args = parser.parse_args()

    # === Bring args from finetune_ckpt_dir ===
    config_path = glob.glob(os.path.join(args.finetune_dir, "log/wandb/run-*", "files/config.yaml"))[0]
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in config.items():
        if k in {"_wandb", "now", "save_dir", "log_dir"}:
            # "_wandb": wandb 관련 정보 - 무시
            # 나머지: 아래서 할당할 예정
            continue
        if k not in vars(args):
            setattr(args, k, v["value"])
    registed = {k for v in categories.values() for k in v}
    runtime = [k for k in vars(args) if k not in registed]
    categories["Finetune Config"] = runtime

    # === Generate `now` only on rank 0 and broadcast to others
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0
    args.now = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") if is_main else "none"
    if dist.is_available() and dist.is_initialized():
        args.now = broadcast_str(args.now)

    # === Structure output directories
    args.save_dir = os.path.join(args.save_dir, args.project, f"{args.now}_{args.dataset_name}")
    args.log_dir = os.path.join(args.save_dir, args.log_folder)

    # === Dataset Config ===
    categories["Dataset Config"] = ["task_type", "num_classes"]

    # === Runtime Generated Fields ===
    registed = {k for v in categories.values() for k in v}
    runtime = [k for k in vars(args) if k not in registed]
    categories["Runtime Generated Fields"] = runtime

    return args, categories


def format_args(args, categories):
    args_dict = vars(args)
    arg_to_cat = {arg: cat for cat, arg_list in categories.items() for arg in arg_list}
    grouped = ddict(list)

    for key, val in args_dict.items():
        cat = arg_to_cat.get(key, "Other")
        val_str = f'"{val}"' if isinstance(val, str) else str(val)
        grouped[cat].append((key, val_str))

    # 정렬: 카테고리 내부는 알파벳 순
    for cat in grouped:
        grouped[cat] = sorted(grouped[cat], key=lambda x: x[0])

    # 전체 최대 폭 계산
    all_items = [(k, v) for items in grouped.values() for (k, v) in items]
    max_key_len = max(len(k) for k, _ in all_items + [("Hyperparameter", "")])
    max_val_len = max(len(v) for _, v in all_items + [("", "Value")])
    total_width = max_key_len + max_val_len + 7

    # Box-drawing
    h = "─"
    v = "│"
    tl, tr, bl, br = "┌", "┐", "└", "┘"
    l_sep, r_sep = "├", "┤"
    t_sep, b_sep = "┬", "┴"

    h1 = h * (max_key_len + 2)
    h2 = h * (max_val_len + 2)

    output = io.StringIO()
    write = lambda line="": output.write(line + "\n")

    # Header
    write(f"{tl}{h1}{t_sep}{h2}{tr}")
    write(f"{v} {'Hyperparameter':<{max_key_len}} {v} {'Value'.center(max_val_len)} {v}")

    for cat in categories:
        if cat not in grouped:
            continue

        write(f"{l_sep}{h1}{b_sep}{h2}{r_sep}")

        label = f"[ {cat} ]"
        label_line = f"{v}{label:^{total_width - 2}}{v}"
        write(label_line)
        write(f"{l_sep}{h1}{t_sep}{h2}{r_sep}")

        for k, v_ in grouped[cat]:
            write(f"{v} {k:<{max_key_len}} {v} {v_:<{max_val_len}} {v}")

    write(f"{bl}{h1}{b_sep}{h2}{br}")
    return output.getvalue()
