import os
import json
import time
import wandb
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from models.ft_lightning_module import FTModule
from utils import format_args, get_args_inference
from data_provider.moleculenet_module import MoleculeNetDataModule

os.environ["OMP_NUM_THREADS"] = "1"  # RDKit 안정화
torch.set_num_threads(1)  # CPU 연산 겹침 방지

# NVIDIA GeForce RTX 3090
# torch.set_float32_matmul_precision("highest")  # Maximum precision (default) – slowest
# torch.set_float32_matmul_precision("high")  # Slight precision trade-off for better performance
torch.set_float32_matmul_precision("medium")  # More precision trade-off for maximum performance


def run_test(args, ckpt_path, logger):
    # ===== Fix Seed =====
    pl.seed_everything(args.seed, workers=True)

    # ===== Data- and LightningModule =====
    dm = MoleculeNetDataModule(args)
    model = FTModule.load_from_checkpoint(ckpt_path, args=args)

    # ===== Trainer =====
    num_devices = torch.cuda.device_count()
    trainer = Trainer(
        accelerator="gpu" if num_devices > 0 else "cpu",
        devices=num_devices if num_devices > 0 else 1,
        strategy="ddp" if num_devices > 1 else "auto",
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
    )

    # ===== Test Only =====
    test_results = trainer.test(model, datamodule=dm)
    return test_results


if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=0 python inference_lightning.py --finetune_dir /path/to/finetune_dir --log_preds

    이때, `finetune_dir`는 다음과 같은 구조를 가지고 있어야 함
    ex) ../save/cmc_atom_down/20250528-142529_tox21/

    /path/to/finetune_dir/
    ├── checkpoints/
    │   ├── fold_0/
    │   │   ├── epoch={epoch}-valid_{metric_name}={metric_value}.ckpt
    │   │   ├── ...
    │   │   └── last.ckpt
    │   ├── fold_1/
    │   │   ├── epoch={epoch}-valid_{metric_name}={metric_value}.ckpt
    │   │   ├── ...
    │   │   └── last.ckpt
    │   ├── ...
    └── log/wandb/
        ├── debug-internal.log
        ├── debug.log
        ├── latest-run/...
        └── run-{datetime}-{run_id}/
            ├── files/
            │   ├── config.yaml
            │   ├── ...
            ├── logs/...
            ├── tmp/code/...
            └── run-{run_id}.wandb
    """
    args, categories = get_args_inference()
    wandb_logger = WandbLogger(project=args.project, save_dir=args.log_dir)  # 공통 logger 정의

    # ===== Find Fold-wise last, best ckpt =====
    ckpt_folder = os.path.join(args.finetune_dir, "checkpoints")
    fold_ckpt_cand = [f for f in os.listdir(ckpt_folder) if ("fold" in f)]
    fold_ckpt_dirs = [f for f in fold_ckpt_cand if os.path.isdir(os.path.join(ckpt_folder, f))]

    last_paths = [os.path.join(ckpt_folder, f, "last.ckpt") for f in fold_ckpt_dirs]
    best_paths = []
    for fold_dir in fold_ckpt_dirs:
        ckpt_files = [
            f
            for f in os.listdir(os.path.join(ckpt_folder, fold_dir))
            if f.endswith(".ckpt") and f != "last.ckpt"  # last.ckpt 빼고 다 가져옴
        ]

        # 비교할 때 '='을 포함해서 같은 값이면 epoch이 많은 걸 가져옴
        if "AUROC" in ckpt_files[0]:
            best_ckpt = ""
            best_value = -1
            for ckpt in ckpt_files:
                metric_value = float(ckpt.split("AUROC=")[-1].replace(".ckpt", ""))
                if metric_value >= best_value:
                    best_value = metric_value
                    best_ckpt = ckpt
        elif "MSE" in ckpt_files[0]:
            best_ckpt = ""
            best_value = 1e10
            for ckpt in ckpt_files:
                metric_value = float(ckpt.split("MSE=")[-1].replace(".ckpt", ""))
                if metric_value <= best_value:
                    best_value = metric_value
                    best_ckpt = ckpt
        else:
            raise ValueError(f"Cannot find valid metric name in checkpoint file: {ckpt_files[0]}")

        best_paths.append(os.path.join(ckpt_folder, fold_dir, best_ckpt))

    # last_paths랑 best_paths의 fold 순서 명시적으로 같게 만들기
    last_paths.sort(key=lambda x: int(x.split("fold_")[-1].split("/")[0]))
    best_paths.sort(key=lambda x: int(x.split("fold_")[-1].split("/")[0]))
    print("\nCheckpoint paths:")
    print("| fold | last_path |             best_path             |")
    print("|------|-----------|-----------------------------------|")
    for i, (last, best) in enumerate(zip(last_paths, best_paths)):
        print(f"| {i:<4} | {os.path.basename(last):<9} | {os.path.basename(best):<33} |")
    print()

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        wandb.termlog(format_args(args, categories))

    # ===== Test =====
    print(f"{' TEST ':=^50}")
    for k, (last, best) in enumerate(zip(last_paths, best_paths)):
        print(f"\n{f' Fold {k} / {args.k_fold} ':=^80}")

        # 결과를 저장할 리스트
        all_last_results = []
        all_best_results = []

        last_results = run_test(args, last, wandb_logger)
        all_last_results.append(last_results)
        best_results = run_test(args, best, wandb_logger)
        all_best_results.append(best_results)

    # 전체 폴드의 평균 계산
    print(f"\n{' AVERAGE RESULTS ':=^80}")
    print("Last checkpoint results:")
    for metric in all_last_results[0][0].keys():
        if metric.startswith("test/"):
            values = [result[0][metric] for result in all_last_results]
            mean_value = np.mean(values)
            std_value = np.std(values)
            print(f"{metric}: {mean_value:.4f} ± {std_value:.4f}")

    print("\nBest checkpoint results:")
    for metric in all_best_results[0][0].keys():
        if metric.startswith("test/"):
            values = [result[0][metric] for result in all_best_results]
            mean_value = np.mean(values)
            std_value = np.std(values)
            print(f"{metric}: {mean_value:.4f} ± {std_value:.4f}")
    print("=" * 80)

    # json 저장 인자가 있으면 저장
    if hasattr(args, "save_metrics_json") and args.save_metrics_json:
        result_dict = {}

        # 각 메트릭에 대한 결과 처리
        for metric in all_last_results[0][0]:
            if metric.startswith("test/"):
                # 기존 전체 평균
                last_values = [res[0][metric] for res in all_last_results]
                result_dict[f"last/{metric}"] = float(np.mean(last_values))

        for metric in all_best_results[0][0]:
            if metric.startswith("test/"):
                # 기존 전체 평균
                best_values = [res[0][metric] for res in all_best_results]
                result_dict[f"best/{metric}"] = float(np.mean(best_values))

        with open(args.save_metrics_json, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"{f' Save results to {args.save_metrics_json} ':=^80}")
