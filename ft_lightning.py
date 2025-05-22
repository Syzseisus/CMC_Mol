import os
import json
import time
import wandb
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from utils import format_args, get_args_ft
from models.ft_lightning_module import FTModule
from data_provider.moleculenet_module import MoleculeNetDataModule

os.environ["OMP_NUM_THREADS"] = "1"  # RDKit 안정화
torch.set_num_threads(1)  # CPU 연산 겹침 방지

# NVIDIA GeForce RTX 3090
# torch.set_float32_matmul_precision("highest")  # Maximum precision (default) – slowest
# torch.set_float32_matmul_precision("high")  # Slight precision trade-off for better performance
torch.set_float32_matmul_precision("medium")  # More precision trade-off for maximum performance


def main(args, categories):
    # ===== Fix Seed =====
    pl.seed_everything(args.seed, workers=True)

    # ===== Data- and LightningModule =====
    dm = MoleculeNetDataModule(args)
    model = FTModule(args)

    # ===== WandB Logger =====
    wandb_logger = WandbLogger(project=args.project, save_dir=args.log_dir)

    # ===== Callbacks =====
    if args.task_type == "regression":
        metric = "MSE"
        mode = "min"
    elif args.task_type == "classification":
        metric = "AUROC"
        mode = "max"
    checkpoint = ModelCheckpoint(
        monitor=f"valid_{metric}",
        mode=mode,
        save_top_k=args.top_k,
        save_last=True,
        filename="{epoch}-{valid_" + metric + ":.4f}",
        dirpath=args.ckpt_dir,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ===== Trainer =====
    num_devices = torch.cuda.device_count()
    trainer = Trainer(
        accelerator="gpu" if num_devices > 0 else "cpu",
        devices=num_devices if num_devices > 0 else 1,
        strategy="ddp" if num_devices > 1 else "auto",
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[checkpoint, lr_monitor],
        # 시작 전에 validiation 돌려서 검증하는 건데,
        # 어차피 돌아가는 코드니까 `=0` 해놓고 바로 train 들어감
        num_sanity_val_steps=0,
    )

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        wandb.termlog(format_args(args, categories))

    # ===== Training =====
    trainer.fit(model, datamodule=dm)

    # ===== Test =====
    print(f"{' LAST ':=^50}")
    last_results = trainer.test(ckpt_path="last", datamodule=dm)
    print(f"{' BEST ':=^50}")
    best_results = trainer.test(ckpt_path="best", datamodule=dm)

    return last_results, best_results


if __name__ == "__main__":
    args, categories = get_args_ft()

    # 결과를 저장할 리스트
    all_last_results = []
    all_best_results = []

    # 시간 로깅
    start_time = time.time()
    for k in range(args.k_fold):
        args.fold = k
        categories.setdefault("Experiment Meta", []).extend(["seed"])
        print(f"\n===== Fold {k} / {args.k_fold} =====")

        last_results, best_results = main(args, categories)
        all_last_results.append(last_results)
        all_best_results.append(best_results)

        # 시간 로깅
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        print(f"{f' Fold {k} took {hours:02d}:{minutes:02d}:{seconds:02d} ':=^80}")
        start_time = time.time()  # Reset timer for next fold

    # 전체 폴드의 평균 계산
    print(f"\n{' FINAL RESULTS ':=^50}")
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
