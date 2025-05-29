import os
import json
import wandb

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from utils import get_args, format_args
from models.ssl_lightning_module import SSLModule
from data_provider.pcqm4mv2_module import PCQM4MV2DM

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
    dm = PCQM4MV2DM(args)
    model = SSLModule(args)

    # ===== WandB Logger =====
    wandb_logger = WandbLogger(project=args.project, save_dir=args.log_dir)

    # ===== Callbacks =====
    checkpoint = ModelCheckpoint(
        monitor="valid_total",
        mode="min",
        save_last=True,
        filename="{epoch}-{valid_total:.4f}",
        every_n_epochs=10,
        dirpath=args.ckpt_dir,
        save_top_k=-1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ===== Trainer =====
    num_devices = torch.cuda.device_count()
    trainer = Trainer(
        accelerator="gpu" if num_devices > 0 else "cpu",
        devices=num_devices if num_devices > 0 else 1,
        strategy="ddp" if num_devices > 1 else "auto",
        # strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[checkpoint, lr_monitor],
        num_sanity_val_steps=1,  # validation monitor 잘 되도록
        check_val_every_n_epoch=5,
    )

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        wandb.termlog(format_args(args, categories))

    # ===== Training =====
    trainer.fit(model, datamodule=dm)

    # ===== logging =====
    # 마지막 에폭 정보
    last_epoch = trainer.current_epoch
    last_train_loss = trainer.callback_metrics.get("train/total", None)
    last_valid_loss = trainer.callback_metrics.get("valid/total", None)

    # best ckpt 기준 정보
    best_score = checkpoint.best_model_score  # usually val_loss

    # best 에폭을 추정할 수 있는 경우
    best_epoch = getattr(checkpoint, "best_model_epoch", None)  # Lightning >=2.1

    # summary 딕셔너리 구성
    summary = {
        "pretrain/epoch": last_epoch,
        "pretrain/train_loss": float(last_train_loss) if last_train_loss else None,
        "pretrain/valid_loss": float(last_valid_loss) if last_valid_loss else None,
        "pretrain/best_epoch": best_epoch,
        "pretrain/best_valid_loss": float(best_score) if best_score else None,
    }

    # args에 들어있는 하이퍼파라미터 저장
    for k in ["lr", "wd", "lambda_dist", "d_scalar", "d_vector", "num_layers"]:
        v = getattr(args, k, None)
        if v is not None:
            summary[f"hparam/{k}"] = v

    # 결과 저장
    summary_path = os.path.join(args.save_dir, "pretrain", "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    args, categories = get_args()
    main(args, categories)
