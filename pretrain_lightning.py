import os
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
    # ===== Print Args =====
    formatted = format_args(args, categories)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(formatted)

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
        # num_sanity_val_steps=0,
    )

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        wandb.termlog(formatted)

    # ===== Training =====
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    args, categories = get_args()
    main(args, categories)
