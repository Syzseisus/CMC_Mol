import math
from wandb import Histogram

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchmetrics import MeanSquaredError
from pytorch_lightning import LightningModule
from torch_geometric.loader import DataLoader

from models import CrossModalFT
from models.modules import FusionHead, GraphCLAUROC


class FTModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.model = CrossModalFT(args)
        if args.pretrain_ckpt != "random_init":
            print(f"{' Load pre-trained model parameters. ':=^80}")
            tmp = torch.load(args.pretrain_ckpt)["state_dict"]
            ckpt = {}
            for k, v in tmp.items():
                if "mask" in k or "head_atom" in k or "head_dist" in k:
                    continue
                if "model" in k:
                    ckpt[k[6:]] = v
                else:
                    ckpt[k] = v
            self.model.load_state_dict(ckpt)
        else:
            print(f"{' Randomly initialize the model parameters. ':=^80}")
        self.fusion_head = FusionHead(args, self.args.num_classes)
        if self.args.task_type == "regression":
            assert self.args.num_classes == 1, f"I got 'regression' as a `task_type`, but multiple `num_classes`."
            self.metric_name = "MSE"
            self.task_loss_fn = nn.MSELoss()
            self.train_metric_fn = MeanSquaredError()
            self.valid_metric_fn = MeanSquaredError()
            self.test_metric_fn = MeanSquaredError()
        elif self.args.task_type == "classification":
            self.metric_name = "AUROC"
            self.task_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
            self.train_metric_fn = GraphCLAUROC(self.args.num_classes)
            self.valid_metric_fn = GraphCLAUROC(self.args.num_classes)
            self.test_metric_fn = GraphCLAUROC(self.args.num_classes)
        else:
            raise ValueError(f"`task_type` must be one of 'regression' or 'classification'. (Got: {self.task_type})")
        self.train_log_kwargs = dict(on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.valid_log_kwargs = dict(on_epoch=True, sync_dist=True, logger=True)
        self.metric_log_kwargs = dict(on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

        if self.args.log_preds:
            self.test_preds = []
            self.test_targets = []

    def setup(self, stage: str):
        if stage == "fit":
            train_dataset = self.trainer.datamodule.train_dataset
            loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)
            labels = [batch.y for batch in loader]
            y = torch.cat(labels, dim=0).to(self.device)

            y_bin = (y == 1).float()
            valid = y**2 > 0
            pos = (y_bin * valid).sum(0)
            neg = valid.sum(0) - pos
            pos_weight = (neg / (pos + 1e-4)).clamp(max=100)
            self.register_buffer("pos_weight", pos_weight)
            self.task_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction="none")

    def forward(self, data):
        return self.model(data)

    # ============================================
    # train, valid, test가 모두 같은 형태여야 함 (test는 loss 계산 제외)
    # ============================================

    def on_train_epoch_start(self):
        self.train_metric_fn.reset()

    def training_step(self, batch, batch_idx):
        s, v = self(batch)
        g = self.fusion_head(s, v, batch.batch)  # [B, num_classes]

        # BCE는 ignore_index 없어서, 일단 0.5 넣고
        loss = self.task_loss_fn(g, (batch.y + 1) / 2)
        is_valid = batch.y**2 > 0
        loss = torch.where(is_valid, loss, torch.zeros_like(loss))  # 0으로 덮기
        loss = torch.sum(loss) / torch.sum(is_valid)

        target = batch.y.clone().long()  # -1 , 0 (==nan), 1 그대로 넣음
        self.train_metric_fn.update(g, target)
        self.log("train/loss", loss, batch_size=batch.num_graphs, **self.train_log_kwargs)
        return loss

    def on_training_epoch_end(self):
        metric, conf_mats, tholds = self.train_metric_fn.compute()
        self.log(f"train/{self.metric_name}", metric, **self.metric_log_kwargs)

    def on_validation_epoch_start(self):
        self.valid_metric_fn.reset()

    def validation_step(self, batch, batch_idx):
        s, v = self(batch)
        g = self.fusion_head(s, v, batch.batch)

        # BCE는 ignore_index 없어서, 일단 0.5 넣고
        loss = self.task_loss_fn(g, (batch.y + 1) / 2)
        is_valid = batch.y**2 > 0
        loss = torch.where(is_valid, loss, torch.zeros_like(loss))  # 0으로 덮기
        loss = torch.sum(loss) / torch.sum(is_valid)

        target = batch.y.clone().long()  # -1, 0 (==nan), 1 그대로 넣음
        self.valid_metric_fn.update(g, target)
        self.log("valid/loss", loss, batch_size=batch.num_graphs, **self.valid_log_kwargs)
        return loss

    def on_validation_epoch_end(self):
        metric, conf_mats, tholds = self.valid_metric_fn.compute()  # Binary - 1D tensor
        self.log(f"valid/{self.metric_name}", metric, **self.metric_log_kwargs)
        self.log(f"valid_{self.metric_name}", metric, prog_bar=False, **self.valid_log_kwargs)

    def on_test_epoch_start(self):
        self.test_metric_fn.reset()

    def test_step(self, batch, batch_idx):
        s, v = self(batch)
        g = self.fusion_head(s, v, batch.batch)

        target = batch.y.clone().long()  # -1, 0 (==nan), 1 그대로 넣음
        self.test_metric_fn.update(g, target)
        if self.args.log_preds:
            self.test_preds.append(g.detach().cpu())
            self.test_targets.append(batch.y.detach().cpu())

    def on_test_epoch_end(self):
        metric, conf_mats, tholds = self.test_metric_fn.compute()
        self.log(f"test/{self.metric_name}", metric, **self.metric_log_kwargs)
        if self.args.log_preds:
            preds = torch.cat(self.test_preds, dim=0).numpy()
            targets = torch.cat(self.test_targets, dim=0).numpy()
            num_labels = preds.shape[1]

            # 전체 예측값 분포 시각화
            flat_preds = preds.flatten()

            self.logger.experiment.log(
                {
                    "test/pred/whole_mean": flat_preds.mean(),
                    "test/pred/whole_std": flat_preds.std(),
                    "test/pred/whole_min": flat_preds.min(),
                    "test/pred/whole_max": flat_preds.max(),
                    "test/pred/whole_hist": Histogram(flat_preds),
                    "test/target/whole_mean": targets.mean(),
                    "test/target/whole_std": targets.std(),
                    "test/target/whole_min": targets.min(),
                    "test/target/whole_max": targets.max(),
                    "test/target/whole_hist": Histogram(targets),
                }
            )
            # 라벨 별 예측값 분포 시각화
            for i in range(num_labels):
                label_preds = preds[:, i]
                label_targets = targets[:, i]
                self.logger.experiment.log(
                    {
                        f"test/pred/label_{i}_mean": label_preds.mean(),
                        f"test/pred/label_{i}_std": label_preds.std(),
                        f"test/pred/label_{i}_min": label_preds.min(),
                        f"test/pred/label_{i}_max": label_preds.max(),
                        f"test/pred/label_{i}_hist": Histogram(label_preds),
                        f"test/target/label_{i}_mean": label_targets.mean(),
                        f"test/target/label_{i}_std": label_targets.std(),
                        f"test/target/label_{i}_min": label_targets.min(),
                        f"test/target/label_{i}_max": label_targets.max(),
                        f"test/target/label_{i}_hist": Histogram(label_targets),
                    }
                )
            self.test_preds = []
            self.test_targets = []

    def configure_optimizers(self):
        # ===== Optimizer =====
        lr_backbone = self.args.lr
        lr_head = self.args.lr * 10
        params_backbone = [p for p in self.model.parameters() if p.requires_grad]
        params_head = [p for p in self.fusion_head.parameters() if p.requires_grad]
        if self.args.full_ft:
            param_groups = [{"params": params_backbone, "lr": lr_backbone}, {"params": params_head, "lr": lr_head}]
        else:
            param_groups = [{"params": params_head, "lr": lr_head}]
            for p in self.model.parameters():
                p.requires_grad_(False)

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.args.wd)

        # ===== Learning Rate Scheduler =====
        if self.args.lr_scheduler == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.max_epochs)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": "valid_loss"},
            }

        elif self.args.lr_scheduler == "cosine_warmup":
            steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
            max_steps = self.args.max_epochs * steps_per_epoch
            warm_steps = int(self.args.warmup_ratio * max_steps)

            def lr_lambda(step):
                if step < warm_steps:
                    return step / warm_steps
                progress = (step - warm_steps) / (max_steps - warm_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        else:
            raise NotImplementedError(f"Invalid lr_scheduler: {self.args.lr_scheduler}")
