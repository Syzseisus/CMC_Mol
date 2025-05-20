import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError
from pytorch_lightning import LightningModule
from torchmetrics.classification import BinaryAUROC, MultilabelAccuracy

from models import CrossModalFT
from models.modules import build_modular_head


class FTModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.model = CrossModalFT(args)
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
        if self.args.freeze_pt:  # 정리하면서 추가한 거라 기존 파일에서 확인 불가 ..
            for p in self.model.parameters():
                p.requires_grad_(False)
        self.fusion_head = build_modular_head(args, self.args.num_classes)
        if self.args.task_type == "regression":
            assert self.args.num_classes == 1, f"I got 'regression' as a `task_type`, but multiple `num_classes`."
            self.metric_name = "MSE"
            self.task_loss_fn = nn.MSELoss()
            self.train_metric_fn = MeanSquaredError()
            self.valid_metric_fn = MeanSquaredError()
            self.test_metric_fn = MeanSquaredError()
        elif self.args.task_type == "classification":
            self.metric_name = "AUROC"
            self.task_loss_fn = nn.BCEWithLogitsLoss()
            if self.args.num_classes == 1:
                self.train_metric_fn = BinaryAUROC()
                self.valid_metric_fn = BinaryAUROC()
                self.test_metric_fn = BinaryAUROC()
            else:
                self.train_metric_fn = MultilabelAccuracy(num_labels=self.args.num_classes, average="macro")
                self.valid_metric_fn = MultilabelAccuracy(num_labels=self.args.num_classes, average="macro")
                self.test_metric_fn = MultilabelAccuracy(num_labels=self.args.num_classes, average="macro")

        else:
            raise ValueError(f"`task_type` must be one of 'regression' or 'classification'. (Got: {self.task_type})")
        self.train_log_kwargs = dict(on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.valid_log_kwargs = dict(on_epoch=True, sync_dist=True, logger=True)
        self.metric_log_kwargs = dict(on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

    #! 수상한 부분 (2/2)
    def format_outputs_and_targets(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds: [B, num_tasks]
        targets: [B, num_tasks]
        """
        if self.args.task_type == "classification":
            if self.args.num_classes == 1:
                # Binary classification: keep raw logits + float targets
                targets = targets.float().view(-1)
                preds = preds.view(-1)  # shape [B]
                return preds, targets
            else:
                # Multiclass classification: use argmax
                preds = preds.float()
                targets = targets.float()
                return preds, targets

        elif self.args.task_type == "regression":
            return preds, targets.float()

        else:
            raise ValueError(f"Unsupported task type: {self.args.task_type}")

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
        preds, targets = self.format_outputs_and_targets(g, batch.y)
        loss = self.task_loss_fn(preds, targets)
        self.train_metric_fn.update(preds, targets)
        self.log("train/loss", loss, batch_size=batch.num_graphs, **self.train_log_kwargs)
        return loss

    def on_training_epoch_end(self):
        metric = self.train_metric_fn.compute()
        self.log(f"train/{self.metric_name}", metric, **self.metric_log_kwargs)

    def on_validation_epoch_start(self):
        self.valid_metric_fn.reset()

    def validation_step(self, batch, batch_idx):
        s, v = self(batch)
        g = self.fusion_head(s, v, batch.batch)
        preds, targets = self.format_outputs_and_targets(g, batch.y)
        loss = self.task_loss_fn(preds, targets)
        self.valid_metric_fn.update(preds, targets)
        self.log("valid/loss", loss, batch_size=batch.num_graphs, **self.valid_log_kwargs)
        return loss

    def on_validation_epoch_end(self):
        metric = self.valid_metric_fn.compute()
        self.log(f"valid/{self.metric_name}", metric, **self.metric_log_kwargs)
        self.log(f"valid_{self.metric_name}", metric, prog_bar=False, **self.valid_log_kwargs)

    def on_test_epoch_start(self):
        self.test_metric_fn.reset()

    def test_step(self, batch, batch_idx):
        s, v = self(batch)
        g = self.fusion_head(s, v, batch.batch)
        preds, targets = self.format_outputs_and_targets(g, batch.y)
        self.test_metric_fn.update(preds, targets)

    def on_test_epoch_end(self):
        metric = self.test_metric_fn.compute()
        self.log(f"test/{self.metric_name}", metric, **self.metric_log_kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.max_epochs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "valid_loss",  # needed if using ReduceLROnPlateau
            },
        }
