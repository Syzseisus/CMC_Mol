from wandb import Histogram

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import Metric, MeanSquaredError
from torchmetrics.functional.classification import auroc

from models import CrossModalFT
from models.modules import build_modular_head


class MaskedMultilabelAUROC(Metric):
    def __init__(self, num_labels, average="macro", thresholds=None, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.average = average
        self.thresholds = thresholds

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds: Tensor of shape (N, L), with probabilities or logits.
        targets: Tensor of shape (N, L), with values 0, 1, or NaN.
        """
        self.preds.append(preds.detach())
        self.targets.append(targets.detach())

    def compute(self):
        preds = torch.cat(self.preds, dim=0)
        targets = torch.cat(self.targets, dim=0)

        aucs = []
        for i in range(self.num_labels):
            y_true = targets[:, i]
            y_pred = preds[:, i]

            mask = ~torch.isnan(y_true)
            if mask.sum() == 0:
                continue  # skip if no valid labels

            y_true_valid = y_true[mask].long()
            y_pred_valid = y_pred[mask]

            auc = auroc(y_pred_valid, y_true_valid, task="binary", thresholds=self.thresholds)
            aucs.append(auc)

        if len(aucs) == 0:
            return torch.tensor(float("nan"))

        return torch.stack(aucs).mean() if self.average == "macro" else aucs


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
            if not self.args.full_ft:
                for p in self.model.parameters():
                    p.requires_grad_(False)
        else:
            print(f"{' Randomly initialize the model parameters. ':=^80}")
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
            self.train_metric_fn = MaskedMultilabelAUROC(num_labels=self.args.num_classes)
            self.valid_metric_fn = MaskedMultilabelAUROC(num_labels=self.args.num_classes)
            self.test_metric_fn = MaskedMultilabelAUROC(num_labels=self.args.num_classes)
        else:
            raise ValueError(f"`task_type` must be one of 'regression' or 'classification'. (Got: {self.task_type})")
        self.train_log_kwargs = dict(on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.valid_log_kwargs = dict(on_epoch=True, sync_dist=True, logger=True)
        self.metric_log_kwargs = dict(on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

        if self.args.log_preds:
            self.test_preds = []
            self.test_targets = []

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
        loss = self.task_loss_fn(g, batch.y)
        self.train_metric_fn.update(g, batch.y)
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
        loss = self.task_loss_fn(g, batch.y)
        self.valid_metric_fn.update(g, batch.y)
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
        self.test_metric_fn.update(g, batch.y)
        if self.args.log_preds:
            self.test_preds.append(g.detach().cpu())
            self.test_targets.append(batch.y.detach().cpu())

    def on_test_epoch_end(self):
        metric = self.test_metric_fn.compute()
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
            self.log(f"test/{self.metric_name}_preds", preds, **self.metric_log_kwargs)

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
