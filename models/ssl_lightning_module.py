import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from models import CrossModalSSL
from data_provider.data_utils import NUM_ATOM_TYPES
from models.modules import ScalarToDistanceModule, VectorToAtomLogitsModule


class SSLModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.model = CrossModalSSL(args)
        self.head_atom = VectorToAtomLogitsModule(args.d_vector, NUM_ATOM_TYPES)
        self.head_dist = ScalarToDistanceModule(args.d_scalar)
        self.train_log_kwargs = dict(on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.val_log_kwargs = dict(on_epoch=True, sync_dist=True, logger=True)

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        s, v = self(batch)
        pred_atoms = self.head_atom(v)
        pred_dist = self.head_dist(s, batch.edge_index)
        loss_atom = F.cross_entropy(pred_atoms[batch.mask_atom], batch.target_atom[batch.mask_atom])
        loss_dist = F.l1_loss(pred_dist[batch.mask_edge], batch.target_edge_len[batch.mask_edge])
        loss = loss_atom + self.args.lambda_dist * loss_dist
        self.log("train/total", loss, batch_size=batch.num_graphs, **self.train_log_kwargs)
        self.log("train/atom", loss_atom, batch_size=batch.num_graphs, **self.train_log_kwargs)
        self.log("train/dist", loss_dist, batch_size=batch.num_graphs, **self.train_log_kwargs)

        return loss

    def validation_step(self, batch, batch_idx):
        s, v = self(batch)
        pred_atoms = self.head_atom(v)
        pred_dist = self.head_dist(s, batch.edge_index)
        loss_atom = F.cross_entropy(pred_atoms[batch.mask_atom], batch.target_atom[batch.mask_atom])
        loss_dist = F.l1_loss(pred_dist[batch.mask_edge], batch.target_edge_len[batch.mask_edge])
        loss = loss_atom + self.args.lambda_dist * loss_dist
        self.log("valid/total", loss, batch_size=batch.num_graphs, prog_bar=True, **self.val_log_kwargs)
        self.log("valid/atom", loss_atom, batch_size=batch.num_graphs, **self.val_log_kwargs)
        self.log("valid/dist", loss_dist, batch_size=batch.num_graphs, **self.val_log_kwargs)

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
