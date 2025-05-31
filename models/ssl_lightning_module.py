import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from models import CrossModalSSL
from data_provider.data_utils import allowable_features
from models.modules import ScalarToBondFeatureModule, VectorToFullAtomFeatureModule


class SSLModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.model = CrossModalSSL(args)
        self.head_atom = VectorToFullAtomFeatureModule(args.d_vector, args.dropout)
        self.head_bond = ScalarToBondFeatureModule(args.d_scalar, args.dropout)
        self.train_log_kwargs = dict(on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.train_log_kwargs_anal = dict(on_step=True, on_epoch=True, sync_dist=True, prog_bar=False, logger=True)
        self.val_log_kwargs = dict(on_epoch=True, sync_dist=True, logger=True)

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        s, v = self(batch)
        atom_logit_list = self.head_atom(v)
        bond_logit_list = self.head_bond(s, batch.edge_index)

        atom_targets = batch.x
        bond_targets = [batch.edge_len, batch.edge_type]

        loss_atom_list = []
        for c, logits in enumerate(atom_logit_list):
            loss_atom_list.append(F.cross_entropy(logits[batch.mask_atom, c], atom_targets[batch.mask_atom, c]))
        loss_atom = sum(loss_atom_list) / len(atom_logit_list)

        loss_bond_dist = F.l1_loss(bond_logit_list[0][batch.mask_edge], bond_targets[0][batch.mask_edge])
        loss_bond_type = F.cross_entropy(bond_logit_list[1][batch.mask_edge], bond_targets[1][batch.mask_edge])
        loss_bond = self.args.lambda_bond_dist * loss_bond_dist + loss_bond_type

        loss = self.args.lambda_atom * loss_atom + self.args.lambda_bond * loss_bond

        self.log("train/total", loss, batch_size=batch.num_graphs, **self.train_log_kwargs)
        self.log("train/atom", loss_atom, batch_size=batch.num_graphs, **self.train_log_kwargs)
        self.log("train/bond", loss_bond, batch_size=batch.num_graphs, **self.train_log_kwargs)
        for c, loss in enumerate(loss_atom_list):
            loss_type = list(allowable_features.keys())[c][9:-6]  # extract * from "possible_*_list"
            self.log(f"train/atom_{loss_type}", loss, batch_size=batch.num_graphs, **self.train_log_kwargs_anal)
        self.log(f"train/bond_dist", loss_bond_dist, batch_size=batch.num_graphs, **self.train_log_kwargs_anal)
        self.log(f"train/bond_type", loss_bond_type, batch_size=batch.num_graphs, **self.train_log_kwargs_anal)

        return loss

    def validation_step(self, batch, batch_idx):
        s, v = self(batch)
        atom_logit_list = self.head_atom(v)
        bond_logit_list = self.head_bond(s, batch.edge_index)

        atom_targets = batch.x
        bond_targets = [batch.edge_len, batch.edge_type]

        loss_atom_list = []
        for c, logits in enumerate(atom_logit_list):
            loss_atom_list.append(F.cross_entropy(logits[batch.mask_atom, c], atom_targets[batch.mask_atom, c]))
        loss_atom = sum(loss_atom_list) / len(atom_logit_list)

        loss_bond_dist = F.l1_loss(bond_logit_list[0][batch.mask_edge], bond_targets[0][batch.mask_edge])
        loss_bond_type = F.cross_entropy(bond_logit_list[1][batch.mask_edge], bond_targets[1][batch.mask_edge])
        loss_bond = self.args.lambda_bond_dist * loss_bond_dist + loss_bond_type

        loss = self.args.lambda_atom * loss_atom + self.args.lambda_bond * loss_bond

        self.log("valid/total", loss, batch_size=batch.num_graphs, prog_bar=True, **self.val_log_kwargs)
        self.log("valid/atom", loss_atom, batch_size=batch.num_graphs, **self.val_log_kwargs)
        self.log("valid/bond", loss_bond, batch_size=batch.num_graphs, **self.val_log_kwargs)
        for c, loss in enumerate(loss_atom_list):
            loss_type = list(allowable_features.keys())[c][9:-6]  # extract * from "possible_*_list"
            self.log(f"valid/atom_{loss_type}", loss, batch_size=batch.num_graphs, **self.val_log_kwargs_anal)
        self.log(f"valid/bond_dist", loss_bond_dist, batch_size=batch.num_graphs, **self.val_log_kwargs_anal)
        self.log(f"valid/bond_type", loss_bond_type, batch_size=batch.num_graphs, **self.val_log_kwargs_anal)
        # for monitoring
        self.log("valid_total", loss, batch_size=batch.num_graphs, **self.val_log_kwargs)

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
