from torch.utils.data.dataset import Subset
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule

from data_provider.moleculenet_stats import METHOD
from data_provider.moleculenet_lmdb import MoleculeNet_LMDBDataset
from data_provider.split_utils import random_split, scaffold_split


class MoleculeNetDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset_name = args.dataset_name
        self.lmdb_path = args.lmdb_path
        self.task_type = args.task_type
        self.num_classes = args.num_classes

        self.batch_size = args.batch_size
        self.loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor if args.num_workers else None,
            timeout=0,
        )

    def setup(self, stage=None):
        full_dataset = MoleculeNet_LMDBDataset(self.lmdb_path, self.dataset_name)
        if self.args.limit:
            full_dataset = Subset(full_dataset, range(self.args.limit))

        # Split 방법은 `./data_provider/moleculenet_split.png` 참고
        original_method = METHOD[self.dataset_name].lower()
        if self.args.split_strat == "default":
            method = original_method
        elif self.args.split_strat == "force_scaffold":
            method = "scaffold"
        elif self.args.split_strat == "force_random":
            method = "random"
        else:
            raise NotImplementedError(f"Invalid split strategy: {self.args.split_strat}")

        if method == "scaffold":
            train_idx, valid_idx, test_idx = scaffold_split(full_dataset, seed=self.args.seed + self.args.fold)
        elif method == "random":
            train_idx, valid_idx, test_idx = random_split(full_dataset, seed=self.args.seed + self.args.fold)
        else:
            raise NotImplementedError(f"Invalid split method: {method}")

        self.train_dataset = Subset(full_dataset, train_idx)
        self.valid_dataset = Subset(full_dataset, valid_idx)
        self.test_dataset = Subset(full_dataset, test_idx)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, shuffle=False, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
