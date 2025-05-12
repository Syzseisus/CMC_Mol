from torch.utils.data.dataset import Subset
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule

from utils import split_dataset
from data_provider.pcqm4mv2_lmdb import PCQM4Mv2_LMDBDataset


class PCQM4MV2DM(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sdf_path = self.args.sdf_path
        self.lmdb_path = self.args.lmdb_path
        self.root = self.args.root
        self.mask_ratio = self.args.mask_ratio

        self.batch_size = args.batch_size
        self.loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.args.prefetch_factor if self.args.num_workers else None,
            timeout=0,
        )

    def setup(self, stage=None):
        full_dataset = PCQM4Mv2_LMDBDataset(self.lmdb_path, self.mask_ratio)
        if self.args.limit:
            full_dataset = Subset(full_dataset, range(self.args.limit))
        if stage == "fit" or stage is None:
            # 그냥 random split임.
            # `args.split`으로 뭘 주던 유연하게 작동하는 함수 만들어둠.
            subsets = split_dataset(full_dataset, self.args.split)
            self.train_dataset = subsets[0]
            self.valid_dataset = subsets[1]
            self.test_dataset = subsets[2] if len(subsets) == 3 else None

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, shuffle=False, **self.loader_kwargs)

    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
        else:
            return None
