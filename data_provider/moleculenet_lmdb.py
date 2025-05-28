import os
import time
import tqdm
import pickle
import argparse

import lmdb
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet

from data_provider.data_utils import add_attr_to_moleculnet
from data_provider.moleculenet_label import label_preprocessor
from data_provider.moleculenet_stats import TASK_TYPE, REGRESSION, CLASSIFICATION


class MoleculeNet_LMDBWriter:
    """
    PyG의 `MoleculeNet` 데이터셋에는 `x`, `edge_index`, `edge_attr`, `smiles`, `y`가 있음
    우리는 여기에 3D 정보도 필요하기 때문에
        1. `dataset` 인자로 PyG를 받아와서
        2. `add_attr_to_moleculnet` 함수를 통해 3D 정보를 추가함
        - 실제로는 PyG의 `smiles`와 `y`만 사용함
    """

    def __init__(self, dataset, lmdb_path, batch_size=1000):
        self.dataset = dataset
        self.TOTAL = len(self.dataset)
        self.batch_size = batch_size
        self.lmdb_path = lmdb_path

        if os.path.exists(self.lmdb_path):
            print(f"[INFO] LMDB path '{self.lmdb_path}' already exists. Skipping LMDB creation.")
            self.env = None
            self.skip_writing = True
        else:
            os.makedirs(os.path.dirname(self.lmdb_path) or ".", exist_ok=True)
            self.env = lmdb.open(self.lmdb_path, map_size=1 << 40)
            self.skip_writing = False

    def write(self):
        if self.skip_writing:
            print("[INFO] Skipping write operation because LMDB already exists.")
            return
        count = 0
        buffer = []
        txn = self.env.begin(write=True)
        for idx, data in tqdm.tqdm(enumerate(self.dataset), total=self.TOTAL):
            try:
                smiles = data.smiles
                y = data.y.clone()
                data = add_attr_to_moleculnet(smiles, y)

                buffer.append((f"{count}".encode(), pickle.dumps(data, protocol=4)))
                count += 1

                # flush buffer every batch_size entries
                if len(buffer) >= self.batch_size:
                    for k, v in buffer:
                        txn.put(k, v)
                    txn.commit()
                    txn = self.env.begin(write=True)
                    buffer = []

            except Exception as e:
                print(f"[SKIP] Invalid molecule at index {idx}: {e}")
                raise e  # for debug
                # continue  # for real

        # flush remaining
        if buffer:
            for k, v in buffer:
                txn.put(k, v)
            txn.commit()
        else:
            txn.abort()

        # Save total length
        with self.env.begin(write=True) as txn:
            txn.put(b"length", str(count).encode())

        print(f"[DONE] Total {count:,} molecules written to {self.lmdb_path}")


class MoleculeNet_LMDBDataset(Dataset):
    """
    위의 `MoleculeNet_LMDBWriter`로 생성된 데이터셋을 load함
        - `moleculenet_lmdb.log` : 생성 시의 로그 + Split 기준에 따른 결과
        - `moleculent_stats.py`  : 데이터셋 정보. MoleculeNet 논문의 표를 옮김.
                                   baseline에 regression이 없는 게 많아서 classification만 썼음.
    """

    def __init__(self, lmdb_path: str, dataset_name: str):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.dataset_name = dataset_name
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            cursor = txn.cursor()
            self.keys = [key for key, _ in cursor if key != b"length"]
            self.keys = sorted(self.keys, key=lambda k: int(k.decode()))  # 보장: '0', '1', ...
        self.length = len(self.keys)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Data:
        key = self.keys[idx]
        with self.env.begin() as txn:
            data_bytes = txn.get(key)
            if data_bytes is None:
                raise KeyError(f"Key {idx} not found in LMDB.")
        data = pickle.loads(data_bytes)
        return data


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="MoleculeNet → LMDB 변환기")
    parser.add_argument("--save_dir", default="/workspace/DATASET/MoleculeNet", help="저장 경로")
    parser.add_argument("--debug", action="store_true", help="디버그 모드: 대표 데이터셋만 저장")
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()
    # fmt: on

    # 생성
    print(f"[INFO] 저장 경로  : {args.save_dir}")
    dataset_names = ["bbbp", "esol"] if args.debug else list(TASK_TYPE.keys())
    NUM_DATASETS = len(dataset_names)

    total = 0
    lmdb_path_list = []
    for i, dn in enumerate(dataset_names):
        # 생성
        print("=" * 50)
        print(f"dataset : {dn} ({i + 1} / {NUM_DATASETS}) 처리 중")
        print("Step 1: PyG `MoleculeNet` dataset 생성")
        pyg = MoleculeNet(args.save_dir, dn)
        print(f"{' Validate ':=^50}")
        info = CLASSIFICATION[dn] if TASK_TYPE[dn] == "classification" else REGRESSION[dn]
        y = pyg.y.shape[-1] if len(pyg.y.shape) == 2 else 1 if len(pyg.y.shape) == 1 else -1
        assert info["num_tasks"] == y, f"tasks diff: {info['num_tasks']} | {pyg.y.shape}"
        assert info["num_data"] == len(pyg), f"data diff: {info['num_data']} | {len(pyg)}"
        print(f"항목    : {'GT':^6} | {'PyG':^6}")
        print(f"# Tasks : {info['num_tasks']:^6} | {y:^6}")
        print(f"# Data  : {info['num_data']:^6} | {len(pyg):^6}")
        lmdb_path = os.path.join(pyg.processed_dir, "lmdb")
        lmdb_path_list.append(lmdb_path)

        print("Step 2: LMDB 생성")
        print(f"[INFO] LMDB 경로   : {lmdb_path}")
        start = time.time()
        writer = MoleculeNet_LMDBWriter(pyg, lmdb_path, args.batch_size)
        writer.write()
        elapsed = time.time() - start
        print(f"[DONE] LMDB 저장 완료. 소요 시간: {elapsed/60:.2f}분")
    print(f"[DONE] 모두 저장 완료. 소요 시간: {total/60:.2f}분")
    print("=" * 50, "\n")

    # 로드
    for dn, lmdb_path in zip(dataset_names, lmdb_path_list):
        print(f"{' Test Load ':=^50}")
        print(f"dataset : {dn}")
        start = time.time()
        dataset = MoleculeNet_LMDBDataset(lmdb_path)
        print(f"{' Validate ':=^50}")
        info = CLASSIFICATION[dn] if TASK_TYPE[dn] == "classification" else REGRESSION[dn]
        y = dataset[0].y.shape[-1] if len(dataset[0].y.shape) == 2 else 1 if len(dataset[0].y.shape) == 1 else -1
        assert info["num_tasks"] == y, f"tasks diff: {info['num_tasks']} | {dataset[0].y.shape}"
        assert info["num_data"] == len(dataset), f"data diff: {info['num_data']} | {len(dataset)}"
        print(f"  항목  : {'GT':^6} | {'LMDB':^6}")
        print(f"# Tasks : {info['num_tasks']:^6} | {y:^6}")
        print(f"# Data  : {info['num_data']:^6} | {len(dataset):^6}")
        print(dataset[0])
        print(f"소요 시간: {time.time() - start}초")
