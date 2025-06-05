import os
import time
import tqdm
import pickle
import argparse

import lmdb
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data

from data_provider.data_utils import apply_mask, mol_to_pyg_data_gt, mol_to_pyg_data_aug_list

TOTAL = 3378606


class PCQM4Mv2_LMDBWriter:
    """
    `RDKit.Mol` 객체를 갖고 있는 `.sdf` 파일을 읽어서 LMDB로 만듦.
    이 객체에는 이미 GT 좌표가 있음.
    근데 2D에는 GT 좌표가 없으니까, 라이브러리로 생성한 좌표 (`aug`)를 추가해서 보완함.
    """

    def __init__(self, args):
        self.sdf_path = args.sdf_path
        self.lmdb_path = args.lmdb_path
        os.makedirs(os.path.dirname(self.lmdb_path) or ".", exist_ok=True)
        self.env = lmdb.open(self.lmdb_path, map_size=1 << 40)

        self.num_conf = args.num_conf
        self.calc_heavy_mol = args.calc_heavy_mol
        self.idx = args.idx
        self.total_parts = args.total_parts
        self.batch_size = args.batch_size
        self.multi_conf = args.multi_conf
        self.TOTAL = TOTAL // self.total_parts

    def write(self):
        suppl = Chem.SDMolSupplier(self.sdf_path)
        count = 0
        buffer = []
        with self.env.begin(write=True) as txn:
            for idx, mol in tqdm.tqdm(enumerate(suppl), total=self.TOTAL, desc=f"Writing part {self.idx}"):
                if mol is None or idx % self.total_parts != self.idx:
                    # 병렬처리하려고 지정해둠.
                    continue
                try:
                    data = mol_to_pyg_data_gt(mol)
                    aug_list = mol_to_pyg_data_aug_list(mol, self.num_conf, self.calc_heavy_mol, self.multi_conf)

                    buffer.append((f"{count}".encode(), pickle.dumps(data, protocol=4)))
                    count += 1
                    for aug in aug_list:
                        buffer.append((f"{count}".encode(), pickle.dumps(aug, protocol=4)))
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

            # Save total length
            with self.env.begin(write=True) as txn:
                txn.put(b"length", str(count).encode())

            print(f"[DONE] Total {count:,} molecules written to {self.lmdb_path}")


class PCQM4Mv2_LMDBDataset(Dataset):
    """
    위에 `PCQM4Mv2_LMDBWriter` 클래스로 만든 LMDB 파일을 로드하는 클래스
    애석하게도 로그는 없음
    """

    def __init__(self, lmdb_path: str, mask_ratio: float = 0.15):
        self.lmdb_path = lmdb_path
        self.mask_ratio = mask_ratio
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.length = int(txn.get("length".encode()).decode())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            data_bytes = txn.get(f"{idx}".encode())
        data = pickle.loads(data_bytes)
        data = Data.from_dict(data)
        return apply_mask(data, self.mask_ratio)
        # `apply_mask`:
        #   - `data.mask_atom` : 마스킹할 원소 인덱스
        #   - `data.mask_edge` : 마스킹할 엣지 인덱스


def merge_lmdb_parts(output_path: str, part_paths: list):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    final_env = lmdb.open(output_path, map_size=1 << 40)
    count = 0
    with final_env.begin(write=True) as final_txn:
        for part_path in part_paths:
            env = lmdb.open(part_path, readonly=True, lock=False)
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    if key == b"length":
                        continue
                    final_txn.put(f"{count}".encode(), value)
                    count += 1
            env.close()
        final_txn.put(b"length", str(count).encode())
    print(f"[MERGED] Total {count:,} entries written to {output_path}")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="PCQM4Mv2 SDF → LMDB 변환기")
    parser.add_argument("--sdf_path", default="/workspace/DATASET/PCQM4M_V2/pcqm4m-v2-train.sdf", help="입력 SDF 파일 경로")
    parser.add_argument("--lmdb_path", type=str, default="/workspace/DATASET/PCQM4M_V2/pcqm4mv2_lmdb_aug", help="저장할 LMDB 경로")
    parser.add_argument("--num_conf", type=int, default=10)
    parser.add_argument("--calc_heavy_mol", action="store_true", default=False)
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--total_parts", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--merge", action="store_true", default=False)
    args = parser.parse_args()
    # fmt: on

    assert (
        args.idx < args.total_parts
    ), f"`idx` should be less than `total_parts`. (Got idx={args.idx}, total_parts={args.total_parts})"

    if args.merge:
        # # 병합
        # print("전체 part 하나로 합치는 중")
        # part_paths = [f"{args.lmdb_path}_{i}" for i in range(args.total_parts)]
        args.lmdb_path = f"{args.lmdb_path}_merged"
        # merge_lmdb_parts(args.lmdb_path, part_paths)
        # print("완료")

        # 로드
        print("시험 로딩")
        start = time.time()
        dataset = PCQM4Mv2_LMDBDataset(args.lmdb_path)
        print(dataset[0], f"소요 시간: {time.time() - start}초")

    else:
        # 생성
        print(f"{args.idx + 1} 번째 part 처리 중")
        args.lmdb_path = f"{args.lmdb_path}_{args.idx}"
        print(f"[INFO] SDF 경로    : {args.sdf_path}")
        print(f"[INFO] LMDB 경로   : {args.lmdb_path}")

        start = time.time()
        writer = PCQM4Mv2_LMDBWriter(args)
        writer.write()
        elapsed = time.time() - start
        print(f"[DONE] LMDB 저장 완료. 소요 시간: {elapsed/60:.2f}분")
