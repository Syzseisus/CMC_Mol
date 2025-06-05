import os
import pickle
from pathlib import Path

import lmdb


def collect_all_lmdb_paths(base_dir):
    part_paths = []
    for entry in sorted(Path(base_dir).iterdir()):
        if entry.is_dir() and (entry / "data.mdb").exists():
            if entry.name.startswith("part_") or entry.name.startswith("remain_"):
                part_paths.append(str(entry))
    return part_paths


def merge_lmdb_parts(output_path: str, part_paths: list):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    final_env = lmdb.open(output_path, map_size=1 << 40)
    count = 0
    with final_env.begin(write=True) as final_txn:
        for part_path in part_paths:
            print(f"[MERGE] Reading from {part_path}")
            env = lmdb.open(part_path, readonly=True, lock=False)
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    if key == b"length":
                        continue
                    try:
                        obj = pickle.loads(value)
                    except Exception as e:
                        print(f"[SKIP] Invalid molecule at index {count}: {e}")
                        continue
                    if not isinstance(obj, dict):
                        continue
                    final_txn.put(f"{count}".encode(), value)
                    count += 1
            env.close()
        final_txn.put(b"length", str(count).encode())
    print(f"[MERGED] Total {count:,} entries written to {output_path}")


if __name__ == "__main__":
    base_lmdb_dir = "/workspace/DATASET/PCQM4M_V2/pcqm4mv2_lmdb_aug_shrinked"
    output_lmdb_path = os.path.join(base_lmdb_dir, "pcqm4mv2_lmdb_aug_5")
    part_paths = collect_all_lmdb_paths(base_lmdb_dir)
    print(f"[INFO] Found {len(part_paths)} LMDB parts.")
    merge_lmdb_parts(output_lmdb_path, part_paths)
