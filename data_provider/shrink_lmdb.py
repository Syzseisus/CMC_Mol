import pickle
import argparse
from tqdm import tqdm
from pathlib import Path

import lmdb

# === Config ===
KEEP_INDICES = set([-1, 0, 1, 2, 3, 4])
REQUIRED_KEYS = ["x", "coord", "edge_vec", "edge_len", "edge_index", "edge_features", "target_atom", "target_edge_len", "num_nodes", "smiles", "coord_method", "aug_index"]  # fmt: skip


def shrink_lmdb(input_path, output_path, map_size=(1 << 40)):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    part_number = input_path.name.split("_")[-1]
    print(f"[START] Shrinking - Part {part_number}", flush=True)
    print(f"Input path: {input_path}", flush=True)
    print(f"Output path: {output_path}", flush=True)
    print(f"Keeping entries with aug_index in {sorted(KEEP_INDICES)}", flush=True)

    env_in = lmdb.open(str(input_path), readonly=True, lock=False, readahead=False)
    env_out = lmdb.open(str(output_path), map_size=map_size)

    smiles_counts = {}
    entries_checked = 0
    invalid_entries = 0
    total_entries = 0

    with env_in.begin() as txn_in:
        cursor = txn_in.cursor()
        print(f"Estimating number of entries in LMDB...", flush=True)
        total_entries = sum(1 for _ in cursor)  # <= 빠르게 entry 수 추산

    with env_in.begin() as txn_in:
        cursor = txn_in.cursor()
        pbar = tqdm(desc=f"Counting entries - Part {part_number}", total=total_entries, dynamic_ncols=True)
        processed = 0
        for key, value in cursor:
            processed += 1
            if processed % 10000 == 0:
                pbar.update(10000)
            try:
                data = pickle.loads(value)
            except Exception:
                invalid_entries += 1
                continue
            if not all(k in data for k in REQUIRED_KEYS):
                invalid_entries += 1
                continue
            smiles = data["smiles"]
            if smiles not in smiles_counts:
                smiles_counts[smiles] = set()
            smiles_counts[smiles].add(data["aug_index"])
        remaining = processed % 10000
        if remaining:
            pbar.update(remaining)
        pbar.close()

    print(f"Total entries in input LMDB: {total_entries}", flush=True)
    print(f"Unique molecules detected: {len(smiles_counts)}", flush=True)

    smiles_written = set()
    count_written = 0
    written_entries_estimate = sum(1 for v in smiles_counts.values() if v.issuperset(KEEP_INDICES)) * len(KEEP_INDICES)

    with env_in.begin() as txn_in, env_out.begin(write=True) as txn_out:
        cursor = txn_in.cursor()
        pbar = tqdm(desc=f"Shrinking - Part {part_number}", total=total_entries, dynamic_ncols=True)
        for idx, (key, value) in enumerate(cursor):
            if idx % 10000 == 0:
                pbar.update(10000)
            entries_checked += 1

            try:
                data = pickle.loads(value)
            except Exception:
                continue
            if not all(k in data for k in REQUIRED_KEYS):
                continue

            smiles = data["smiles"]
            aug_index = data["aug_index"]

            # smiles가 조건을 만족하고 해당 index가 필요한 경우에만 저장
            if smiles in smiles_counts and smiles_counts[smiles].issuperset(KEEP_INDICES):
                if smiles not in smiles_written:
                    smiles_written.add(smiles)

                if aug_index in KEEP_INDICES:
                    txn_out.put(key, value)
                    count_written += 1
                    if count_written % 10000 == 0:
                        print(f"[INFO] Shrinked {count_written:,} / {total_entries:,} entries", flush=True)

        txn_out.put(b"__len__", pickle.dumps(count_written))

        # 마지막 남은 pbar 마무리
        remaining = entries_checked % 10000
        if remaining:
            pbar.update(remaining)
        pbar.close()

    with open(output_path / "processed_smiles.pkl", "wb") as f:
        pickle.dump(list(smiles_written), f)

    print(f"[COMPLETE] Shrinking - Part {part_number}", flush=True)
    print(f"Total entries checked: {entries_checked}", flush=True)
    print(f"Invalid entries (missing or broken): {invalid_entries}", flush=True)
    print(f"Valid SMILES written: {len(smiles_written)}", flush=True)
    print(f"Total entries written: {count_written} (expected: {len(smiles_written) * len(KEEP_INDICES)})", flush=True)

    with env_out.begin() as txn:
        cursor = txn.cursor()
        actual_written = pickle.loads(txn.get(b"__len__"))
        print(f"[VALIDATION] Output LMDB entry count: {actual_written}", flush=True)
        if actual_written != count_written:
            print("[WARNING] Mismatch between expected and actual written count!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    shrink_lmdb(args.input_path, args.output_path)
