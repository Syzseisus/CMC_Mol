import pickle
import argparse
from tqdm import tqdm
from pathlib import Path

from rdkit import Chem


def load_all_processed_smiles(pkl_dir: Path) -> set:
    all_smiles = set()
    for pkl_file in pkl_dir.glob("part_*/processed_smiles.pkl"):
        with open(pkl_file, "rb") as f:
            smiles_list = pickle.load(f)
            all_smiles.update(smiles_list)
    print(f"[INFO] Loaded {len(all_smiles):,} unique processed SMILES from {pkl_dir}")
    return all_smiles


def extract_unprocessed_sdf(org_sdf_path: Path, processed_smiles_set: set, output_sdf_path: Path):
    suppl = Chem.SDMolSupplier(str(org_sdf_path), removeHs=False)
    writer = Chem.SDWriter(str(output_sdf_path))
    count = 0

    for mol in tqdm(suppl, desc="Checking original SDF"):
        if mol is None:
            continue
        try:
            smiles = Chem.MolToSmiles(mol)
        except Exception:
            continue
        if smiles not in processed_smiles_set:
            mol.SetProp("_Name", smiles)
            writer.write(mol)
            count += 1

    writer.close()
    print(f"[DONE] Saved {count:,} unprocessed molecules to {output_sdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--org_sdf", type=str, required=True, help="Original input SDF file")
    parser.add_argument(
        "--processed_pkl_dir", type=str, required=True, help="Directory containing part_*/processed_smiles.pkl"
    )
    parser.add_argument("--output_sdf", type=str, required=True, help="Output path for unprocessed SDF")
    args = parser.parse_args()

    org_sdf = Path(args.org_sdf)
    pkl_dir = Path(args.processed_pkl_dir)
    output_sdf = Path(args.output_sdf)

    processed_smiles = load_all_processed_smiles(pkl_dir)
    extract_unprocessed_sdf(org_sdf, processed_smiles, output_sdf)
