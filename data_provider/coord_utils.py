# THX TO Uni-Mol and 3DGCL

import numpy as np

from rdkit.Chem import AllChem
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

"""
`get_coord_augs`가 메인임. 큰 분자는 2D 좌표만 사용함. (`calc_heavy_mol=False`)

`mol2_3Dcoords`는 3D 좌표를 생성하는 함수임.
1. 3D 좌표를 받기 위해 `Conformer`를 만드려면 H를 추가해야 함. (`addHs_preserve_mapping`)
2. 종류 별 개수
    - 2D : inference 시 2D 밖에 못 쓰는 경우를 대비해 2D는 무조건 만들고,
    - 3D : MMFF, ETKDG 반 씩 생성
        - MMFF  : 느리지만 좀 더 정확해서 홀수면 얘 하나 더 씀
        - ETKDG : 좀 더 빠르지만 정확도가 약간 떨어짐
3. MMFF는
    1. `EmbedMolecule`로 임베딩 == Conformer 생성
    2. `MMFFOptimizeMolecule`로 최적화함 == MMFF 방법 계산
    3. H를 추가해뒀기 때문에 H도 받지만 실제로는 H 안 써서, 제거함 (`get_non_h_coords`)
    4. 만약 임베딩 실패하면 (`res == -1`) 좀 더 (1000번) 시도해봄. -> 그래도 실패하면 2D 사용
    5. Optimize 중에 실패하면 (`except`) 2D 사용
4. ETKDG는
    1. `EmbedMolecule`로 할 때 `ETKDGv3` 파라미터를 사용 == ETKDG 방법
    3. 만약 임베딩 실패하면 (`res == -1`) 좀 더 (1000번) 시도해봄. -> 그래도 실패하면 2D 사용
    4. Optimize 중에 실패하면 (`except`) 2D 사용
"""


# ========== Original Atom Mapping ==========
def addHs_preserve_mapping(mol):
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetIntProp("orig_idx", i)
    mol_H = Chem.AddHs(mol)
    return mol_H


def get_non_h_coords(mol_with_H):
    conf = mol_with_H.GetConformer()
    positions = conf.GetPositions()
    coords = []

    for atom in mol_with_H.GetAtoms():
        if atom.HasProp("orig_idx"):
            orig_idx = atom.GetIntProp("orig_idx")
            coords.append((orig_idx, positions[atom.GetIdx()]))

    coords.sort(key=lambda x: x[0])  # restore original heavy atom order
    coords_only = np.array([pos for _, pos in coords], dtype=np.float32)
    return coords_only


# ========== 2D Conformer ==========
def mol2_2Dcoords(mol):
    mol_H = addHs_preserve_mapping(mol)
    AllChem.Compute2DCoords(mol_H)
    coords = get_non_h_coords(mol_H)
    return coords


# ========== 3D Conformer (MMFF or ETKDG) ==========
def try_embed_optimize(mol_H, seed, use_etkdg=False):
    mol_copy = Chem.Mol(mol_H)

    if use_etkdg:
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        try:
            res = AllChem.EmbedMolecule(mol_copy, params)
            if res != 0:
                raise ValueError("ETKDG embedding failed")
        except Exception as e:
            print(f"Seed {seed} (ETKDG) failed: {e}")
            return None

    else:
        try:
            res = AllChem.EmbedMolecule(mol_copy, randomSeed=seed)
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol_copy)
                except:
                    raise ValueError("MMFF embedding failed")
        except Exception as e:
            print(f"Seed {seed} (MMFF) failed: {e}")
            return None

    coords = get_non_h_coords(mol_copy)
    return coords.astype(np.float32)


# ========== Full 3D Generation with 2D fallback ==========
def mol2_3Dcoords(mol, num_conf):
    mol_H = addHs_preserve_mapping(mol)

    assert num_conf > 0, f"Bad `num_conf` which should be positive. (Got {num_conf})"
    return_2d = num_conf > 1
    num_remain = num_conf - 1
    num_etkdg = num_remain // 2
    num_mmff = num_remain - num_etkdg
    if num_conf == 1:
        num_etkdg = 1

    coord_list = []
    method_list = []
    atom_list = []
    smiles_list = []
    # MMFF
    for seed in range(num_mmff):
        try:
            res = AllChem.EmbedMolecule(mol_H, randomSeed=seed)
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol_H)
                    coords = get_non_h_coords(mol_H).astype(np.float32)
                    method = "MMFF"
                except:
                    print(f"Failed to generate MMFF at seed={seed}")
                    coords = mol2_2Dcoords(mol)
                    method = "2D"
            elif res == -1:
                mol_tmp = Chem.Mol(mol_H)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=1000, randomSeed=seed)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)
                    coords = get_non_h_coords(mol_H).astype(np.float32)
                    method = "MMFF"
                except:
                    print(f"Failed to generate MMFF at seed={seed}")
                    coords = mol2_2Dcoords(mol)
                    method = "2D"
        except:
            print(f"Failed to generate MMFF at seed={seed}")
            coords = mol2_2Dcoords(mol)
            method = "2D"

        coord_list.append(coords)
        method_list.append(method)
        atom_list.append([atom.GetSymbol() for atom in mol.GetAtoms()])
        smiles_list.append(Chem.MolToSmiles(mol))

    # ETKDG
    params = AllChem.ETKDGv3()
    for seed in range(num_etkdg):
        try:
            params.randomSeed = seed
            res = AllChem.EmbedMolecule(mol_H, params)
            if res == 0:
                try:
                    coords = get_non_h_coords(mol_H).astype(np.float32)
                    method = "ETKDG"
                except:
                    print(f"Failed to generate ETKDG at seed={seed}")
                    coords = mol2_2Dcoords(mol)
                    method = "2D"
            elif res == -1:
                mol_tmp = Chem.Mol(mol_H)
                params.maxAttempts = 1000
                AllChem.EmbedMolecule(mol_tmp, params)
                try:
                    coords = get_non_h_coords(mol_H).astype(np.float32)
                    method = "ETKDG"
                except:
                    print(f"Failed to generate ETKDG at seed={seed}")
                    coords = mol2_2Dcoords(mol)
                    method = "2D"
        except:
            print(f"Failed to generate ETKDG at seed={seed}")
            coords = mol2_2Dcoords(mol)
            method = "2D"

        coord_list.append(coords)
        method_list.append(method)
        atom_list.append([atom.GetSymbol() for atom in mol.GetAtoms()])
        smiles_list.append(Chem.MolToSmiles(mol))

    # Fixed 2D
    if return_2d:
        coord_list.append(mol2_2Dcoords(mol))
        method_list.append("2D")
        atom_list.append([atom.GetSymbol() for atom in mol.GetAtoms()])
        smiles_list.append(Chem.MolToSmiles(mol))

    return coord_list, method_list, atom_list, smiles_list


# ========== Entry Point ==========
def get_coord_augs(mol_org, num_conf=10, calc_heavy_mol=False, multi_conf=False):
    if multi_conf:
        try:
            return get_coord_augs_fixed(mol_org, num_conf, calc_heavy_mol)
        except Exception as e:
            print(f"[ERROR] get_coord_augs_fixed 실패: {e}")
            return fallback_all_2d(mol_org, num_conf)

    mol = Chem.Mol(mol_org)
    if mol.GetNumAtoms() > 400 and not calc_heavy_mol:
        coord_np = mol2_2Dcoords(mol)
        coordinate_list = np.stack([coord_np] * num_conf)  # shape: (num_conf, N, 3)
        method_list = ["2D"] * num_conf
        atom_list = [[atom.GetSymbol() for atom in mol.GetAtoms()]] * num_conf
        smiles_list = [Chem.MolToSmiles(mol_org)] * num_conf
        print("atom num > 400, using 2D coords only.")
    else:
        coordinate_list, method_list, atom_list, smiles_list = mol2_3Dcoords(mol, num_conf)
        try:
            if len(coordinate_list) > 1:
                coordinate_list = np.stack(coordinate_list)
        except Exception as e:
            for methods, coord in zip(method_list, coordinate_list):
                print(methods, coord.shape)
            raise e

    return {
        "coordinates": coordinate_list,
        "methods": method_list,
        "atoms": atom_list,
        "mol": mol,
        "smiles": smiles_list,
    }


# ========== Multi-Conformer ==========
def get_coord_augs_fixed(mol_org, num_conf=10, calc_heavy_mol=False):
    mol = Chem.Mol(mol_org)
    atom_list = [atom.GetSymbol() for atom in mol.GetAtoms()]
    smiles = Chem.MolToSmiles(mol)

    coord_list, method_list = [], []

    num_2d = 1
    num_etkdg = (num_conf - num_2d) // 2
    num_mmff = num_conf - num_2d - num_etkdg  # 홀수일 땐 얘를 한 개 더 많이

    # === MMFF ===
    mol_mmff = Chem.AddHs(mol)
    try:
        mmff_ids = AllChem.EmbedMultipleConfs(mol_mmff, numConfs=num_mmff, randomSeed=42, numThreads=0)
        AllChem.MMFFOptimizeMoleculeConfs(mol_mmff, maxIters=50, numThreads=0)
        for cid in mmff_ids:
            coords = mol_mmff.GetConformer(cid).GetPositions().astype(np.float32)
            coord_list.append(coords)
            method_list.append("MMFF")
    except:
        print("[WARN] MMFF 생성 실패 → 2D로 대체")
        return fallback_all_2d(mol, num_conf)

    # === ETKDG ===
    mol_etkdg = Chem.AddHs(mol)
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = 0
        params.numThreads = 0
        params.maxAttempts = 100
        etkdg_ids = AllChem.EmbedMultipleConfs(mol_etkdg, numConfs=num_etkdg, params=params)
        for cid in etkdg_ids:
            coords = mol_etkdg.GetConformer(cid).GetPositions().astype(np.float32)
            coord_list.append(coords)
            method_list.append("ETKDG")
    except:
        print("[WARN] ETKDG 생성 실패 → 2D로 대체")
        return fallback_all_2d(mol, num_conf)

    # === 2D 1개 ===
    try:
        mol_2d = Chem.AddHs(Chem.Mol(mol))
        AllChem.Compute2DCoords(mol_2d)
        coords = mol_2d.GetConformer().GetPositions().astype(np.float32)
        coord_list.append(coords)
        method_list.append("2D")
    except:
        print("[WARN] 2D 생성 실패 → 나머지 모두 사용")

    # === 최종 반환 ===
    assert len(coord_list) == num_conf, f"[ERROR] 좌표 개수 불일치: {len(coord_list)}개"
    return {
        "coordinates": coord_list,
        "methods": method_list,
        "atoms": [atom_list] * num_conf,
        "mol": mol,
        "smiles": [smiles] * num_conf,
    }


def fallback_all_2d(mol, num_conf):
    coord_2d = mol2_2Dcoords(mol)
    return {
        "coordinates": [coord_2d] * num_conf,
        "methods": ["2D"] * num_conf,
        "atoms": [[atom.GetSymbol() for atom in mol.GetAtoms()]] * num_conf,
        "smiles": [Chem.MolToSmiles(mol)] * num_conf,
        "mol": mol,
    }
