import numpy as np
from collections import OrderedDict

import torch
from rdkit import Chem
from torch_geometric.data import Data

from data_provider.coord_utils import get_coord_augs

allowable_features = OrderedDict(
    [
        ("possible_atomic_num_list", list(range(1, 119)) + ["misc"]),
        ("possible_chirality_list", ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"]),
        ("possible_degree_list", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"]),
        ("possible_formal_charge_list", [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"]),
        ("possible_numH_list", [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"]),
        ("possible_number_radical_e_list", [0, 1, 2, 3, 4, "misc"]),
        ("possible_hybridization_list", ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"]),
        ("possible_is_aromatic_list", [False, True]),
        ("possible_is_in_ring_list", [False, True]),
    ]
)
NUM_ATOM_TYPES = len(allowable_features["possible_atomic_num_list"])
NUM_CLASSES_LIST = [len(v) for v in allowable_features.values()]

allowable_bond_features = OrderedDict(
    [
        ("possible_bond_type_list", ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"]),
        ("possible_bond_stereo_list", ["STEREONONE", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS", "STEREOANY"]),
        ("possible_is_conjugated_list", [False, True]),
    ]
)


def safe_index(l, e):
    try:
        return l.index(e)
    except ValueError:
        return len(l) - 1


def atom_to_feature_vector(atom):
    return [
        safe_index(allowable_features["possible_atomic_num_list"], atom.GetAtomicNum()),
        safe_index(allowable_features["possible_chirality_list"], str(atom.GetChiralTag())),
        safe_index(allowable_features["possible_degree_list"], atom.GetTotalDegree()),
        safe_index(allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()),
        safe_index(allowable_features["possible_numH_list"], atom.GetTotalNumHs()),
        safe_index(allowable_features["possible_number_radical_e_list"], atom.GetNumRadicalElectrons()),
        safe_index(allowable_features["possible_hybridization_list"], str(atom.GetHybridization())),
        safe_index(allowable_features["possible_is_aromatic_list"], atom.GetIsAromatic()),
        safe_index(allowable_features["possible_is_in_ring_list"], atom.IsInRing()),
    ]


def bond_to_feature_vector(atom):
    return [
        safe_index(allowable_bond_features["possible_bond_type_list"], str(atom.GetBondType())),
        safe_index(allowable_bond_features["possible_bond_stereo_list"], str(atom.GetStereo())),
        safe_index(allowable_bond_features["possible_is_conjugated_list"], atom.GetIsConjugated()),
    ]


def get_2d_features(mol):
    # node feature
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # edge index
    if len(mol.GetBonds()) > 0:
        edge_index = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])
            feat = bond_to_feature_vector(bond)
            edge_features_list.extend([feat, feat])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_features = torch.tensor(edge_features_list, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_features = torch.empty((0,), dtype=torch.long)
    return x, edge_index, edge_features


def random_mask_atom(x, ratio):
    """Mask non-carbon atoms first, then mask carbon atoms if needed."""
    assert x.shape[0] > 0, "x must have at least one atom"
    assert x.shape[1] == 9, "x must have 9 features"
    assert all(a in range(1, 119) for a in x[:, 0]), "x must have atomic number between 1 and 118"

    # Identify C and non-C atom positions
    length = x.shape[0]
    atom_num = x[:, 0]
    idx_non_c = (atom_num != 6).nonzero(as_tuple=True)[0]
    idx_c = (atom_num == 6).nonzero(as_tuple=True)[0]

    # Init mask
    num_mask = max(1, int(length * ratio))
    num_non_c = idx_non_c.shape[0]
    mask = torch.zeros(length, dtype=torch.bool)

    # Mask separately for non-C and C atoms
    if num_non_c >= num_mask:
        # Randomly mask only among non-C atoms
        perm = torch.randperm(num_non_c)
        mask[idx_non_c[perm[:num_mask]]] = True
    else:
        # Mask all non-C atoms,
        # then randomly mask C atoms for the remainder
        mask[idx_non_c] = True
        num_remaining = num_mask - num_non_c
        if num_remaining > 0 and idx_c.numel() > 0:
            perm = torch.randperm(idx_c.shape[0])
            mask[idx_c[perm[:num_remaining]]] = True

    return mask


def random_mask(length: int, ratio: float):
    if ratio == 0:
        print("[WARNING] Given ratio is 0, returning all False mask")
        return torch.zeros(length, dtype=torch.bool)

    num_mask = max(1, int(length * ratio))
    mask = torch.zeros(length, dtype=torch.bool)
    perm = torch.randperm(length)
    mask[perm[:num_mask]] = True
    return mask


def apply_mask(data: Data, ratio: float = 0.15, mask_atom_strat: str = "random"):
    if mask_atom_strat == "anti_c_dominant":
        data.mask_atom = random_mask_atom(data.x, ratio)
    elif mask_atom_strat == "random":
        data.mask_atom = random_mask(data.num_nodes, ratio)
    data.mask_edge = random_mask(data.edge_index.shape[1], ratio)
    return data


def mol_to_pyg_data_gt(mol) -> Data:
    """
    Convert an RDKit molecule to a PyG Data object with masked SSL features.
    """
    # ===== 공통 2D 정보 =====
    x, edge_index, edge_features = get_2d_features(mol)
    N_atoms = mol.GetNumAtoms()

    # ===== GT 좌표 =====
    coords = mol.GetConformer().GetPositions()
    pos = torch.tensor(coords, dtype=torch.float)
    src, dst = edge_index
    edge_vec = pos[src] - pos[dst]
    edge_len = edge_vec.norm(dim=-1)

    # Create Data object; num_nodes/num_motifs can be computed downstream
    data = {
        "x": x,
        "coord": pos,
        "edge_vec": edge_vec,
        "edge_len": edge_len,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "target_atom": x[:, 0].clone(),
        "target_edge_len": edge_len.clone(),
        "num_nodes": N_atoms,
        "smiles": Chem.MolToSmiles(mol),
        "coord_method": "GT",
        "aug_index": -1,
    }

    return data


def mol_to_pyg_data_aug_list(mol, num_conf=10, calc_heavy_mol=False) -> Data:
    """
    Convert an RDKit molecule to a PyG Data object with masked SSL features.
    """
    # ===== 공통 2D 정보 =====
    x, edge_index, edge_features = get_2d_features(mol)
    org_atom = [atom.GetSymbol() for atom in mol.GetAtoms()]
    N_atoms = mol.GetNumAtoms()

    # ===== Augmentation 좌표 =====
    res = get_coord_augs(mol, num_conf, calc_heavy_mol)
    coordinate_list = res["coordinates"]
    method_list = res["methods"]
    atom_list = res["atoms"]
    smiles_list = res["smiles"]

    i = 0
    data_list = []
    # 생성한 augmentation 좌표들로 3D 정보 계산
    for coords, method, atoms, smiles in zip(coordinate_list, method_list, atom_list, smiles_list):
        if atoms != org_atom:
            raise ValueError("분자가 바뀜 !!!!")
        pos = torch.from_numpy(coords).float()
        src, dst = edge_index
        edge_vec = pos[src] - pos[dst]
        edge_len = edge_vec.norm(dim=-1)

        # 빠른 전처리를 위해서 dictionary 객체 사용
        data_list.append(
            {
                "x": x,
                "coord": pos,
                "edge_vec": edge_vec,
                "edge_len": edge_len,
                "edge_index": edge_index,
                "edge_features": edge_features,
                "target_atom": x[:, 0].clone(),
                "target_edge_len": edge_len.clone(),
                "num_nodes": N_atoms,
                "smiles": smiles,
                "coord_method": method,
                "aug_index": i,
            }
        )
        i += 1

    return data_list


def add_attr_to_moleculnet(smiles, y) -> Data:
    """
    Convert an RDKit molecule to a PyG Data object with masked SSL features.
    """
    mol = Chem.MolFromSmiles(smiles)
    x, edge_index = get_2d_features(mol)  # index를 맞추기 위해서 PyG의 x, edge_index 대신 이거 사용
    org_atom = [atom.GetSymbol() for atom in mol.GetAtoms()]
    N_atoms = mol.GetNumAtoms()

    # PT에서 사용한 augmentation 방법을 그대로 사용 여기선 하나 씩만 만듦.
    # 여러 개 만들면 cheating이 될 거 같아서 하나만.
    res = get_coord_augs(mol, num_conf=1)
    coords = res["coordinates"][0]
    method = res["methods"][0]
    atom = res["atoms"][0]
    if atom != org_atom:
        raise ValueError("분자가 바뀜 !!!!")
    pos = torch.from_numpy(coords).float()
    src, dst = edge_index
    edge_vec = pos[src] - pos[dst]
    edge_len = edge_vec.norm(dim=-1)

    data = Data(
        x=x,
        coord=pos,
        edge_vec=edge_vec,
        edge_len=edge_len,
        edge_index=edge_index,
        num_nodes=N_atoms,
        smiles=smiles,
        coord_method=method,
        aug_index=0,
        y=y,
    )

    return data
