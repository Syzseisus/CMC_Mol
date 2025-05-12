import torch
from rdkit.Chem.Scaffolds import MurckoScaffold


def to_scaffold(smiles, chirality=True):
    """Return Murcko scaffold SMILES for a molecule."""
    return MurckoScaffold.MurckoScaffoldSmiles(smiles, includeChirality=chirality)


def round_to_boundary(indexes, keys, i):
    """Adjust index so that scaffold groups are not split."""
    n = len(indexes)

    for j in range(1, min(i, n - i)):
        if i - j - 1 >= 0 and keys[indexes[i - j]] != keys[indexes[i - j - 1]]:
            return i - j
        if i + j < n and keys[indexes[i + j]] != keys[indexes[i + j - 1]]:
            return i + j

    return 0 if i < n - i else n


def key_split_indexes(keys, key_lengths=None, rng=None):
    """Split by scaffold keys and return index lists."""
    keys = torch.as_tensor(keys)
    key_set, keys = torch.unique(keys, return_inverse=True)

    perm = torch.randperm(len(key_set), generator=rng)
    keys = perm[keys]
    indexes = keys.argsort().tolist()

    if key_lengths is not None:
        key_counts = keys.bincount()
        key_offset = 0
        lengths = []
        for kl in key_lengths:
            lengths.append(key_counts[key_offset : key_offset + kl].sum().item())
            key_offset += kl

    offset = 0
    offsets = [offset]
    for length in key_lengths:
        target = offset + length
        if target >= len(indexes):
            target = len(indexes) - 1
        offset = round_to_boundary(indexes, keys, target)
        offsets.append(offset)
    offsets[-1] = len(indexes)

    return [indexes[offsets[i] : offsets[i + 1]] for i in range(len(lengths))]


def scaffold_split(dataset, seed=42):
    """Return train/valid/test index lists with scaffold-aware splitting."""
    # print(f"[DEBUG] dataset length: {len(dataset)}")
    # for i in range(len(dataset)):
    #     print(f"[DEBUG] trying idx {i}")
    #     sample = dataset[i]

    rng = torch.Generator().manual_seed(seed)  # local seed
    scaffold_to_id = {}
    keys = []

    for sample in dataset:
        scaffold = to_scaffold(sample.smiles)
        if scaffold not in scaffold_to_id:
            scaffold_to_id[scaffold] = len(scaffold_to_id)
        keys.append(scaffold_to_id[scaffold])

    keys = torch.tensor(keys)
    unique_keys = torch.unique(keys)
    num_scaffolds = len(unique_keys)
    print(f"[DEBUG] num_scaffolds: {num_scaffolds}")

    # Fixed split ratio: 80% train, 10% valid, 10% test
    num_train = int(0.8 * num_scaffolds)
    num_valid = int(0.1 * num_scaffolds)
    num_test = num_scaffolds - num_train - num_valid
    key_lengths = [num_train, num_valid, num_test]

    train_idx, valid_idx, test_idx = key_split_indexes(keys, key_lengths=key_lengths, rng=rng)
    all_indices = train_idx + valid_idx + test_idx
    print(f"[DEBUG] max index in split: {max(all_indices)}")
    print(f"[DEBUG] split lengths → train: {len(train_idx)}, val: {len(valid_idx)}, test: {len(test_idx)}")

    return train_idx, valid_idx, test_idx


def random_split(dataset, seed=42):
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=rng).tolist()

    # Fixed split ratio: 80% train, 10% valid, 10% test
    n = len(indices)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return train_idx, val_idx, test_idx
