import torch


def label_preprocessor(data, dataname):
    """
    MoleculeNet 라벨 전처리:
    - NaN → 0
    - 0, 1은 그대로 유지 (multilabel metric을 위해)

    Args:
        data (torch_geometric.data.Data): PyG Data 객체
        dataname (str): 'muv', 'tox21', 'toxcast', 'sider', 'clintox', 'hiv', ...

    Returns:
        data: 전처리된 data 객체
    """
    if not hasattr(data, "y"):
        return data

    if dataname in {"muv", "tox21", "toxcast", "sider", "clintox"}:
        y = data.y.clone().float()
        y = torch.nan_to_num(y, nan=0.0)  # NaN → 0
        data.y = y

    elif dataname in {"hiv", "bace", "bbbp"}:
        y = data.y.clone().float()
        data.y = y  # NaN 없는 binary classification

    elif dataname in {"esol", "freesolv", "lipo"}:
        pass  # regression: 그대로 유지

    else:
        raise ValueError(f"Unknown dataset name: {dataname}")

    return data
