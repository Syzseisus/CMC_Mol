import torch


# deprecated
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

    # esol, freesolv, lipo는 그대로 유지
    # 나머지는 0 → -1, nan → 0 변환
    if dataname in {"bbbp", "tox21", "toxcast", "sider", "clintox", "muv", "hiv", "bace"}:
        y = data.y.clone().float()
        y[y == 0] = -1  # 0 → -1
        y = torch.nan_to_num(y, nan=0.0)  # NaN → 0
        data.y = y
    elif dataname in {"esol", "freesolv", "lipo"}:
        pass
    else:
        raise ValueError(f"Unknown dataset name: {dataname}")

    return data
