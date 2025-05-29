import numpy as np

import torch
from torchmetrics import Metric
from sklearn.metrics import roc_auc_score


def get_rounded_thresholds(preds: np.ndarray, thold_step: float = 0.1) -> np.ndarray:
    if preds.size == 0:
        return np.array([0.0, 1.0])

    preds = preds.flatten()
    rounded = np.round(preds / thold_step) * thold_step
    thresholds = np.unique(np.concatenate([rounded, [0.0, 1.0]]))
    return np.sort(thresholds)


class GraphCLAUROC(Metric):
    def __init__(self, num_labels: int, thold_step: float = 0.1):
        super().__init__()
        self.num_labels = num_labels
        self.thold_step = thold_step
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        shape: [B, C]
        target: ±1 for pos/neg, 0 for NaN
        """
        assert preds.shape == targets.shape, f"Different shape | preds: {preds.shape}, targets: {targets.shape}"
        assert (
            preds.shape[1] == self.num_labels
        ), f"Not enough labels in preds | preds: {preds.shape}, num_labels: {self.num_labels}"
        assert (
            targets.shape[1] == self.num_labels
        ), f"Not enough labels in targets | targets: {targets.shape}, num_labels: {self.num_labels}"

        self.preds.append(preds.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self):
        preds = torch.cat(self.preds, dim=0).numpy()
        targets = torch.cat(self.targets, dim=0).numpy()

        # num_labels = N, num_valid_labels = V
        # roc는 V 개만 계산하지만 conf, thold는 분석을 위해 N개 모두 반환
        # invalid에서 계산은 되지 않기 때문에, placeholder로 -1 반환
        roc_list = []  # (V,)
        conf_mats = []  # (N, 4)
        best_tholds = []  # (N,)
        for i in range(self.num_labels):
            pred_i = preds[:, i]
            target_i = targets[:, i]

            if np.sum(target_i == 1) > 0 and np.sum(target_i == -1) > 0:
                is_valid = target_i**2 > 0
                pred_i = pred_i[is_valid]
                target_i = (target_i[is_valid] + 1) / 2

                roc = roc_auc_score(target_i, pred_i)
                roc_list.append(roc)

                tholds = get_rounded_thresholds(pred_i, self.thold_step)
                best_th = None
                best_cm = [0, 0, 0, 0]
                best_f1 = -1

                for th in tholds:
                    pred_bin = (pred_i >= th).astype(int)
                    tp = np.sum((pred_bin == 1) & (target_i == 1))
                    fp = np.sum((pred_bin == 1) & (target_i == 0))
                    tn = np.sum((pred_bin == 0) & (target_i == 0))
                    fn = np.sum((pred_bin == 0) & (target_i == 1))

                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)

                    if f1 > best_f1:
                        best_f1 = f1
                        best_th = th
                        best_cm = [tp, fp, tn, fn]

                conf_mats.append(best_cm)
                best_tholds.append(best_th)
            else:
                conf_mats.append([-1, -1, -1, -1])
                best_tholds.append(-1)

        if len(roc_list) == 0:
            print(f"[DEBUG] All target is missing! Returning 0.")
            return torch.tensor(0.0)
        elif len(roc_list) < self.num_labels:
            num_missing = self.num_labels - len(roc_list)
            print(f"[DEBUG] Missing ratio: {1 - len(roc_list) / self.num_labels:.2%}")

        roc_auc = torch.tensor(sum(roc_list) / len(roc_list))
        conf_mats = torch.tensor(conf_mats, dtype=torch.long)
        best_tholds = torch.tensor(best_tholds, dtype=torch.float)
        return roc_auc, conf_mats, best_tholds
