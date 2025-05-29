from typing import Union, Literal, Optional

import torch
from torch import Tensor, tensor
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.classification import BinaryAUROC, MultilabelAUROC
from torchmetrics.utilities.compute import _auc_compute_without_check
from torchmetrics.functional.classification.auroc import _reduce_auroc, _binary_auroc_compute
from torchmetrics.functional.classification.roc import _binary_roc_compute, _multilabel_roc_compute


def _binary_auroc_compute_return_thold(
    state: Union[Tensor, tuple[Tensor, Tensor]],
    thresholds: Optional[Tensor],
    max_fpr: Optional[float] = None,
    pos_label: int = 1,
) -> Tensor:
    # @@@@@@@@@@@@@@ 다른 건 여기서 `_`를 `thres`로 바꾸고
    fpr, tpr, thres = _binary_roc_compute(state, thresholds, pos_label)
    if max_fpr is None or max_fpr == 1 or fpr.sum() == 0 or tpr.sum() == 0:
        # @@@@@@@@@@@@@@ `thres` 또한 반환하게 만든 것.
        return _auc_compute_without_check(fpr, tpr, 1.0), thres

    _device = fpr.device if isinstance(fpr, Tensor) else fpr[0].device
    max_area: Tensor = tensor(max_fpr, device=_device)
    # Add a single point at max_fpr and interpolate its tpr value
    stop = torch.bucketize(max_area, fpr, out_int32=True, right=True)
    weight = (max_area - fpr[stop - 1]) / (fpr[stop] - fpr[stop - 1])
    interp_tpr: Tensor = torch.lerp(tpr[stop - 1], tpr[stop], weight)
    tpr = torch.cat([tpr[:stop], interp_tpr.view(1)])
    fpr = torch.cat([fpr[:stop], max_area.view(1)])

    # Compute partial AUC
    partial_auc = _auc_compute_without_check(fpr, tpr, 1.0)

    # McClish correction: standardize result to be 0.5 if non-discriminant and 1 if maximal
    min_area: Tensor = 0.5 * max_area**2
    # @@@@@@@@@@@@@@ `thres` 또한 반환하게 만든 것.
    return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area)), thres


class BinaryAUROC_withTholdList(BinaryAUROC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self) -> Tensor:  # type: ignore[override]
        """Compute metric."""
        state = (dim_zero_cat(self.preds), dim_zero_cat(self.target)) if self.thresholds is None else self.confmat
        score, tholds = _binary_auroc_compute_return_thold(state, self.thresholds, self.max_fpr)
        # @@@@@@@@@@@@@@ 그래서 metric 값이랑 threshold list를 같이 얻을 수 있음
        return score, tholds


def _multilabel_auroc_compute_return_thold(
    state: Union[Tensor, tuple[Tensor, Tensor]],
    num_labels: int,
    average: Optional[Literal["micro", "macro", "weighted", "none"]],
    thresholds: Optional[Tensor],
    ignore_index: Optional[int] = None,
) -> Tensor:
    if average == "micro":
        if isinstance(state, Tensor) and thresholds is not None:
            return _binary_auroc_compute(state.sum(1), thresholds, max_fpr=None)

        preds = state[0].flatten()
        target = state[1].flatten()
        if ignore_index is not None:
            idx = target == ignore_index
            preds = preds[~idx]
            target = target[~idx]
        return _binary_auroc_compute((preds, target), thresholds, max_fpr=None)

    # @@@@@@@@@@@@@@ 다른 건 여기서 `_`를 `thres`로 바꾸고
    fpr, tpr, thres = _multilabel_roc_compute(state, num_labels, thresholds, ignore_index)

    # @@@@@@@@@@@@@@ `thres` 또한 반환하게 만든 것.
    return (
        _reduce_auroc(
            fpr,
            tpr,
            average,
            weights=(state[1] == 1).sum(dim=0).float() if thresholds is None else state[0][:, 1, :].sum(-1),
        ),
        thres,
    )


class MultilabelAUROC_withTholdList(MultilabelAUROC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self) -> Tensor:  # type: ignore[override]
        """Compute metric."""
        state = (dim_zero_cat(self.preds), dim_zero_cat(self.target)) if self.thresholds is None else self.confmat
        score, tholds = _multilabel_auroc_compute_return_thold(
            state, self.num_labels, self.average, self.thresholds, self.ignore_index
        )
        # @@@@@@@@@@@@@@ 그래서 metric 값이랑 threshold list를 같이 얻을 수 있음
        return score, tholds
