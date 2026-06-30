# ============================================================
# metrics_utils.py — 统一聚类指标计算
# 所有 ACC/NMI/ARI 计算必须经过此文件，禁止训练脚本各自实现
# ============================================================

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from munkres import Munkres


def ensure_1d_label(y, name="label"):
    """
    Ensure label array is 1-D with shape [n].
    Accepts [n], [n,1], [1,n] and converts them to [n].
    Raises error for invalid shapes.
    """
    y = np.asarray(y)
    raw_shape = y.shape
    y = np.squeeze(y)

    if y.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D after squeeze, raw_shape={raw_shape}, got_shape={y.shape}"
        )

    return y.astype(np.int64)


def cluster_acc(y_true, y_pred):
    """
    Clustering accuracy with Hungarian matching.
    y_true and y_pred must represent one label per sample.
    """
    y_true = ensure_1d_label(y_true, "y_true")
    y_pred = ensure_1d_label(y_pred, "y_pred")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true.shape={y_true.shape}, y_pred.shape={y_pred.shape}"
        )

    # Shift true labels to start from 0, without changing clustering structure.
    y_true = y_true - np.min(y_true)

    true_labels = list(set(y_true))
    pred_labels = list(set(y_pred))

    num_true = len(true_labels)
    num_pred = len(pred_labels)
    n_class = max(num_true, num_pred)

    cost = np.zeros((n_class, n_class), dtype=np.int64)

    for i, true_label in enumerate(true_labels):
        true_idx = np.where(y_true == true_label)[0]
        for j, pred_label in enumerate(pred_labels):
            pred_idx = np.where(y_pred == pred_label)[0]
            cost[i, j] = len(set(true_idx) & set(pred_idx))

    m = Munkres()
    indexes = m.compute((-cost).tolist())
    total = sum(cost[i][j] for i, j in indexes)

    return total / len(y_true)


def clustering_metrics(y_true, y_pred):
    """
    Return ACC, NMI, ARI.
    All outputs are decimals in [0,1], not percentages.
    """
    y_true = ensure_1d_label(y_true, "y_true")
    y_pred = ensure_1d_label(y_pred, "y_pred")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true.shape={y_true.shape}, y_pred.shape={y_pred.shape}"
        )

    acc = cluster_acc(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    return acc, nmi, ari


def check_metric_inputs(y_true, y_pred, dataset="", method=""):
    """
    Explicit runtime check before metric computation.
    Useful for debugging reproduction experiments.
    """
    y_true_checked = ensure_1d_label(y_true, "y_true")
    y_pred_checked = ensure_1d_label(y_pred, "y_pred")

    if y_true_checked.shape[0] != y_pred_checked.shape[0]:
        raise ValueError(
            f"[{method} {dataset}] metric input length mismatch: "
            f"y_true={y_true_checked.shape}, y_pred={y_pred_checked.shape}"
        )

    return y_true_checked, y_pred_checked
