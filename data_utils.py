# ============================================================
# data_utils.py — 统一数据加载与格式规范化
# 解决 Y=[n,1]/[1,n] 兼容、view=[n,d]/[d,n] 兼容问题
# ============================================================

import numpy as np
import scipy.io as sio
from pathlib import Path


def normalize_label(y):
    """
    Convert label array to shape [n].
    Accepts [n,1], [1,n], [n].
    """
    y = np.asarray(y)
    raw_shape = y.shape
    y = np.squeeze(y)

    if y.ndim != 1:
        raise ValueError(
            f"Y must be 1-D after squeeze, raw_shape={raw_shape}, got={y.shape}"
        )

    return y.astype(np.int64)


def normalize_view(x, n_samples, view_name=""):
    """
    Ensure feature matrix is sample-first: [n,d].
    Accepts [n,d] or [d,n].
    """
    x = np.asarray(x)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    raw_shape = x.shape

    if x.shape[0] == n_samples:
        return x

    if x.ndim == 2 and x.shape[1] == n_samples:
        return x.T

    raise ValueError(
        f"View {view_name} shape mismatch: raw_shape={raw_shape}, n_samples={n_samples}"
    )


def get_view_keys(mat_dict):
    """
    Return sorted X-view keys: X1, X2, ...
    """
    keys = []
    for k in mat_dict.keys():
        if k.startswith("X") and k[1:].isdigit():
            keys.append(k)
    keys = sorted(keys, key=lambda x: int(x[1:]))
    return keys


def load_multiview_mat(mat_path):
    """
    Load .mat multi-view dataset.
    Returns:
        X_list: list of [n, d_v] arrays
        Y: [n] int64 array
        meta: dict with shapes and keys
    """
    mat_path = Path(mat_path)
    data = sio.loadmat(mat_path)

    if "Y" not in data:
        raise KeyError(f"{mat_path} does not contain key 'Y'")

    Y_raw = data["Y"]
    Y = normalize_label(Y_raw)
    n_samples = len(Y)

    view_keys = get_view_keys(data)
    if not view_keys:
        raise KeyError(f"{mat_path} has no X1/X2/... view keys")

    X_list = []
    view_shapes = {}

    for key in view_keys:
        x_raw = data[key]
        x = normalize_view(x_raw, n_samples=n_samples, view_name=key)
        X_list.append(x)
        view_shapes[key] = {
            "raw_shape": tuple(x_raw.shape),
            "normalized_shape": tuple(x.shape),
        }

    meta = {
        "path": str(mat_path),
        "Y_raw_shape": tuple(Y_raw.shape),
        "Y_shape": tuple(Y.shape),
        "view_keys": view_keys,
        "view_shapes": view_shapes,
        "n_samples": n_samples,
    }

    return X_list, Y, meta
