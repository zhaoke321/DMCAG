# ============================================================
# config_utils.py — 数据集配置表
# 只允许数据集固有配置（save_path, n_clusters, viewNumber 等）
# 严禁覆盖 arch, gamma, seed, noise（这些由命令行控制）
# ============================================================

import numpy as np

DATASET_CONFIG = {
    "BDGP": {
        "method": "BDGP",
        "save_path": "./data/BDGP.pkl",
        "n_clusters": 5,
        "viewNumber": 2,
        "n_input": [1750, 79],
        "instanceNumber": 2500,
        "batch_size": 2500,
        "use_conv": False,
    },
    "HW": {
        "method": "HW",
        "save_path": "./data/HW.pkl",
        "n_clusters": 10,
        "viewNumber": 6,
        "n_input": [216, 76, 64, 6, 240, 47],
        "instanceNumber": 2000,
        "batch_size": 2000,
        "use_conv": False,
    },
    "Fmnist-MV": {
        "method": "Fmnist-MV",
        "save_path": "./data/Fmnist-MV.pkl",
        "n_clusters": 10,
        "viewNumber": 3,
        "n_input": [784, 784, 784],
        "instanceNumber": 10000,
        "batch_size": 10000,
        "use_conv": False,
    },
    "UCI-3V": {
        "method": "UCI-3V",
        "save_path": "./data/UCI_3V.pkl",
        "n_clusters": 10,
        "viewNumber": 3,
        "n_input": [240, 76, 64],
        "instanceNumber": 2000,
        "batch_size": 2000,
        "use_conv": False,
    },
    "MNIST_USPS": {
        "method": "MNIST_USPS",
        "save_path": "./data/MNIST_USPS.pkl",
        "n_clusters": 10,
        "viewNumber": 2,
        "n_input": [[1, 28, 28], [1, 28, 28]],
        "instanceNumber": 5000,
        "batch_size": 5000,
        "use_conv": True,
    },
    "MNIST-USPS": {
        "method": "MNIST_USPS",
        "save_path": "./data/MNIST_USPS.pkl",
        "n_clusters": 10,
        "viewNumber": 2,
        "n_input": [[1, 28, 28], [1, 28, 28]],
        "instanceNumber": 5000,
        "batch_size": 5000,
        "use_conv": True,
    },
    "Multi-COIL-10": {
        "method": "Multi-COIL-10",
        "save_path": "./data/Multi-COIL-10.pkl",
        "n_clusters": 10,
        "viewNumber": 3,
        "n_input": [[1, 32, 32], [1, 32, 32], [1, 32, 32]],
        "instanceNumber": 720,
        "batch_size": 720,
        "use_conv": True,
    },
}


def apply_dataset_config(args):
    """
    Apply dataset-specific configuration to args.
    ONLY sets: method, save_path, n_clusters, viewNumber, n_input,
               instanceNumber, batch_size, use_conv
    NEVER sets: arch, gamma, seed, noise (controlled by command line)
    """
    if args.dataset not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset={args.dataset}. "
            f"Available datasets: {list(DATASET_CONFIG.keys())}"
        )

    cfg = DATASET_CONFIG[args.dataset]

    # Dataset-intrinsic attributes only
    args.method = cfg["method"]
    args.save_path = cfg["save_path"]
    args.n_clusters = cfg["n_clusters"]
    args.viewNumber = cfg["viewNumber"]
    args.n_input = cfg["n_input"]
    args.instanceNumber = cfg["instanceNumber"]
    args.batch_size = cfg["batch_size"]
    args.use_conv = cfg.get("use_conv", False)

    return args
