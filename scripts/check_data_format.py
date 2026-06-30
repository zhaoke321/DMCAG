#!/usr/bin/env python3
"""Check data format for all .mat datasets. Output: results/data_format_check.csv"""

import csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from data_utils import load_multiview_mat

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Expected datasets from task.md
EXPECTED_DATASETS = ["BDGP", "HW", "MNIST_USPS", "Multi-COIL-10", "UCI-3V", "Fmnist-MV"]


def check_dataset(ds_name):
    mat_path = DATA_DIR / f"{ds_name}.mat"
    result = {
        "dataset": ds_name,
        "file_exists": mat_path.exists(),
        "Y_raw_shape": "",
        "Y_shape": "",
        "Y_1d_ok": "",
        "n_samples": "",
        "view_keys": "",
        "views_shape_ok": "",
        "error": "",
    }
    if not mat_path.exists():
        result["error"] = "NOT_AVAILABLE"
        return result

    try:
        X_list, Y, meta = load_multiview_mat(str(mat_path))
        result["Y_raw_shape"] = str(meta["Y_raw_shape"])
        result["Y_shape"] = str(meta["Y_shape"])
        result["Y_1d_ok"] = "PASS" if Y.ndim == 1 else "FAIL"
        result["n_samples"] = meta["n_samples"]

        view_keys = meta["view_keys"]
        result["view_keys"] = ",".join(view_keys)

        views_ok = True
        for k in view_keys:
            norm_shape = meta["view_shapes"][k]["normalized_shape"]
            if norm_shape[0] != meta["n_samples"]:
                views_ok = False
                break
        result["views_shape_ok"] = "PASS" if views_ok else "FAIL"

    except Exception as e:
        result["error"] = str(e)[:200]

    return result


def main():
    results = []
    for ds in EXPECTED_DATASETS:
        results.append(check_dataset(ds))

    # Print summary
    print(f"{'Dataset':<16} {'Exists':<8} {'Y_raw':<14} {'Y_1d':<6} {'Views':<6} {'Status'}")
    print("-" * 70)
    for r in results:
        exists = "YES" if r["file_exists"] else "NO"
        status = "OK" if (r["Y_1d_ok"] == "PASS" and r["views_shape_ok"] == "PASS") else r["error"][:30]
        print(f"{r['dataset']:<16} {exists:<8} {r['Y_raw_shape']:<14} {r['Y_1d_ok']:<6} {r['views_shape_ok']:<6} {status}")

    # Write CSV
    csv_path = RESULTS_DIR / "data_format_check.csv"
    fieldnames = [
        "dataset", "file_exists", "Y_raw_shape", "Y_shape", "Y_1d_ok",
        "n_samples", "view_keys", "views_shape_ok", "error"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to: {csv_path}")


if __name__ == "__main__":
    main()
