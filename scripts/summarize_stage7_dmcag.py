#!/usr/bin/env python3
"""Summarize stage 7 DMCAG grid search results. Generates best-single and mean±std CSVs."""

import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Paper Table 2 reference values (percentages → decimals)
PAPER_RESULTS = {
    "BDGP": {"acc": 0.9800, "nmi": 0.9469, "ari": 0.9511},
    "HW":   {"acc": 0.9790, "nmi": 0.9525, "ari": 0.9540},
}


def load_raw(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                continue
            try:
                r["acc"] = float(r["acc"])
                r["nmi"] = float(r["nmi"])
                r["ari"] = float(r["ari"])
                r["arch"] = int(r["arch"])
                r["gamma"] = float(r["gamma"])
                r["seed"] = int(r["seed"])
            except (ValueError, TypeError):
                continue
            rows.append(r)
    return rows


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  → {path} ({len(rows)} rows)")


def best_single(rows, by_metric):
    """Best single run per dataset by given metric."""
    best = {}
    for r in rows:
        ds = r["dataset"]
        if ds not in best or r[by_metric] > best[ds][by_metric]:
            best[ds] = r
    result = []
    for v in best.values():
        result.append({
            "dataset": v["dataset"],
            "arch": v["arch"],
            "gamma": v["gamma"],
            "seed": v["seed"],
            "acc": v["acc"],
            "nmi": v["nmi"],
            "ari": v["ari"],
            "metric": by_metric,
            "log_file": v.get("log_file", ""),
        })
    return result


def mean_std_by_param(rows):
    """Mean±std by dataset, arch, gamma."""
    groups = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["arch"], r["gamma"])
        groups[key].append(r)

    result = []
    for (ds, arch, gamma), runs in sorted(groups.items()):
        accs = [r["acc"] for r in runs]
        nmis = [r["nmi"] for r in runs]
        aris = [r["ari"] for r in runs]
        result.append({
            "dataset": ds,
            "arch": arch,
            "gamma": gamma,
            "num_runs": len(runs),
            "acc_mean": round(np.mean(accs), 6),
            "acc_std": round(np.std(accs, ddof=1) if len(accs) > 1 else 0, 6),
            "nmi_mean": round(np.mean(nmis), 6),
            "nmi_std": round(np.std(nmis, ddof=1) if len(nmis) > 1 else 0, 6),
            "ari_mean": round(np.mean(aris), 6),
            "ari_std": round(np.std(aris, ddof=1) if len(aris) > 1 else 0, 6),
        })
    return result


def best_meanstd(meanstd_rows, by_metric_mean, by_metric_std):
    """Best mean±std config per dataset by specified mean metric."""
    best = {}
    for r in meanstd_rows:
        ds = r["dataset"]
        if r["num_runs"] < 5:
            continue  # Prefer complete runs
        if ds not in best or r[by_metric_mean] > best[ds][by_metric_mean]:
            best[ds] = r
    # Fallback: if no complete run, pick best among incomplete
    if not best:
        for r in meanstd_rows:
            ds = r["dataset"]
            if ds not in best or r[by_metric_mean] > best[ds][by_metric_mean]:
                best[ds] = r
    return [
        {**v, "metric": by_metric_mean}
        for v in best.values()
    ]


def vs_paper(rows, meanstd_rows):
    """Compare best single and mean±std vs paper Table 2."""
    best_acc = {r["dataset"]: r for r in best_single(rows, "acc")}
    best_ms = {r["dataset"]: r for r in best_meanstd(meanstd_rows, "acc_mean", "acc_std")}

    result = []
    for ds in ["BDGP", "HW"]:
        if ds not in PAPER_RESULTS:
            continue
        paper = PAPER_RESULTS[ds]
        bs = best_acc.get(ds, {})
        bm = best_ms.get(ds, {})

        row = {
            "dataset": ds,
            "paper_acc": paper["acc"],
            "paper_nmi": paper["nmi"],
            "paper_ari": paper["ari"],
            "best_single_acc": bs.get("acc", ""),
            "best_single_nmi": bs.get("nmi", ""),
            "best_single_ari": bs.get("ari", ""),
            "best_single_arch": bs.get("arch", ""),
            "best_single_gamma": bs.get("gamma", ""),
            "best_single_seed": bs.get("seed", ""),
            "meanstd_acc_mean": bm.get("acc_mean", ""),
            "meanstd_acc_std": bm.get("acc_std", ""),
            "meanstd_nmi_mean": bm.get("nmi_mean", ""),
            "meanstd_nmi_std": bm.get("nmi_std", ""),
            "meanstd_ari_mean": bm.get("ari_mean", ""),
            "meanstd_ari_std": bm.get("ari_std", ""),
            "meanstd_arch": bm.get("arch", ""),
            "meanstd_gamma": bm.get("gamma", ""),
        }
        # Gaps (Best Single vs Paper)
        if bs.get("acc"):
            row["gap_best_acc"] = round(bs["acc"] - paper["acc"], 4)
        else:
            row["gap_best_acc"] = ""
        if bs.get("nmi"):
            row["gap_best_nmi"] = round(bs["nmi"] - paper["nmi"], 4)
        else:
            row["gap_best_nmi"] = ""
        if bs.get("ari"):
            row["gap_best_ari"] = round(bs["ari"] - paper["ari"], 4)
        else:
            row["gap_best_ari"] = ""

        # Gaps (Mean±Std vs Paper)
        if bm.get("acc_mean"):
            row["gap_mean_acc"] = round(bm["acc_mean"] - paper["acc"], 4)
        else:
            row["gap_mean_acc"] = ""
        if bm.get("nmi_mean"):
            row["gap_mean_nmi"] = round(bm["nmi_mean"] - paper["nmi"], 4)
        else:
            row["gap_mean_nmi"] = ""
        if bm.get("ari_mean"):
            row["gap_mean_ari"] = round(bm["ari_mean"] - paper["ari"], 4)
        else:
            row["gap_mean_ari"] = ""
        result.append(row)
    return result


def main():
    csv_path = RESULTS_DIR / "stage7_dmcag_grid_raw.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run grid search first.")
        return

    rows = load_raw(csv_path)
    print(f"Loaded {len(rows)} ok runs from {csv_path}")

    # 1. Best single by ACC
    write_csv(
        RESULTS_DIR / "stage7_dmcag_best_single_by_acc.csv",
        ["dataset", "arch", "gamma", "seed", "acc", "nmi", "ari", "metric", "log_file"],
        best_single(rows, "acc"),
    )

    # 2. Best single by NMI
    write_csv(
        RESULTS_DIR / "stage7_dmcag_best_single_by_nmi.csv",
        ["dataset", "arch", "gamma", "seed", "acc", "nmi", "ari", "metric", "log_file"],
        best_single(rows, "nmi"),
    )

    # 3. Best single by ARI
    write_csv(
        RESULTS_DIR / "stage7_dmcag_best_single_by_ari.csv",
        ["dataset", "arch", "gamma", "seed", "acc", "nmi", "ari", "metric", "log_file"],
        best_single(rows, "ari"),
    )

    # 4. Mean±Std by parameter
    meanstd = mean_std_by_param(rows)
    write_csv(
        RESULTS_DIR / "stage7_dmcag_mean_std_by_param.csv",
        ["dataset", "arch", "gamma", "num_runs",
         "acc_mean", "acc_std", "nmi_mean", "nmi_std", "ari_mean", "ari_std"],
        meanstd,
    )

    # 5-7. Final mean±std
    for metric_name, metric_key in [("acc", "acc_mean"), ("nmi", "nmi_mean"), ("ari", "ari_mean")]:
        write_csv(
            RESULTS_DIR / f"stage7_dmcag_final_meanstd_by_{metric_name}mean.csv",
            ["dataset", "arch", "gamma", "num_runs",
             "acc_mean", "acc_std", "nmi_mean", "nmi_std", "ari_mean", "ari_std", "metric"],
            best_meanstd(meanstd, metric_key, f"{metric_name}_std"),
        )

    # 8. Vs Paper
    write_csv(
        RESULTS_DIR / "stage7_dmcag_vs_paper.csv",
        [
            "dataset",
            "paper_acc", "paper_nmi", "paper_ari",
            "best_single_acc", "best_single_nmi", "best_single_ari",
            "best_single_arch", "best_single_gamma", "best_single_seed",
            "meanstd_acc_mean", "meanstd_acc_std",
            "meanstd_nmi_mean", "meanstd_nmi_std",
            "meanstd_ari_mean", "meanstd_ari_std",
            "meanstd_arch", "meanstd_gamma",
            "gap_best_acc", "gap_best_nmi", "gap_best_ari",
            "gap_mean_acc", "gap_mean_nmi", "gap_mean_ari",
        ],
        vs_paper(rows, meanstd),
    )

    print("\nDone. All CSVs generated.")


if __name__ == "__main__":
    main()
