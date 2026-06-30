#!/usr/bin/env python3
"""Check same-seed reproducibility for stage 6. Output: results/stage6_reproducibility_check.csv"""

import re
import csv
from pathlib import Path
from collections import defaultdict

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs" / "stage6_quick_training"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_result(log_path):
    """Extract FINAL_RESULT from a log."""
    if not log_path.exists():
        return None
    with open(log_path, "r", errors="replace") as f:
        content = f.read()
    m = re.search(
        r"FINAL_RESULT\s+dataset=\S+\s+arch=(\d+)\s+gamma=([\d.]+)\s+seed=(\d+)\s+"
        r"Acc\s+([\d.]+),\s*nmi\s+([\d.]+),\s*ari\s+([\d.]+)",
        content,
    )
    if m:
        return {
            "arch": int(m.group(1)),
            "gamma": float(m.group(2)),
            "seed": int(m.group(3)),
            "acc": float(m.group(4)),
            "nmi": float(m.group(5)),
            "ari": float(m.group(6)),
        }
    return None


def main():
    # Group logs by dataset+arch+gamma+seed
    groups = defaultdict(list)
    for log_path in sorted(LOGS_DIR.glob("*.log")):
        m = re.match(r"(\S+?)_arch(\d+)_gamma([\d.]+)_seed(\d+)(?:_run\d+)?\.log", log_path.name)
        if not m:
            continue
        ds, arch, gamma, seed = m.group(1), m.group(2), m.group(3), m.group(4)
        key = (ds, arch, gamma, seed)
        result = extract_result(log_path)
        if result:
            groups[key].append((log_path.name, result))

    rows = []
    for (ds, arch, gamma, seed), runs in sorted(groups.items()):
        if len(runs) < 2:
            # Single run — can't check reproducibility
            rows.append({
                "dataset": ds,
                "arch": arch,
                "gamma": gamma,
                "seed": seed,
                "run1_acc": runs[0][1]["acc"] if runs else "",
                "run1_nmi": runs[0][1]["nmi"] if runs else "",
                "run1_ari": runs[0][1]["ari"] if runs else "",
                "run2_acc": "",
                "run2_nmi": "",
                "run2_ari": "",
                "acc_diff": "",
                "nmi_diff": "",
                "ari_diff": "",
                "reproducible": "single_run",
                "notes": f"Only {len(runs)} run(s), can't check reproducibility",
            })
            continue

        r1, r2 = runs[0][1], runs[1][1]
        acc_diff = abs(r1["acc"] - r2["acc"])
        nmi_diff = abs(r1["nmi"] - r2["nmi"])
        ari_diff = abs(r1["ari"] - r2["ari"])

        # Tolerance: < 0.001 is "identical" (floating point), < 0.01 is "acceptable"
        if acc_diff < 0.001 and nmi_diff < 0.001 and ari_diff < 0.001:
            reproducible = "exact"
            notes = f"Identical (diff < 1e-3)"
        elif acc_diff < 0.01 and nmi_diff < 0.01 and ari_diff < 0.01:
            reproducible = "acceptable"
            notes = f"Minor variation (delta acc={acc_diff:.4f} nmi={nmi_diff:.4f} ari={ari_diff:.4f})"
        else:
            reproducible = "not_reproducible"
            notes = f"Significant difference! delta acc={acc_diff:.4f} nmi={nmi_diff:.4f} ari={ari_diff:.4f}"

        rows.append({
            "dataset": ds,
            "arch": arch,
            "gamma": gamma,
            "seed": seed,
            "run1_acc": r1["acc"],
            "run1_nmi": r1["nmi"],
            "run1_ari": r1["ari"],
            "run2_acc": r2["acc"],
            "run2_nmi": r2["nmi"],
            "run2_ari": r2["ari"],
            "acc_diff": round(acc_diff, 6),
            "nmi_diff": round(nmi_diff, 6),
            "ari_diff": round(ari_diff, 6),
            "reproducible": reproducible,
            "notes": notes,
        })

    csv_path = RESULTS_DIR / "stage6_reproducibility_check.csv"
    fieldnames = [
        "dataset", "arch", "gamma", "seed",
        "run1_acc", "run1_nmi", "run1_ari",
        "run2_acc", "run2_nmi", "run2_ari",
        "acc_diff", "nmi_diff", "ari_diff",
        "reproducible", "notes",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Reproducibility check → {csv_path}")
    for r in rows:
        icon = "✅" if r["reproducible"] in ("exact", "acceptable") else "⚠️" if r["reproducible"] != "not_reproducible" else "❌"
        print(f"  {icon} {r['dataset']} arch={r['arch']} gamma={r['gamma']} seed={r['seed']}: "
              f"{r['reproducible']} | {r['notes']}")


if __name__ == "__main__":
    main()
