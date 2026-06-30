#!/usr/bin/env python3
"""Stage 5 quick test: verify RUN_CONFIG matches expected args from log filenames."""

import re
import csv
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs" / "stage5_hardcoding_tests"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_log(log_path):
    """Extract RUN_CONFIG and FINAL_RESULT from a log file."""
    result = {
        "log_file": log_path.name,
        "run_config": None,
        "final_result": None,
        "acc": None,
        "nmi": None,
        "ari": None,
        "status": "no_log",
    }

    if not log_path.exists():
        result["status"] = "missing"
        return result

    with open(log_path, "r") as f:
        content = f.read()

    # Extract RUN_CONFIG
    rc_match = re.search(r"RUN_CONFIG\s+(.+)", content)
    if rc_match:
        result["run_config"] = rc_match.group(1).strip()
        result["status"] = "has_run_config"

    # Extract FINAL_RESULT
    fr_match = re.search(
        r"FINAL_RESULT\s+dataset=(\S+)\s+arch=(\d+)\s+gamma=([\d.]+)\s+seed=(\d+)\s+"
        r"Acc\s+([\d.]+),\s*nmi\s+([\d.]+),\s*ari\s+([\d.]+)",
        content,
    )
    if fr_match:
        result["final_result"] = fr_match.group(0)
        result["dataset_found"] = fr_match.group(1)
        result["arch_found"] = int(fr_match.group(2))
        result["gamma_found"] = float(fr_match.group(3))
        result["seed_found"] = int(fr_match.group(4))
        result["acc"] = float(fr_match.group(5))
        result["nmi"] = float(fr_match.group(6))
        result["ari"] = float(fr_match.group(7))
        result["status"] = "complete"

    return result


def parse_filename(filename):
    """Extract expected args from filename pattern like BDGP_arch10_gamma0.1_seed0.log"""
    patterns = [
        r"(?P<dataset>\S+)_arch(?P<arch>\d+)_gamma(?P<gamma>[\d.]+)_seed(?P<seed>\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, filename)
        if m:
            return {
                "dataset_expected": m.group("dataset"),
                "arch_expected": int(m.group("arch")),
                "gamma_expected": float(m.group("gamma")),
                "seed_expected": int(m.group("seed")),
            }
    return None


def main():
    print(f"Scanning logs in: {LOGS_DIR}")
    logs = sorted(LOGS_DIR.glob("*.log"))
    if not logs:
        print("No log files found. Run quick tests first.")
        return

    rows = []
    for log_path in logs:
        parsed = parse_log(log_path)
        expected = parse_filename(log_path.stem)

        row = {
            "log_file": log_path.name,
            "dataset_expected": expected["dataset_expected"] if expected else "?",
            "arch_expected": expected["arch_expected"] if expected else "?",
            "gamma_expected": expected["gamma_expected"] if expected else "?",
            "seed_expected": expected["seed_expected"] if expected else "?",
            "dataset_found": parsed.get("dataset_found", ""),
            "arch_found": parsed.get("arch_found", ""),
            "gamma_found": parsed.get("gamma_found", ""),
            "seed_found": parsed.get("seed_found", ""),
            "run_config_match": "",
            "has_final_result": "yes" if parsed.get("final_result") else "no",
            "acc": parsed.get("acc", ""),
            "nmi": parsed.get("nmi", ""),
            "ari": parsed.get("ari", ""),
            "status": parsed["status"],
        }

        # Check match
        if expected and parsed.get("final_result"):
            match = (
                parsed["dataset_found"] == expected["dataset_expected"]
                and parsed["arch_found"] == expected["arch_expected"]
                and parsed["gamma_found"] == expected["gamma_expected"]
                and parsed["seed_found"] == expected["seed_expected"]
            )
            row["run_config_match"] = "MATCH" if match else "MISMATCH"
        elif expected and parsed.get("run_config"):
            # Try to extract from RUN_CONFIG
            rc = parsed["run_config"]
            match = all(
                f"{k}={v}" in rc
                for k, v in [
                    ("dataset", expected["dataset_expected"]),
                    ("arch", str(expected["arch_expected"])),
                    ("gamma", str(expected["gamma_expected"])),
                    ("seed", str(expected["seed_expected"])),
                ]
            )
            row["run_config_match"] = "MATCH" if match else "FALLBACK_CHECK"

        rows.append(row)

    # Write CSV
    csv_path = RESULTS_DIR / "stage5_run_config_check.csv"
    fieldnames = [
        "log_file", "dataset_expected", "arch_expected", "gamma_expected",
        "seed_expected", "dataset_found", "arch_found", "gamma_found",
        "seed_found", "run_config_match", "has_final_result",
        "acc", "nmi", "ari", "status",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    print(f"\nResults written to: {csv_path}")
    for row in rows:
        match_str = row["run_config_match"]
        status_icon = "✅" if match_str == "MATCH" else "⚠️" if match_str else "❌"
        print(f"  {status_icon} {row['log_file']}: {match_str} | status={row['status']}")
        if row["has_final_result"] == "yes":
            print(f"     FINAL_RESULT: Acc={row['acc']}, nmi={row['nmi']}, ari={row['ari']}")

    all_match = all(r["run_config_match"] == "MATCH" for r in rows if r["status"] == "complete")
    print(f"\nAll match: {all_match}")


if __name__ == "__main__":
    main()
