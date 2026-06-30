#!/usr/bin/env python3
"""Parse stage 6 quick training logs. Output: results/stage6_quick_training_results.csv"""

import re
import csv
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs" / "stage6_quick_training"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_filename_meta(filename):
    """Extract dataset, arch, gamma, seed, run_id from filename."""
    # Pattern: BDGP_arch10_gamma0.1_seed0_run1.log or BDGP_arch10_gamma0.1_seed1.log
    m = re.match(
        r"(?P<dataset>\S+?)_arch(?P<arch>\d+)_gamma(?P<gamma>[\d.]+)_seed(?P<seed>\d+)(?:_run(?P<run_id>\d+))?\.log",
        filename,
    )
    if m:
        return {
            "dataset": m.group("dataset"),
            "arch": int(m.group("arch")),
            "gamma": float(m.group("gamma")),
            "seed": int(m.group("seed")),
            "run_id": m.group("run_id") if m.group("run_id") else "1",
        }
    return None


def parse_log(log_path):
    """Extract RUN_CONFIG and FINAL_RESULT from a log file."""
    result = {
        "run_config_found": "no",
        "run_config_match": "",
        "final_result_found": "no",
        "acc": "",
        "nmi": "",
        "ari": "",
        "status": "unknown",
        "message": "",
    }

    if not log_path.exists():
        result["status"] = "missing"
        result["message"] = "Log file not found"
        return result

    with open(log_path, "r", errors="replace") as f:
        content = f.read()

    if "Traceback" in content or "Error" in content:
        result["status"] = "error"
        # Extract last error line
        for line in content.split("\n"):
            if "Error" in line or "Traceback" in line:
                result["message"] = line.strip()[:200]
        return result

    # Parse RUN_CONFIG
    rc_match = re.search(
        r"RUN_CONFIG\s+dataset=(\S+)\s+method=(\S+)\s+arch=(\d+)\s+gamma=([\d.]+)\s+seed=(\d+)\s+"
        r"noise=([\d.]+)\s+lr=([\d.]+)\s+n_z=(\d+)\s+save_path=(\S+)",
        content,
    )
    if rc_match:
        result["run_config_found"] = "yes"
        result["dataset_from_rc"] = rc_match.group(1)
        result["arch_from_rc"] = int(rc_match.group(3))
        result["gamma_from_rc"] = float(rc_match.group(4))
        result["seed_from_rc"] = int(rc_match.group(5))

    # Parse FINAL_RESULT
    fr_match = re.search(
        r"FINAL_RESULT\s+dataset=(\S+)\s+arch=(\d+)\s+gamma=([\d.]+)\s+seed=(\d+)\s+"
        r"Acc\s+([\d.]+),\s*nmi\s+([\d.]+),\s*ari\s+([\d.]+)",
        content,
    )
    if fr_match:
        result["final_result_found"] = "yes"
        result["acc"] = float(fr_match.group(5))
        result["nmi"] = float(fr_match.group(6))
        result["ari"] = float(fr_match.group(7))
        result["status"] = "complete"

    if result["status"] == "unknown" and result["run_config_found"] == "yes":
        result["status"] = "no_final_result"

    return result


def main():
    logs = sorted(LOGS_DIR.glob("BDGP_*.log")) + sorted(LOGS_DIR.glob("HW_*.log"))
    if not logs:
        print("No stage 6 logs found.")
        return

    rows = []
    for log_path in logs:
        meta = parse_filename_meta(log_path.name)
        parsed = parse_log(log_path)

        row = {
            "log_file": log_path.name,
            "dataset": meta["dataset"] if meta else "?",
            "arch": meta["arch"] if meta else "",
            "gamma": meta["gamma"] if meta else "",
            "seed": meta["seed"] if meta else "",
            "noise": 0,
            "run_id": meta["run_id"] if meta else "",
            "run_config_found": parsed["run_config_found"],
            "run_config_match": "",
            "final_result_found": parsed["final_result_found"],
            "acc": parsed["acc"],
            "nmi": parsed["nmi"],
            "ari": parsed["ari"],
            "status": parsed["status"],
            "message": parsed.get("message", ""),
        }

        # Check RUN_CONFIG match
        if meta and parsed.get("dataset_from_rc"):
            match = (
                parsed["dataset_from_rc"] == meta["dataset"]
                and parsed["arch_from_rc"] == meta["arch"]
                and parsed["gamma_from_rc"] == meta["gamma"]
                and parsed["seed_from_rc"] == meta["seed"]
            )
            row["run_config_match"] = "MATCH" if match else "MISMATCH"

        rows.append(row)

    # Write CSV
    csv_path = RESULTS_DIR / "stage6_quick_training_results.csv"
    fieldnames = [
        "log_file", "dataset", "arch", "gamma", "seed", "noise", "run_id",
        "run_config_found", "run_config_match", "final_result_found",
        "acc", "nmi", "ari", "status", "message",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    print(f"Parsed {len(rows)} logs → {csv_path}")
    for r in rows:
        icon = "✅" if r["status"] == "complete" else "❌"
        rc_icon = "✅" if r["run_config_match"] == "MATCH" else "⚠️"
        result_str = ""
        if r["final_result_found"] == "yes":
            result_str = f" Acc={r['acc']:.4f} nmi={r['nmi']:.4f} ari={r['ari']:.4f}"
        print(f"  {icon} {r['log_file']}: {r['status']} {rc_icon}{result_str}")
        if r["message"]:
            print(f"     message: {r['message']}")


if __name__ == "__main__":
    main()
