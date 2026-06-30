#!/usr/bin/env python3
"""Stage 7: DMCAG grid search for BDGP and HW datasets."""

import argparse
import csv
import itertools
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/home/ubuntu/miniconda3/envs/zk/bin/python3"

RUN_CONFIG_PATTERN = re.compile(
    r"RUN_CONFIG.*?dataset=(?P<dataset>\S+).*?"
    r"arch=(?P<arch>[0-9]+).*?"
    r"gamma=(?P<gamma>[0-9.]+).*?"
    r"seed=(?P<seed>[0-9]+).*?"
    r"noise=(?P<noise>[0-9.]+)",
    re.IGNORECASE | re.DOTALL,
)

FINAL_RESULT_PATTERN = re.compile(
    r"FINAL_RESULT.*?dataset=(?P<dataset>\S+).*?"
    r"arch=(?P<arch>[0-9]+).*?"
    r"gamma=(?P<gamma>[0-9.]+).*?"
    r"seed=(?P<seed>[0-9]+).*?"
    r"Acc\s+(?P<acc>[0-9.]+).*?"
    r"nmi\s+(?P<nmi>[0-9.]+).*?"
    r"ari\s+(?P<ari>[0-9.]+)",
    re.IGNORECASE | re.DOTALL,
)


def append_csv(path, row, fieldnames):
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_done_keys(csv_path):
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("status") == "ok":
                try:
                    key = (
                        r["dataset"],
                        int(r["arch"]),
                        float(r["gamma"]),
                        int(r["seed"]),
                    )
                    done.add(key)
                except (ValueError, KeyError):
                    pass
    return done


def parse_log(text, expected_dataset, expected_arch, expected_gamma, expected_seed, expected_noise):
    run_config_found = False
    run_config_match = False
    final_result_found = False
    acc = nmi = ari = ""
    status = "failed"
    message = ""

    rc = RUN_CONFIG_PATTERN.search(text)
    if rc:
        run_config_found = True
        found_dataset = rc.group("dataset")
        found_arch = int(rc.group("arch"))
        found_gamma = float(rc.group("gamma"))
        found_seed = int(rc.group("seed"))
        found_noise = float(rc.group("noise"))

        run_config_match = (
            found_dataset == expected_dataset
            and found_arch == expected_arch
            and abs(found_gamma - expected_gamma) < 1e-12
            and found_seed == expected_seed
            and abs(found_noise - expected_noise) < 1e-12
        )
    else:
        message += "RUN_CONFIG not found; "

    fr = FINAL_RESULT_PATTERN.search(text)
    if fr:
        final_result_found = True
        try:
            acc = float(fr.group("acc"))
            nmi = float(fr.group("nmi"))
            ari = float(fr.group("ari"))
        except ValueError:
            pass
    else:
        message += "FINAL_RESULT not found; "

    if run_config_found and run_config_match and final_result_found:
        status = "ok"
        message = "ok"
    elif run_config_found and not run_config_match:
        status = "run_config_mismatch"
        message += "RUN_CONFIG mismatch"
    elif final_result_found:
        status = "partial_ok"
    elif run_config_found:
        status = "no_final_result"

    return {
        "run_config_found": run_config_found,
        "run_config_match": run_config_match,
        "final_result_found": final_result_found,
        "acc": acc,
        "nmi": nmi,
        "ari": ari,
        "status": status,
        "message": message.strip(),
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 7 DMCAG Grid Search")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 8 runs")
    parser.add_argument("--datasets", nargs="+", default=["BDGP", "HW"])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_z", type=int, default=10)
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--force", action="store_true", help="Re-run even if already done")
    args = parser.parse_args()

    valid_datasets = ["BDGP", "HW"]
    datasets = [d for d in args.datasets if d in valid_datasets]
    if not datasets:
        print(f"ERROR: No valid datasets. Choose from {valid_datasets}")
        sys.exit(1)

    if args.quick:
        arch_list = [10, 50]
        gamma_list = [0.1, 1]
        seed_list = [0]
    else:
        arch_list = [10, 20, 30, 40, 50, 60, 80, 100]
        gamma_list = [0.1, 1, 10]
        seed_list = [0, 1, 2, 3, 4]

    result_dir = PROJECT_ROOT / "results"
    log_root = PROJECT_ROOT / "logs" / "stage7_dmcag_grid"
    result_dir.mkdir(exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    csv_path = result_dir / "stage7_dmcag_grid_raw.csv"

    fieldnames = [
        "method", "dataset", "script",
        "arch", "gamma", "seed", "lr", "n_z", "noise",
        "run_config_found", "run_config_match", "final_result_found",
        "acc", "nmi", "ari",
        "status", "message", "log_file",
    ]

    done_keys = set() if args.force else load_done_keys(csv_path)
    total = len(datasets) * len(arch_list) * len(gamma_list) * len(seed_list)
    remaining = total - len(done_keys)

    print(f"Grid Search Config:")
    print(f"  Mode: {'quick' if args.quick else 'full'}")
    print(f"  Datasets: {datasets}")
    print(f"  arch: {arch_list}")
    print(f"  gamma: {gamma_list}")
    print(f"  seed: {seed_list}")
    print(f"  Total runs: {total}")
    print(f"  Already done: {len(done_keys)}")
    print(f"  Remaining: {remaining}")
    print(f"  CSV: {csv_path}")
    print()

    count = 0
    for dataset, arch_val, gamma_val, seed_val in itertools.product(datasets, arch_list, gamma_list, seed_list):
        key = (dataset, arch_val, float(gamma_val), seed_val)
        if key in done_keys:
            continue

        count += 1
        script = "train_fc.py"
        ds_log_dir = log_root / dataset
        ds_log_dir.mkdir(parents=True, exist_ok=True)

        gamma_str = str(gamma_val).replace(".", "p")
        log_file = ds_log_dir / f"{dataset}_arch{arch_val}_gamma{gamma_str}_seed{seed_val}.log"

        cmd = [
            PYTHON, str(PROJECT_ROOT / script),
            "--dataset", dataset,
            "--lr", str(args.lr),
            "--n_z", str(args.n_z),
            "--arch", str(arch_val),
            "--gamma", str(gamma_val),
            "--seed", str(seed_val),
            "--noise", str(args.noise),
        ]

        print(f"[{count}/{remaining}] {' '.join(cmd[1:])}")

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_ROOT),
                timeout=None,
            )
            text = proc.stdout or ""
            log_file.write_text(text, encoding="utf-8", errors="ignore")

            parsed = parse_log(
                text,
                expected_dataset=dataset,
                expected_arch=arch_val,
                expected_gamma=float(gamma_val),
                expected_seed=seed_val,
                expected_noise=float(args.noise),
            )

            if proc.returncode != 0 and parsed["status"] != "ok":
                parsed["status"] = f"failed_rc_{proc.returncode}"
                parsed["message"] = parsed["message"] + f" returncode={proc.returncode}"

        except Exception as e:
            log_file.write_text(str(e), encoding="utf-8", errors="ignore")
            parsed = {
                "run_config_found": False,
                "run_config_match": False,
                "final_result_found": False,
                "acc": "",
                "nmi": "",
                "ari": "",
                "status": f"exception_{type(e).__name__}",
                "message": str(e)[:200],
            }

        row = {
            "method": "DMCAG",
            "dataset": dataset,
            "script": script,
            "arch": arch_val,
            "gamma": gamma_val,
            "seed": seed_val,
            "lr": args.lr,
            "n_z": args.n_z,
            "noise": args.noise,
            "run_config_found": parsed["run_config_found"],
            "run_config_match": parsed["run_config_match"],
            "final_result_found": parsed["final_result_found"],
            "acc": parsed["acc"],
            "nmi": parsed["nmi"],
            "ari": parsed["ari"],
            "status": parsed["status"],
            "message": parsed["message"],
            "log_file": str(log_file.relative_to(PROJECT_ROOT)),
        }

        append_csv(csv_path, row, fieldnames)

        status_icon = "✅" if parsed["status"] == "ok" else "❌"
        result_str = ""
        if parsed["final_result_found"]:
            result_str = f" Acc={parsed['acc']:.4f} NMI={parsed['nmi']:.4f} ARI={parsed['ari']:.4f}"
        print(f"  {status_icon} {parsed['status']}{result_str}")

    print(f"\nDone. {count} runs executed.")
    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
