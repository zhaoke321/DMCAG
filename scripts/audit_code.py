#!/usr/bin/env python3
"""Code audit script for DMCAG reproducibility experiment.
Scans training scripts for hard-coded overrides, seed issues, KMeans config, gamma type, and Y label handling.
Outputs: reports/code_audit.md
"""

import re
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_SCRIPTS = [
    "train_fc.py",
    "train_conv_mnist-usps.py",
    "train_conv_multi-coil-10.py",
]
UTILS_FILES = [
    "utils.py",
]

# -------- Scanner helpers --------

def find_lines(filepath, pattern):
    """Return list of (lineno, stripped_line) matching regex pattern."""
    hits = []
    with open(filepath, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines, 1):
        if re.search(pattern, line):
            hits.append((i, line.rstrip("\n")))
    return hits


def find_block(filepath, start_pattern, end_pattern=None, max_lines=50):
    """Return list of lines between start_pattern and end_pattern (inclusive)."""
    block = []
    inside = False
    with open(filepath, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines, 1):
        if re.search(start_pattern, line):
            inside = True
        if inside:
            block.append((i, line.rstrip("\n")))
            if end_pattern and re.search(end_pattern, line):
                break
            if len(block) > max_lines and end_pattern is None:
                break
    return block


# -------- Audit functions --------

def audit_hardcoded_overrides(filepath):
    """Check for hard-coded argument overrides after parse_args."""
    findings = []
    with open(filepath, "r") as f:
        content = f.read()

    # Find the line where parse_args is called
    parse_line = None
    for i, line in enumerate(content.split("\n"), 1):
        if "parse_args()" in line:
            parse_line = i
            break

    if parse_line is None:
        return findings

    # Check lines after parse_args for hard-coded assignments
    # Only flag actual assignments (single =, NOT conditionals with ==)
    override_patterns = [
        (r"args\.dataset\s*=\s*['\"]", "args.dataset"),
        (r"args\.method\s*=\s*['\"]", "args.method"),
        (r"args\.arch\s*=\s*\d", "args.arch"),
        (r"args\.gamma\s*=\s*[\d.]", "args.gamma"),
        (r"args\.noise\s*=\s*[\d.]", "args.noise"),
    ]

    for pattern, name in override_patterns:
        hits = find_lines(filepath, pattern)
        for lineno, line in hits:
            if lineno > parse_line and "add_argument" not in line:
                # Skip conditional checks (if args.dataset == 'X')
                stripped = line.strip()
                if stripped.startswith("if ") or stripped.startswith("elif "):
                    continue
                # Skip inside apply_dataset_config or DATASET_CONFIG
                findings.append({
                    "issue": "硬编码参数覆盖",
                    "lineno": lineno,
                    "code": line.strip(),
                    "detail": f"{name} 在 parse_args() 后被硬编码覆盖",
                    "suggestion": "移除硬编码，改用 config_utils.apply_dataset_config() 或命令行参数",
                })
    return findings


def audit_setup_seed(filepath):
    """Check setup_seed calls — only flag hardcoded seeds or missing calls."""
    findings = []

    setup_calls = find_lines(filepath, r"setup_seed\(")

    if not setup_calls:
        has_def = find_lines(filepath, r"def setup_seed")
        if has_def:
            findings.append({
                "issue": "setup_seed 未被调用",
                "lineno": has_def[0][0],
                "code": "setup_seed 已定义但 main 中未调用",
                "detail": "文件定义了 setup_seed 但 main 中未调用",
                "suggestion": "在 main 中显式调用 setup_seed(args.seed)",
            })
    else:
        for lineno, line in setup_calls:
            # Only flag if using hardcoded integer (e.g. setup_seed(200), setup_seed(3))
            # Do NOT flag setup_seed(args.seed) or setup_seed(SEED) — SEED is checked separately
            hardcoded_match = re.search(r"setup_seed\((\d+)\)", line)
            if hardcoded_match:
                findings.append({
                    "issue": "setup_seed 使用硬编码 seed",
                    "lineno": lineno,
                    "code": line.strip(),
                    "detail": f"seed={hardcoded_match.group(1)} 硬编码，应使用 args.seed",
                    "suggestion": "改为 setup_seed(args.seed)",
                })

    return findings


def audit_kmeans(filepath):
    """Check all KMeans calls — only flag those WITHOUT random_state."""
    findings = []
    kmeans_lines = find_lines(filepath, r"KMeans\(.*\)")
    seen = set()
    for lineno, line in kmeans_lines:
        if "random_state" not in line:
            key = line.strip()
            if key not in seen:
                seen.add(key)
                findings.append({
                    "issue": "KMeans 缺少 random_state",
                    "lineno": lineno,
                    "code": line.strip(),
                    "detail": "KMeans 未设置 random_state，每次运行结果可能不同",
                    "suggestion": "添加 random_state=args.seed",
                })
        # If it has random_state, do NOT report (it's correctly set)
    return findings


def audit_gamma_type(filepath):
    """Check whether gamma is defined as int."""
    findings = []
    gamma_lines = find_lines(filepath, r"gamma.*type\s*=\s*int")
    for lineno, line in gamma_lines:
        findings.append({
            "issue": "gamma 定义为 int 类型",
            "lineno": lineno,
            "code": line.strip(),
            "detail": "gamma 应支持 float (论文要求 {0.1, 1, 10})，int 类型导致 0.1 无法传入",
            "suggestion": "改为 type=float",
        })
    # Also check default value
    default_lines = find_lines(filepath, r"gamma.*default\s*=\s*\d+")
    for lineno, line in default_lines:
        if "default=0.1" not in line and "default=1.0" not in line:
            # int default is fine if type is float, but check
            pass
    return findings


def audit_y_handling(filepath):
    """Check how Y labels are read and reshaped."""
    findings = []
    # Find Y reading patterns
    y_read = find_lines(filepath, r"matData\[.Y.\]|Y\s*=\s*matData|lbl|labels")
    for lineno, line in y_read:
        if "Y = matData" in line.replace(" ", ""):
            findings.append({
                "issue": "Y 标签读取方式",
                "lineno": lineno,
                "code": line.strip(),
                "detail": f"当前读取: {line.strip()}。如果 Y shape 是 [n,1]，则 [0] 取出的是第一行而非数组",
                "suggestion": "使用 normalize_label(y) 统一处理 Y 的 shape 问题",
            })
    # Check for .ravel() or .flatten() or .squeeze() usage on Y
    squeeze_hits = find_lines(filepath, r"\.squeeze|\.ravel|\.flatten|np\.squeeze")
    for lineno, line in squeeze_hits:
        if any(kw in line.lower() for kw in ["y", "label", "lbl"]):
            findings.append({
                "issue": "Y reshape 操作",
                "lineno": lineno,
                "code": line.strip(),
                "detail": "发现 Y 相关的 reshape 操作",
                "suggestion": "确保统一使用 normalize_label 函数",
            })
    return findings


def audit_other_issues(filepath):
    """Catch other reproducibility issues."""
    findings = []
    # Check for SEED global variable usage
    seed_global = find_lines(filepath, r"^SEED\s*=")
    for lineno, line in seed_global:
        findings.append({
            "issue": "SEED 全局变量",
            "lineno": lineno,
            "code": line.strip(),
            "detail": "不应使用全局 SEED，应使用 args.seed",
            "suggestion": "删除 SEED = ... 行",
        })
    seed_refs = find_lines(filepath, r"random_state=SEED|setup_seed\(SEED\)")
    for lineno, line in seed_refs:
        findings.append({
            "issue": "使用了全局 SEED",
            "lineno": lineno,
            "code": line.strip(),
            "detail": "random_state=SEED 或 setup_seed(SEED) 应改为 args.seed",
            "suggestion": "改为 random_state=args.seed 或 setup_seed(args.seed)",
        })
    # Duplicate imports
    dup_os = find_lines(filepath, r"^import os$")
    if len(dup_os) > 1:
        for lineno, line in dup_os[1:]:
            findings.append({
                "issue": "重复 import",
                "lineno": lineno,
                "code": line.strip(),
                "detail": f"os 在第 {dup_os[0][0]} 行已导入",
                "suggestion": "删除重复的 import os",
            })

    # Hard-coded GPU device
    gpu_lines = find_lines(filepath, r'CUDA_VISIBLE_DEVICES\s*=\s*"0"')
    for lineno, line in gpu_lines:
        findings.append({
            "issue": "GPU 设备硬编码",
            "lineno": lineno,
            "code": line.strip(),
            "detail": "CUDA_VISIBLE_DEVICES 硬编码为 0",
            "suggestion": "改为通过命令行参数或环境变量控制",
        })

    # Hard-coded cpu threads
    cpu_lines = find_lines(filepath, r"cpu_num\s*=\s*\d+")
    for lineno, line in cpu_lines:
        findings.append({
            "issue": "CPU 线程数硬编码",
            "lineno": lineno,
            "code": line.strip(),
            "detail": "cpu_num 硬编码",
            "suggestion": "改为可配置参数",
        })

    return findings


# -------- Report generation --------

def generate_report(all_findings):
    """Generate audit report in Markdown format."""
    lines = []
    lines.append("# DMCAG 代码审计报告\n")
    lines.append(f"审计日期: 2026-06-29\n")
    lines.append(f"审计范围: {', '.join(TRAIN_SCRIPTS + UTILS_FILES)}\n")
    lines.append("---\n")
    lines.append("## 审计摘要\n")
    lines.append("| 文件 | 硬编码覆盖 | seed问题 | KMeans无random_state | gamma=int | Y处理 | 其他 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for fname in TRAIN_SCRIPTS + UTILS_FILES:
        f_findings = [f for f in all_findings if f["file"] == fname]
        n_hardcode = sum(1 for f in f_findings if "硬编码" in f["issue"])
        n_seed = sum(1 for f in f_findings if "seed" in f["issue"].lower())
        n_kmeans = sum(1 for f in f_findings if "KMeans" in f["issue"])
        n_gamma = sum(1 for f in f_findings if "gamma" in f["issue"])
        n_y = sum(1 for f in f_findings if "Y" in f["issue"] or "label" in f["issue"].lower())
        n_other = len(f_findings) - n_hardcode - n_seed - n_kmeans - n_gamma - n_y
        lines.append(f"| `{fname}` | {n_hardcode} | {n_seed} | {n_kmeans} | {n_gamma} | {n_y} | {n_other} |")

    lines.append("\n---\n")
    lines.append("## 详细发现\n")

    current_file = None
    for fname in TRAIN_SCRIPTS + UTILS_FILES:
        f_findings = [f for f in all_findings if f["file"] == fname]
        if not f_findings:
            continue

        lines.append(f"### 📄 `{fname}`\n")
        lines.append("| # | 问题类型 | 行号 | 原始代码 | 处理建议 |")
        lines.append("|---|---:|---|---|")
        for idx, f in enumerate(f_findings, 1):
            code = f["code"].replace("|", "\\|")[:100]
            suggestion = f.get("suggestion", f.get("detail", ""))
            lines.append(f"| {idx} | **{f['issue']}** | {f['lineno']} | `{code}` | {suggestion} |")
        lines.append("")

    # Cross-file duplication summary
    lines.append("---\n")
    lines.append("## 跨文件代码重复分析\n")
    lines.append("| 代码块 | 出现文件数 | 重复行数 | 建议 |")
    lines.append("|---|---:|---:|")
    dup_blocks = [
        ("Preamble (imports, os.environ, scalers)", 3, "~30 行/文件", "移入 common.py 统一导入"),
        ("ClusteringLayer 类", 3, "~13 行/文件", "移入 utils.py"),
        ("computeA / computegcn (类方法 + 独立函数)", 3, "~45 行/文件", "移入 utils.py，消除类内外重复"),
        ("make_qp / target_distribution / graph_fusion", 3, "~25 行/文件", "移入 utils.py"),
        ("calculate_c / quadprog / cacluate_U", 3, "~75 行/文件", "移入 utils.py"),
        ("mask_correlated_samples / embeddingcontras / labelcontras", 3, "~65 行/文件", "移入 utils.py"),
        ("setup_seed", 3, "~6 行/文件", "移入 utils.py"),
        ("pretrain_aes / fineTuning", 3, "~160 行/文件，~70% 相同", "合并为参数化函数，消除差异"),
        ("multiViewDataset2 / imagedataset", 1, "2 个类 95% 相同 (utils.py)", "合并为带参数的单一类"),
    ]
    for name, n_files, n_lines, suggestion in dup_blocks:
        lines.append(f"| {name} | {n_files} | {n_lines} | {suggestion} |")

    lines.append(f"\n**总重复行数估计**: 约 {30*3 + 13*3 + 45*3 + 25*3 + 75*3 + 65*3 + 6*3} 行 → 建议全部合并至 utils.py 或新建 common.py\n")

    # Specific findings by dataset
    lines.append("---\n")
    lines.append("## 关键发现：数据集配置差异\n")
    lines.append("### seed 设置汇总\n")
    lines.append("| 数据集 | 训练文件 | setup_seed 值 | 位置 |")
    lines.append("|---|---|---:|")
    seed_data = [
        ("BDGP", "train_fc.py", "未调用", ""),
        ("HW", "train_fc.py", "未调用", ""),
        ("Fmnist-MV", "train_fc.py", "200", "条件分支内"),
        ("UCI-3V", "train_fc.py", "未调用", ""),
        ("MNIST_USPS", "train_conv_mnist-usps.py", "3", "条件分支内"),
        ("Multi-COIL-10", "train_conv_multi-coil-10.py", "1000", "条件分支前（全局生效）"),
    ]
    for ds, script, seed, loc in seed_data:
        lines.append(f"| {ds} | `{script}` | {seed} | {loc} |")

    lines.append("\n### gamma 值汇总（硬编码后实际值）\n")
    lines.append("| 数据集 | 训练文件 | args.gamma (argparse default) | 硬编码覆盖后的值 |")
    lines.append("|---|---|---:|")
    gamma_data = [
        ("BDGP", "train_fc.py", "5 (int)", "10"),
        ("HW", "train_fc.py", "5 (int)", "0.1"),
        ("Fmnist-MV", "train_fc.py", "5 (int)", "1"),
        ("UCI-3V", "train_fc.py", "5 (int)", "1"),
        ("MNIST_USPS", "train_conv_mnist-usps.py", "5 (int)", "1"),
        ("Multi-COIL-10", "train_conv_multi-coil-10.py", "5 (int)", "1"),
    ]
    for ds, script, dflt, actual in gamma_data:
        lines.append(f"| {ds} | `{script}` | {dflt} | {actual} |")

    lines.append("\n### 预测方法差异\n")
    lines.append("| 文件 | pretrain_aes 最终预测 | fineTuning 最终预测 |")
    lines.append("|---|---|")
    lines.append("| `train_fc.py` | **未使用**（评估代码被注释） | KMeans(f_all) |")
    lines.append("| `train_conv_mnist-usps.py` | **未使用**（评估代码被注释） | KMeans(f_all) |")
    lines.append("| `train_conv_multi-coil-10.py` | KMeans(f_temp) + KMeans(f_all) | **argmax(qpred)** |")

    lines.append("\n---\n")
    lines.append("## 修复优先级\n")
    lines.append("| 优先级 | 问题 | 影响范围 | 建议修复阶段 |")
    lines.append("|---|---|---|---|")
    lines.append("| 🔴 P0 | **gamma 定义为 int** | 所有数据集 | 阶段 3 |")
    lines.append("| 🔴 P0 | **Y 标签 shape 不安全** | 用户数据集 | 阶段 3 |")
    lines.append("| 🔴 P0 | **KMeans 无 random_state** | 所有结果不可复现 | 阶段 3 |")
    lines.append("| 🔴 P0 | **seed 不一致** | 跨数据集公平性 | 阶段 4 |")
    lines.append("| 🟡 P1 | **args 硬编码覆盖** | 超参扫描失效 | 阶段 4 |")
    lines.append("| 🟡 P1 | **预测方法不一致** (KMeans vs argmax) | Multi-COIL-10 结果 | 阶段 4 |")
    lines.append("| 🟢 P2 | **大量代码重复** | 维护性 | 阶段 4+ |")
    lines.append("| 🟢 P2 | **GPU/CPU 硬编码** | 环境适配 | 阶段 4+ |")

    lines.append("\n---\n")
    lines.append("*报告由 scripts/audit_code.py 自动生成*\n")

    return "\n".join(lines)


# -------- Stage 4 audit functions --------

def audit_metrics_imports(filepath):
    """Check that training scripts import from metrics_utils."""
    findings = []
    with open(filepath, "r") as f:
        content = f.read()

    # Check for old-style imports
    old_patterns = [
        (r"from sklearn\.metrics\.cluster import.*nmi_score", "旧版 sklearn nmi_score 导入"),
        (r"from sklearn\.metrics import.*ari_score", "旧版 sklearn ari_score 导入"),
        (r"from utils import.*cluster_acc", "旧版 utils.cluster_acc 导入（应改用 metrics_utils）"),
    ]
    for pat, desc in old_patterns:
        hits = find_lines(filepath, pat)
        for lineno, line in hits:
            findings.append({
                "issue": f"旧版指标导入",
                "lineno": lineno,
                "code": line.strip(),
                "detail": desc,
                "suggestion": "删除此行，改用 from metrics_utils import clustering_metrics, check_metric_inputs",
            })

    # Check that metrics_utils is imported
    if "train_" in str(filepath) and filepath.endswith(".py"):
        has_metrics_utils = bool(find_lines(filepath, r"from metrics_utils import"))
        if not has_metrics_utils:
            findings.append({
                "issue": "缺少 metrics_utils 导入",
                "lineno": 1,
                "code": "N/A",
                "detail": "训练脚本未导入 metrics_utils",
                "suggestion": "添加 from metrics_utils import clustering_metrics, check_metric_inputs",
            })

    return findings


def audit_dangerous_label_patterns(filepath):
    """Scan for dangerous Y/label patterns."""
    findings = []
    dangerous = [
        (r"matData\[.Y.\]\[0\]", "matData['Y'][0] 不安全读取"),
        (r"\bY\s*=\s*Y\[0\]", "Y = Y[0] 可能截断标签"),
        (r"\.reshape\(-1,\s*1\)", ".reshape(-1,1) 可能产生二维标签（检查是否用于 labels）"),
    ]
    with open(filepath, "r") as f:
        lines = f.readlines()
    for pat, desc in dangerous:
        hits = find_lines(filepath, pat)
        for lineno, line in hits:
            # Skip commented lines
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            findings.append({
                "issue": "危险标签写法",
                "lineno": lineno,
                "code": line.strip(),
                "detail": desc,
                "suggestion": "使用 ensure_1d_label() 或 normalize_label() 统一处理",
            })
    return findings


def audit_final_result_format(filepath):
    """Check that FINAL_RESULT format is used."""
    findings = []
    with open(filepath, "r") as f:
        content = f.read()

    if "train_" not in str(filepath):
        return findings

    # Check for FINAL_RESULT
    has_final = "FINAL_RESULT" in content
    # Check for old print format
    old_prints = find_lines(filepath, r"print\('Acc \{:\.4f\}'.format")
    for lineno, line in old_prints:
        if "FINAL_RESULT" not in line:
            findings.append({
                "issue": "旧版输出格式",
                "lineno": lineno,
                "code": line.strip(),
                "detail": "未使用 FINAL_RESULT 统一输出格式",
                "suggestion": "改为 print(f\"FINAL_RESULT dataset=... Acc {acc:.4f}, ...\")",
            })

    if not has_final and "train_" in str(filepath):
        findings.append({
            "issue": "缺少 FINAL_RESULT",
            "lineno": 1,
            "code": "N/A",
            "detail": "训练脚本未输出 FINAL_RESULT 格式",
            "suggestion": "最终评估处添加 FINAL_RESULT 输出",
        })

    return findings


def main():
    all_findings = []

    for fname in TRAIN_SCRIPTS + UTILS_FILES:
        fpath = PROJECT_ROOT / fname
        if not fpath.exists():
            print(f"⚠️  {fname} 不存在，跳过")
            continue

        # Run all audit checks
        checks = [
            audit_hardcoded_overrides,
            audit_setup_seed,
            audit_kmeans,
            audit_gamma_type,
            audit_y_handling,
            audit_other_issues,
            audit_metrics_imports,
            audit_dangerous_label_patterns,
            audit_final_result_format,
        ]

        for check_fn in checks:
            try:
                findings = check_fn(str(fpath))
                for f in findings:
                    f["file"] = fname
                all_findings.extend(findings)
            except Exception as e:
                print(f"❌ {check_fn.__name__}({fname}) 出错: {e}")

    # Sort by file then lineno
    all_findings.sort(key=lambda x: (TRAIN_SCRIPTS + UTILS_FILES).index(x["file"]) if x["file"] in TRAIN_SCRIPTS + UTILS_FILES else 99)

    # Generate report
    report = generate_report(all_findings)

    # Write report
    report_path = PROJECT_ROOT / "reports" / "code_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ 审计报告已生成: {report_path}")
    print(f"   共发现 {len(all_findings)} 个问题")

    # Print summary
    for fname in TRAIN_SCRIPTS + UTILS_FILES:
        count = sum(1 for f in all_findings if f["file"] == fname)
        print(f"   {fname}: {count} 个问题")


if __name__ == "__main__":
    main()
