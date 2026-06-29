你现在作为我的实验执行助手，负责完成 DMCAG 论文的横向对比复现实验。请严格按阶段执行，不要一次性全量开跑。每完成一个阶段，输出修改文件、运行命令、日志路径、检查结果。

论文：Deep Multi-View Subspace Clustering with Anchor Graph, DMCAG。

当前已知问题：
1. GitHub README 中的 Acc 0.8564, nmi 0.7321, ari 0.6459 只是示例输出，不能视为论文 Table 2 最终结果。
2. 官方代码中不同数据集 seed 设置不一致：有的调用 setup_seed，有的没调用，有的 seed=200/3/1000。
3. 官方代码存在命令行参数被硬编码覆盖的问题，例如 args.dataset、args.arch、args.gamma 在脚本内部被重新赋值。
4. 论文要求 gamma 扫描 {0.1, 1, 10}，但代码里 gamma 可能被定义成 int，导致 0.1 无法正常传入。
5. 我的数据集中标签 Y 的 shape 是 [n,1]，而作者代码默认可能按 [1,n] 读取。必须统一处理成一维标签 [n]，否则 ACC/NMI/ARI 计算会错。
6. 横向对比实验必须统一数据读取、seed、指标计算、参数记录和日志保存。

论文实验设定：
- 数据集包括 MNIST-USPS、Multi-COIL-10、BDGP、UCI-digits、Fashion-MV、HW。
- 评价指标为 ACC、NMI、ARI，数值越高越好。
- 图像数据 MNIST-USPS、Multi-COIL-10 使用卷积自编码器。
- 向量数据 BDGP、UCI-digits、Fashion-MV、HW 使用全连接自编码器。
- anchor number 在 [10,100] 范围内选择。
- gamma 从 {0.1, 1, 10} 中选择。
- tau=1，alpha=0.001。
- 论文表格只给单值，没有 mean±std，所以本次复现必须同时报告 best single run 和 mean±std。

阶段 1：建立可追踪实验分支

1. 新建分支：

git checkout -b repro_fair_comparison

2. 记录原始代码版本：

git rev-parse HEAD > results_git_commit.txt

3. 创建目录：

mkdir -p scripts
mkdir -p results
mkdir -p logs
mkdir -p reports
mkdir -p debug_outputs

4. 不要覆盖原始训练脚本。修改前先备份：

cp train_fc.py train_fc.py.bak
cp train_conv_mnist-usps.py train_conv_mnist-usps.py.bak
cp train_conv_multi-coil-10.py train_conv_multi-coil-10.py.bak

阶段 2：先做代码审计，不急着改

请写 scripts/audit_code.py，扫描以下内容：

1. 所有训练脚本中是否存在硬编码覆盖：
- args.dataset =
- args.method =
- args.arch =
- args.gamma =
- setup_seed(...)

2. 所有 KMeans 是否设置 random_state。

3. gamma 是否定义为 int。

4. 标签 Y 是如何读取和 reshape 的。

输出 reports/code_audit.md，格式如下：

| 文件 | 问题类型 | 行号 | 原始代码 | 处理建议 |
|---|---|---:|---|---|

必须重点检查：
- train_fc.py
- train_conv_mnist-usps.py
- train_conv_multi-coil-10.py
- utils.py
- load_data 相关文件
- metrics 相关文件

阶段 3：修复数据读取，解决 Y shape 不一致

我的数据标签 Y 是 [n,1]，作者代码可能默认 [1,n]。请统一写一个安全函数，不允许在各脚本里重复乱写。

在 utils.py 或新建 data_utils.py 中加入：

```python
import numpy as np

def normalize_label(y):
    """
    Convert label array to shape [n].
    Acceptable input shapes:
    [n, 1], [1, n], [n].
    """
    y = np.asarray(y)
    y = np.squeeze(y)

    if y.ndim != 1:
        raise ValueError(f"Label Y must be converted to 1-D, but got shape {y.shape}")

    return y.astype(np.int64)


def normalize_view(x, n_samples=None, view_name=""):
    """
    Ensure each view is sample-first: [n, d].
    Some .mat files may store features as [d, n].
    We detect and transpose only when necessary.
    """
    x = np.asarray(x)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if n_samples is not None:
        if x.shape[0] == n_samples:
            return x
        if x.shape[1] == n_samples:
            return x.T

        raise ValueError(
            f"View {view_name} shape {x.shape} does not match n_samples={n_samples}"
        )

    return x