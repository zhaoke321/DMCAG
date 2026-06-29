# DMCAG 代码重复分析报告

> 所有重复代码已在源文件中用 `[DUPLICATE]` / `[PARTIAL-DUPLICATE]` / `[DUPLICATE-BUG]` 标记。

---

## 概览

| 指标 | 数值 |
|------|------|
| 总代码行数 (4 个 .py 文件) | ~2,264 行 |
| 完全重复的行数 (含 3 次复制) | ~530 行 |
| 部分重复的行数 (pretrain_aes, fineTuning) | ~300 行 |
| 重复率 (含部分重复) | **~60-65%** |
| 可削减到的合理行数 (重构后) | ~800-1000 行 |

---

## 一、3 个训练文件完全相同的代码块（逐字节一致）

以下代码出现在 `train_fc.py` / `train_conv_mnist-usps.py` / `train_conv_multi-coil-10.py` 三个文件中完全一致：

### 1.1 导入 + 环境设置（~28 行 × 3 = ~84 行浪费）
| 文件 | 行号范围 |
|------|----------|
| `train_fc.py` | 1-39 (含标记) |
| `train_conv_mnist-usps.py` | 1-38 (含标记) |
| `train_conv_multi-coil-10.py` | 1-38 (含标记) |

额外 Bug：`min_max_scaler` / `normalize` 被重复定义两次（3 文件均存在）。

### 1.2 `ClusteringLayer` 类（11 行 × 3 = 33 行）
中心点聚类层，完全一致。

### 1.3 `SingleViewModel.computeA()`（22 行 × 3 = 66 行）
图构建方法，支持 cos / kernel / knn / sigmod 四种模式。

### 1.4 `SingleViewModel.computegcn()`（8 行 × 3 = 24 行）
图卷积方法（与独立函数版 `computegcn` 也重复）。

### 1.5 独立辅助函数（~177 行 × 3 = ~531 行）
| 函数 | 行数 | 用途 |
|------|------|------|
| `computegcn()` | 8 | 图卷积（与 1.4 完全相同） |
| `make_qp()` | 5 | 计算软标签 q 和目标分布 p |
| `target_distribution()` | 3 | 目标分布变换 |
| `graph_fusion()` | 8 | 多视图图融合 |
| `calculate_c()` | 40 | QP 求解系数矩阵 c |
| `quadprog()` | 30 | MATLAB quadprog 的 cvxopt 封装 |
| `cacluate_U()` | 5 | SVD 分解取 top-k 奇异向量 |

### 1.6 对比学习函数（~68 行 × 3 = ~204 行）
| 函数 | 行数 | 用途 |
|------|------|------|
| `mask_correlated_samples()` | 7 | 构造对比学习 mask |
| `embeddingcontras()` | 19 | 嵌入层对比损失 |
| `labelcontras()` | 28 | 标签层对比损失 + 熵正则 |

### 1.7 `setup_seed()`（6 行 × 3 = 18 行）
随机种子设置。

### 1.8 `MultiViewModel` 类 — `forward()` + 全局参数（~35 行 × 3 = ~105 行）
构造函数签名略有不同，但 `forward()` 方法和 `Al_weight`/`cl_weight` 参数完全一致。

---

## 二、3 个训练文件部分相同（结构一致，少量参数不同）

### 2.1 `pretrain_aes()`（~60 行 × 3 = ~180 行，~70% 相同）
**差异点：**
- 使用的 Dataset 类不同（`multiViewDataset2` vs `imagedataset`）
- 是否使用 `tqdm` 进度条
- 预训练后评估代码：train_fc.py **注释掉** / train_conv_mnist-usps.py **注释掉** / train_conv_multi-coil-10.py **激活**

### 2.2 `fineTuning()`（~140 行 × 3 = ~420 行，~70% 相同）
**差异点：**
- `view_loss` 权重：train_fc.py: `0.1` / train_conv_mnist-usps.py: `0` / train_conv_multi-coil-10.py: **无此项**
- 自监督阶段 epoch 边界条件：`if epoch==0` vs `if epoch<=500` vs `if epoch<=1000`
- 最终预测方法：train_fc.py: `KMeans on f_all` / train_conv_mnist-usps.py: `KMeans on f_all` / train_conv_multi-coil-10.py: `argmax(qpred)`
- 是否使用 `tqdm` / 打印阶段横幅

### 2.3 `argparse + main`（~30 行 × 3 = ~90 行，~80% 相同）
仅数据集配置参数不同。

---

## 三、utils.py 内部重复

### `multiViewDataset2` vs `imagedataset`（~30 行 × 2 = ~60 行）
两个类 95% 相同，**唯一差异**是 MinMaxScaler 的触发阈值：
- `multiViewDataset2`: `if self.viewNumber >= 2`
- `imagedataset`: `if self.viewNumber >= 6`

建议合并为带 `scaler_threshold` 参数的单一类。

---

## 四、额外 Bug 发现

| 文件 | 行号 | 问题 |
|------|------|------|
| 全部 3 个训练文件 | ~23-28 | `min_max_scaler` 和 `normalize` 被定义两次 |
| 全部 3 个训练文件 | ~31 | `import os` 重复（第 3 行已导入） |
| `train_fc.py` | ~27 | 未使用的 `from utils import WKLDiv`（仅 `WKLDiv` 在 train_fc.py 中未被使用） |

---

## 五、推荐重构方案

```
utils.py (共享模块)
├── ClusteringLayer          ← 从 train_*.py 移入
├── computeA()              ← 从 SingleViewModel 移入
├── computegcn()            ← 从 3 文件合并
├── make_qp()               ← 从 3 文件合并
├── target_distribution()   ← 从 3 文件合并
├── graph_fusion()          ← 从 3 文件合并
├── calculate_c()           ← 从 3 文件合并
├── quadprog()              ← 从 3 文件合并
├── cacluate_U()            ← 从 3 文件合并
├── mask_correlated_samples() ← 从 3 文件合并
├── embeddingcontras()      ← 从 3 文件合并
├── labelcontras()          ← 从 3 文件合并
├── setup_seed()            ← 从 3 文件合并
└── MultiViewDataset (合并 multiViewDataset2 + imagedataset)

models.py (模型定义)
├── ClusteringLayer
├── BaseAutoEncoder
│   ├── FCAutoEncoder (原 train_fc.py 的 SingleViewModel)
│   └── ConvAutoEncoder (参数化 latent_h/latent_w)

train.py (统一训练脚本)
├── pretrain_aes(model_type='fc'|'conv')
├── fineTuning(model_type='fc'|'conv')
└── 数据集配置从 YAML/JSON 配置文件加载
```

预期效果：代码行数从 ~2,264 行降至 ~800-1000 行，消除所有重复。
