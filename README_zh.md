<div align="center">

# FastAD: 面向实时工业视觉异常检测的可部署框架

[![Paper](https://img.shields.io/badge/ICANN%202026-Paper%20%23291-005EA6.svg)](https://doi.org/10.1007/978-3-032-38401-0_22)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)

**[English](README.md)** | **[中文说明](README_zh.md)**

*ICANN 2026 论文官方开源 PyTorch 实现*  
**"FastAD: A Deployable Framework for Real-Time Visual Anomaly Detection"**

---

</div>

## 📌 项目概述

尽管近年来无监督异常检测方法在检测精度上取得了显著进展，但其在工业部署中仍面临以下两个关键问题：

**（1）计算资源消耗较高，难以满足实时部署需求。**

现有高精度方法往往依赖大规模特征记忆库、复杂的最近邻检索或多尺度特征对齐策略，导致训练周期较长、推理延迟较高、显存和存储开销较大。这使得部分实验室方法难以直接部署到边缘端或资源受限的产线设备中，限制了其在真实工业场景中的应用价值。

**（2）异常类型复杂，单一路径建模难以同时覆盖结构异常与逻辑异常。**

工业异常既包括表面划痕、裂纹、污渍等局部结构性缺陷，也包括组件缺失、位置错乱等依赖全局上下文关系的逻辑异常。前者要求模型对局部纹理和细粒度几何变化保持敏感，后者则要求模型具备对整体结构和语义关系的建模能力。若仅依赖局部特征或单一尺度表示，容易造成某类异常的漏检。

针对上述问题，本章提出一种名为 FastAD 的轻量级无监督异常检测框架，旨在实现高检测精度、低计算开销与实时推理能力的统一。该框架主要从以下两个方面进行设计：

**（1）基于经验累积分布函数的紧凑统计描述符。**

FastAD 利用经验累积分布函数（ECDF）刻画正常图像特征响应的统计分布，并将高维特征转换为紧凑的统计描述符。相比直接存储大规模特征库或进行高成本特征检索，该方式能够以更低的存储和计算开销表达正常样本的分布特征，从而提升模型在资源受限场景下的部署效率。

**（2）局部图块估计器与全局估计器相结合的双分支架构。**

FastAD 分别构建局部图块估计器与全局估计器，对正常样本进行局部和全局两个层面的统计建模。其中，局部图块估计器在图块级别提取特征分布描述符，用于捕捉划痕、污渍、破损等结构性异常；全局估计器则在整图特征图级别构建全局描述符，用于建模整体上下文信息，从而增强对组件缺失、位置错乱等逻辑异常的感知能力。

基于上述描述符，FastAD 分别拟合多元高斯分布模型，通过估计正常样本的均值与协方差矩阵，构建正常特征模式的概率表达。在推理阶段，测试样本经由相同流程提取局部与全局描述符，并通过计算其与正常分布之间的马氏距离量化异常程度。该框架在保持较高检测精度的同时，实现了毫秒级推理性能，具备良好的实时部署潜力。

---

## 🏆 核心评测亮点

### 1. 精度与效率对比
| 算法 | MVTec-LOCO (Image AUROC) | 全局平均 AUROC | 单类别训练时间 | 推理吞吐速度 (FPS) |
| :--- | :---: | :---: | :---: | :---: |
| **PatchCore** (CVPR '22) | 79.4% | 93.8% | ~3 分钟 | 12 FPS |
| **SimpleNet** (CVPR '23) | 84.5% | 94.6% | ~15 分钟 | 85 FPS |
| **EfficientAD-M** (WACV '24) | 90.7% | 96.5% | ~10 分钟 | 180 FPS |
| **FastAD (本文方法)** | **99.3%** | **98.7%** | **~16 秒** | **384 FPS** |

### 2. 训练样本量敏感度分析 (数据稀缺鲁棒性)
在极端少样本条件下，FastAD 展现出了出色的统计稳定性：


- **100% 样本**: 平均 AUROC 98.50%
- **25% 样本**: 平均 AUROC 95.27%（仅降 **3.23 pp**，部分类别仅用 12~72 张图片即可可靠检测）

---

## 🛠️ 环境准备与安装

### 系统要求
- 推荐 Python 3.12（最低 3.10），与 `.python-version` 一致
- PyTorch >= 2.0.0，请根据硬件和 CUDA 版本（使用 GPU 时）选择相应构建版本
- 兼容性取决于依赖是否提供相应构建版本，尚未验证所有操作系统与 Python 版本组合。

### 方式 1: 使用 `pip` 安装
```bash
# 克隆仓库
git clone https://github.com/jingfrank/FastAD.git
cd FastAD

# 以可编辑模式安装 Python 包及其依赖
pip install -e .
```

### 方式 2: 使用 `conda` 安装
```bash
conda env create -f environment.yml
conda activate fastad
```

wheel 仅包含 Python 包。示例配置（`configs/`）位于源码仓库中，需克隆仓库后使用。运行结果会写入本地 `outputs/` 目录，不随仓库发布。

### 极速自检验证 (冒烟测试)
无需下载任何数据集，在 CPU 上运行核心组件冒烟测试：
```bash
python tests/test_smoke.py
```

---

## 📂 数据集准备

FastAD 原生支持标准工业视觉评测基准：**MVTec-AD**、**MVTec-LOCO** 和 **VisA**。

请将数据集解压至 `./data` 目录（或通过环境变量 `DATASETS_ROOT` 自定义路径）：

```
data/
├── mvtec/
│   ├── bottle/
│   │   ├── train/
│   │   ├── test/
│   │   └── ground_truth/
│   └── ...
├── mvtec_loco/
│   ├── breakfast_box/
│   │   ├── train/
│   │   ├── validation/
│   │   ├── test/
│   │   └── ground_truth/
│   └── ...
└── visa/
    ├── candle/
    └── ...
```

> [!TIP]
> 如果您的数据集已存放在其他盘符，无需拷贝，直接指定环境变量即可：
> ```bash
> # Linux/macOS
> export DATASETS_ROOT=/path/to/your/datasets
> 
> # Windows PowerShell
> $env:DATASETS_ROOT="D:\path\to\your\datasets"
> ```

---

## 🚀 快速上手使用

项目仅保留 `fastad/fastad.py` 实现，由 `main_fastad` 入口调用。安装后可查看命令行选项：
```bash
python -m fastad.main_fastad --help
```

不支持旧实现生成的检查点，需使用当前实现重新训练模型。

### 1. 单类别训练与评测
在仓库根目录指定对应类别的 YAML 配置文件启动运行：
```bash
# 评测 MVTec-AD 'bottle' 类别
python fastad/main_fastad.py --config configs/MVTec_AD/bottle/base.yaml

# 评测 MVTec-LOCO 'splicing_connectors' 类别
python fastad/main_fastad.py --config configs/MVTec_LOCO/splicing_connectors/base.yaml
```

标准主干网络权重（`wide_resnet50_2`、`resnet50` 等）在首次运行时将通过 `timm` 自动在线下载。

### 2. 命令行超参数覆盖
支持通过 CLI 灵活修改参数：
```bash
python fastad/main_fastad.py \
    --config configs/MVTec_AD/cable/base.yaml \
    --backbone wide_resnet50_2 \
    --n-quantiles 5 \
    --patchsize 3 \
    --quantile-method channelwise
```

### 3. 分位数计算模式

在 YAML 中设置 `model.quantile_method`：
```yaml
model:
  quantile_method: channelwise  # 默认值；vectorized 一次处理所有通道
```

也可通过 `--quantile-method {channelwise,vectorized}` 覆盖：
```bash
python fastad/main_fastad.py \
    --config configs/MVTec_AD/bottle/base.yaml \
    --quantile-method vectorized
```

- **`channelwise`（默认）**：逐通道计算，降低分位数计算的峰值内存/显存占用，适合显存较小的 GPU。
- **`vectorized`**：在输入张量原设备上同时处理全部通道，可能更快，但峰值内存/显存占用更高；速度提升并无保证，取决于硬件与实际负载。

两种模式分别对每个通道或全部通道调用一次 `torch.quantile`，同时取得下、上分位数；均不迁移到 CPU，也不会在显存不足（OOM）时静默回退至另一模式。所选模式在 Patch 与 Global 两个分支的拟合、分数校准和推理阶段保持一致，只改变计算策略，不应改变分位数定义、阈值或 CDF 的统计含义。

Python API `compute_cdf_and_binary_features(...)` 和 `FastAD(...)` 均在参数列表末尾支持可选参数 `quantile_method='channelwise'`。模式随模型配置保存并在加载时恢复；当前实现中缺少该字段的旧检查点默认使用 `channelwise`，但这不代表支持旧实现生成的检查点。

---

## 📖 学术引用

如果您在科研或工程应用中使用了本项目的代码或方法，请引用我们的论文：

[正式出版页面 — Springer](https://doi.org/10.1007/978-3-032-38401-0_22)

会议为 ICANN 2026，论文章节上线日期为 2026-09-14；出版方正式引用年份为 **2027**。

```bibtex
@inproceedings{jing2027fastad,
  title={FastAD: A Deployable Framework for Real-Time Visual Anomaly Detection},
  author={Jing, Yuebin and Ding, Guangtai},
  booktitle={Artificial Neural Networks and Machine Learning – ICANN 2026},
  series={Lecture Notes in Computer Science},
  volume={17090},
  pages={269--281},
  year={2027},
  publisher={Springer},
  address={Cham},
  doi={10.1007/978-3-032-38401-0_22},
  url={https://doi.org/10.1007/978-3-032-38401-0_22}
}
```

---

## 📄 开源协议

本项目采用 **Apache License 2.0** 开源协议 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 🤝 联系与交流

如有使用问题、错误报告或改进建议，请提交 [GitHub Issue](https://github.com/jingfrank/FastAD/issues)。
