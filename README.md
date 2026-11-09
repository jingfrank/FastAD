<div align="center">

# FastAD: A Deployable Framework for Real-Time Visual Anomaly Detection

[![Paper](https://img.shields.io/badge/ICANN%202026-Paper%20%23291-005EA6.svg)](https://doi.org/10.1007/978-3-032-38401-0_22)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)

**[English](README.md)** | **[中文说明](README_zh.md)**

*Official PyTorch implementation of the ICANN 2026 paper*  
**"FastAD: A Deployable Framework for Real-Time Visual Anomaly Detection"**

---

</div>

## 📌 Overview

Although unsupervised anomaly detection methods have made significant progress in detection accuracy in recent years, two key challenges remain for industrial deployment:

**(1) High computational resource consumption makes real-time deployment difficult.**

Existing high-accuracy methods often rely on large feature memory banks, complex nearest-neighbor retrieval, or multi-scale feature alignment strategies, resulting in long training times, high inference latency, and substantial GPU memory and storage requirements. Consequently, some methods developed in laboratory settings cannot be readily deployed on edge devices or resource-constrained production-line equipment, limiting their practical value in real industrial environments.

**(2) Diverse anomaly types make it difficult for a single modeling pathway to cover both structural and logical anomalies.**

Industrial anomalies include local structural defects, such as surface scratches, cracks, and stains, as well as logical anomalies, such as missing or misplaced components, that depend on global contextual relationships. The former require sensitivity to local textures and fine-grained geometric changes, while the latter require the ability to model overall structure and semantic relationships. Relying solely on local features or a single-scale representation can therefore lead to missed detections for certain types of anomalies.

To address these challenges, FastAD provides a lightweight unsupervised anomaly detection framework designed to combine high detection accuracy, low computational cost, and real-time inference. The framework is built around two main design choices:

**(1) Compact statistical descriptors based on empirical cumulative distribution functions.**

FastAD uses empirical cumulative distribution functions (ECDFs) to characterize the statistical distributions of feature responses in normal images and converts high-dimensional features into compact statistical descriptors. Compared with storing large feature banks or performing costly feature retrieval, this approach represents normal-sample distributions with lower storage and computational costs, improving deployment efficiency in resource-constrained environments.

**(2) A dual-branch architecture combining local patch and global estimators.**

FastAD constructs local patch and global estimators to statistically model normal samples at both local and global levels. The local patch estimator extracts feature-distribution descriptors at the patch level to capture structural anomalies such as scratches, stains, and damage. The global estimator builds global descriptors at the whole-image feature-map level to model overall contextual information, improving sensitivity to logical anomalies such as missing or misplaced components.

Using these descriptors, FastAD fits multivariate Gaussian models by estimating the means and covariance matrices of normal samples, forming probabilistic representations of normal feature patterns. During inference, local and global descriptors are extracted from test samples through the same pipeline, and their Mahalanobis distances from the learned normal distributions quantify the degree of anomaly. The framework achieves millisecond-level inference while maintaining high detection accuracy, demonstrating potential for real-time deployment.

---

## 🏆 Benchmark Highlights

### 1. State-of-the-Art Accuracy & Efficiency
| Method | MVTec-LOCO (Image AUROC) | Overall Mean AUROC | Train Time (per class) | Inference Speed (FPS) |
| :--- | :---: | :---: | :---: | :---: |
| **PatchCore** (CVPR '22) | 79.4% | 93.8% | ~3 min | 12 FPS |
| **SimpleNet** (CVPR '23) | 84.5% | 94.6% | ~15 min | 85 FPS |
| **EfficientAD-M** (WACV '24) | 90.7% | 96.5% | ~10 min | 180 FPS |
| **FastAD (Ours)** | **99.3%** | **98.7%** | **~16 sec** | **384 FPS** |

### 2. Sample Size Sensitivity (Data Scarcity Robustness)
FastAD maintains outstanding discriminative power even under heavy data constraints:


- **100% data**: 98.50% Mean AUROC
- **25% data**: 95.27% Mean AUROC (only **-3.23 pp** drop with as few as 12–72 training images)

---

## 🛠️ Installation

### Prerequisites
- Python 3.12 recommended (minimum 3.10), consistent with `.python-version`
- PyTorch >= 2.0.0, with a build appropriate for your hardware and CUDA version (if using a GPU)
- Compatibility depends on the available dependency builds; not all operating system and Python version combinations have been validated.

### Option 1: Using `pip`
```bash
# Clone the repository
git clone https://github.com/jingfrank/FastAD.git
cd FastAD

# Install the package and its dependencies in editable mode
pip install -e .
```

### Option 2: Using `conda`
```bash
conda env create -f environment.yml
conda activate fastad
```

The wheel contains only the Python package. Example configurations (`configs/`) are in the source repository; clone it to use them. Runtime results are written to a local `outputs/` directory and are not included in the repository.

### Quick Verification (Smoke Test)
Run the CPU smoke tests without downloading any datasets:
```bash
python tests/test_smoke.py
```

---

## 📂 Dataset Preparation

FastAD supports standard industrial benchmarks: **MVTec-AD**, **MVTec-LOCO**, and **VisA**.

Organize datasets under `./data` (or specify via `DATASETS_ROOT` environment variable):

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
> You can also link your existing dataset directory using an environment variable without copying files:
> ```bash
> # Linux/macOS
> export DATASETS_ROOT=/path/to/your/datasets
> 
> # Windows PowerShell
> $env:DATASETS_ROOT="D:\path\to\your\datasets"
> ```

---

## 🚀 Quick Start

The single implementation is `fastad/fastad.py`, used by the `main_fastad` entry point. After installation, inspect the CLI options with:
```bash
python -m fastad.main_fastad --help
```

Checkpoints from the legacy implementation are not supported; retrain models with the current implementation.

### 1. Single Category Training & Evaluation
From the repository root, run FastAD on any category using its corresponding YAML configuration:
```bash
# Evaluate on MVTec-AD 'bottle'
python fastad/main_fastad.py --config configs/MVTec_AD/bottle/base.yaml

# Evaluate on MVTec-LOCO 'splicing_connectors'
python fastad/main_fastad.py --config configs/MVTec_LOCO/splicing_connectors/base.yaml
```

Pretrained weights for standard backbones (`wide_resnet50_2`, `resnet50`, etc.) will be downloaded automatically via `timm` if not found locally.

### 2. CLI Parameter Overrides
You can easily override hyperparameters from the command line:
```bash
python fastad/main_fastad.py \
    --config configs/MVTec_AD/cable/base.yaml \
    --backbone wide_resnet50_2 \
    --n-quantiles 5 \
    --patchsize 3 \
    --quantile-method channelwise
```

### 3. Quantile Computation Mode

Set `model.quantile_method` in YAML:
```yaml
model:
  quantile_method: channelwise  # default; use vectorized for all channels at once
```

Or override it with `--quantile-method {channelwise,vectorized}`:
```bash
python fastad/main_fastad.py \
    --config configs/MVTec_AD/bottle/base.yaml \
    --quantile-method vectorized
```

- **`channelwise` (default)** processes one channel at a time to reduce peak quantile-computation memory; prefer it on GPUs with limited memory.
- **`vectorized`** processes all channels together on the input device. It may be faster, but uses more peak memory; a speedup is not guaranteed and depends on hardware and workload.

Both modes compute the lower and upper quantiles together in a single `torch.quantile` call per channel or for all channels, respectively. Neither moves computation to the CPU nor silently falls back to another mode on out-of-memory errors. The selected mode is used consistently by both the patch and global branches during fitting, score calibration, and inference; it changes the computation strategy, not the quantile definitions, thresholds, or CDF statistics.

The Python APIs `compute_cdf_and_binary_features(...)` and `FastAD(...)` also accept a trailing optional `quantile_method='channelwise'` argument. The mode is saved in the model configuration and restored when loading. Checkpoints from the current implementation that lack this field default to `channelwise`; this does not add support for checkpoints from the legacy implementation.

---

## 📖 Citation

If you find this repository or our paper useful for your research, please cite:

[Published chapter — Springer](https://doi.org/10.1007/978-3-032-38401-0_22)

The conference is ICANN 2026 and the chapter went online on 2026-09-14; the publisher's formal citation year is **2027**.

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

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

## 🤝 Contact

For questions, bug reports, and suggestions, please open a [GitHub issue](https://github.com/jingfrank/FastAD/issues).
