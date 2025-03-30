# Truncated EfficientNet for Tuberculosis Classification

## 📖 Description

This repository contains the full implementation and experimental analysis from my Master's thesis, which investigates **truncated versions of EfficientNet-B0** for binary classification of Tuberculosis (TB) in Chest X-Rays (CXRs).

The goal: reduce model size by strategically removing blocks from EfficientNet-B0—**without sacrificing diagnostic accuracy**. This lightweight approach is ideal for **real-world deployment in resource-constrained healthcare settings**.

---

### 🏗️ Methodology Overview

The study evaluates five models: the full EfficientNet-B0 and four truncated variants (removing 1 to 4 blocks). The core idea is to **systematically remove later-stage MBConv blocks**, replacing them with a lightweight `Conv1x1` layer to preserve feature flow.

#### 🔹 Full EfficientNet-B0 Architecture

The baseline model includes 10 blocks, starting with a `Conv3x3` stem, followed by MBConv1 and MBConv6 layers, and ending with a high-capacity `Conv1x1` and fully connected classifier.

| Block | Operator         | Resolution     | # Channels | # Layers | Pretrained |
|-------|------------------|----------------|------------|-----------|------------|
| 1     | Conv3x3          | 224 × 224      | 32         | 1         | ✅          |
| 2     | MBConv1, k3x3    | 112 × 112      | 16         | 1         | ✅          |
| 3     | MBConv6, k3x3    | 112 × 112      | 24         | 2         | ✅          |
| 4     | MBConv6, k5x5    | 56 × 56        | 40         | 2         | ✅          |
| 5     | MBConv6, k3x3    | 28 × 28        | 80         | 3         | ✅          |
| 6     | MBConv6, k5x5    | 14 × 14        | 112        | 3         | ✅          |
| 7     | MBConv6, k5x5    | 14 × 14        | 192        | 4         | ✅          |
| 8     | MBConv6, k3x3    | 7 × 7          | 320        | 1         | ✅          |
| 9     | Conv1x1          | 7 × 7          | 1280       | 1         | ✅          |
| 10    | Pooling & FC     | 7 × 7          | 1280       | 1         | ❌          |

---

#### 🔹 Proposed B0(-3) Truncated Architecture

The **B0(-3)** model removes blocks **6, 7, and 8**, reducing complexity while preserving core feature pathways. A new untrained `Conv1x1` block with 112 output channels bridges the remaining network to the classifier.

| Block | Operator         | Resolution     | # Channels | # Layers | Pretrained |
|-------|------------------|----------------|------------|-----------|------------|
| 1     | Conv3x3          | 224 × 224      | 32         | 1         | ✅          |
| 2     | MBConv1, k3x3    | 112 × 112      | 16         | 1         | ✅          |
| 3     | MBConv6, k3x3    | 112 × 112      | 24         | 2         | ✅          |
| 4     | MBConv6, k5x5    | 56 × 56        | 40         | 2         | ✅          |
| 5     | MBConv6, k3x3    | 28 × 28        | 80         | 3         | ✅          |
| 6     | Conv1x1          | 7 × 7          | 112        | 1         | ❌          |
| 7     | Pooling & FC     | 7 × 7          | 112        | 1         | ❌          |

> 🔍 The new Conv1x1 in block 6 ensures smooth dimensional transition while slashing the model to just **~308K parameters**—over **13× smaller** than the original B0.

---

### 🧪 Experimental Setup

- **Datasets**: 
  - Training: Balanced Kaggle dataset (3,500 TB + 3,500 Normal CXRs)
  - External Testing: Mendeley TB datasets from Pakistan, annotated with Urdu text (∼4,800 images)
- **Training Strategy**:
  - 40 epochs
  - Adam optimizer with LR scheduler
  - No data augmentation (but supported in codebase)
- **Evaluation**:
  - Internal test set (10% of Kaggle data)
  - External generalization on Mendeley data
  - **500 bootstrap iterations** for confidence intervals on external test set

---

### ✅ Core Insight

Despite its simplicity, **B0(-3)** generalizes extremely well, achieving:
- **97.38% external test accuracy**
- **98.96% sensitivity**
- **95.68% specificity**

These results meet **WHO guidelines for TB diagnostics** and outperform some existing full-size models—all while being lightweight enough for edge deployment.


Key contributions:
- 🔬 Systematic truncation of EfficientNet-B0 (from -1 to -4 blocks)
- 📊 Rigorous evaluation on internal (Kaggle) and external (Mendeley) datasets
- 📈 Bootstrap analysis with 95% confidence intervals
- ⚖️ Trade-off analysis between accuracy and model efficiency

---

## 🏆 Key Findings

- ✅ **100% accuracy** on internal Kaggle test set with all models
- 🌍 **97.4% external accuracy** with B0(-3), including:
  - Sensitivity: **98.96%**
  - Specificity: **95.68%**
- ⚡ B0(-3) uses only **308K parameters**, compared to **4.1M** in the full B0
- 📉 63× smaller than prior DenseNet-201–based approaches
- 🚀 Real-world potential for clinical use in low-resource settings

---

## 📁 Repository Structure

```
📁 cxr_thesis/
├── custom_lib/
│   ├── data_prep.py
│   ├── eval_tools.py
│   └── custom_models/
│       ├── truncated_b0.py, truncated_b0_leaky.py, etc.
│
├── run_model.py                  # Training script (full + truncated models)
├── run_model.ipynb              # Jupyter training variant
├── run_experiments.sh           # Batch experiment runner
├── run_bootstraps.ipynb         # Bootstrap CI evaluation
├── explore_model.ipynb          # Early exploration / architecture tests
├── replot_training_loss.ipynb   # Plot regeneration for report
├── results/                     # Saved metrics, checkpoints
├── external_bootstrap_results/  # External test bootstrap metrics
├── paper_figs/                  # Final figure exports for thesis
├── results_figs.ipynb           # Plot generation scripts
├── plots_presentation.pptx      # Presentation slides
├── requirements.txt             # Dependency list
└── thesis.pdf                   # Final paper
```

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch ≥ 1.10
- torchvision
- scikit-learn
- pandas, matplotlib, seaborn

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Prepare Data
Download and organize the datasets:
- [Kaggle TB Dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
- [Mendeley Pakistan TB Dataset](https://data.mendeley.com/datasets/jctsfj2sfn/1)

Place the images into a `data/` directory using the format expected by `data_prep.py`.

---

### 2. Train Model
```bash
python run_model.py --model b0_minus_3
```

### 3. Evaluate on External Data
```bash
python run_model.py --model b0_minus_3 --eval_only --external
```

### 4. Bootstrap Confidence Intervals
```bash
jupyter notebook run_bootstraps.ipynb
```

---

## 📊 Results Summary

| Model     | Params | Internal Acc | External Acc | Sensitivity | Specificity |
|-----------|--------|--------------|--------------|-------------|-------------|
| B0(-0)    | 4.1M   | 100%         | 97.26%       | 98.72%      | 95.68%      |
| B0(-3) 🔥 | 308K   | 100%         | 97.38%       | 98.96%      | 95.68%      |

> 🔥 B0(-3) is 13× smaller than B0(-0), with overlapping performance and better efficiency.



