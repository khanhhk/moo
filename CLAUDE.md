# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repository implementing Multi-Objective Optimization (MOO) and Multi-Task Learning (MTL) algorithms, primarily focused on finding Pareto-optimal solutions when training neural networks on multiple tasks simultaneously. The core algorithms are MGDA (Multiple Gradient Descent Algorithm) and Pareto MTL with preference vectors.

## Setup

```bash
cd mtl-moo
source configure.sh    # Sets PYTHONPATH for multi_task module
```

Install full dependencies (not just requirements.txt):
```bash
pip install torch torchvision numpy scipy tensorboardX tqdm click pillow matplotlib gdown
# For research notebooks:
pip install autograd pymoo nltk plotly
```

## Running Experiments

**Train MGDA model (main framework):**
```bash
cd mtl-moo
python -m multi_task.train_multi_task --param_file params.json
tensorboard --logdir runs/
```

**Params JSON structure** (see `mtl-moo/sample.json` for CelebA example):
```json
{
  "dataset": "multiMNIST",      // multiMNIST | celeba | cityscapes
  "tasks": ["L", "R"],          // depends on dataset
  "algorithm": "mgda",          // mgda
  "normalization_type": "loss+", // none | l2 | loss | loss+
  "optimizer": "Adam",
  "lr": 1e-4,
  "batch_size": 256
}
```

**Pareto MTL (preference-based):**
```bash
# MultiMNIST
cd ParetoMTL/multiMNIST
python train_pref_multiMNIST_LS.py --npref 5

# Synthetic 2D toy problem (no data needed)
cd ParetoMTL/synthetic_example
python toy_example.py
```

**Research notebooks:** All notebooks in `anh/` and `minh/` run top-to-bottom with no interactive widgets.

**Toy synthetic problems** (no dataset needed):
```bash
cd anh
python VD1.py   # or VD2-VD5
```

## Dataset Download

```bash
python download_gdrive.py  # Downloads to /data/ (gitignored)
```

Configure dataset paths in `mtl-moo/configs.json`. Supported datasets:
- **MultiMNIST** — 36×36 images, 2 tasks (L/R digit), `.pickle` format
- **CelebA** — 64×64 images, 40 binary attribute tasks
- **Cityscapes** — 256×512, 3 tasks: segmentation/instance/depth

## Architecture

### Algorithm Core

The central algorithm solves: `min ||Σ cᵢ·gᵢ||₂²` subject to `Σ cᵢ = 1, cᵢ ≥ 0`

- **MinNormSolver** — QP solver: analytical 2D case, projected gradient for n>2.
  - `mtl-moo/multi_task/min_norm_solvers.py` (PyTorch)
  - `mtl-moo/multi_task/min_norm_solvers_numpy.py` (NumPy)
- **MGDA_UB** — Default mode: computes gradients only on encoder output (faster upper bound approximation)
- **Full MGDA** — Computes gradients over all encoder parameters (slower, more precise)

### Key Files

| File | Purpose |
|------|---------|
| `mtl-moo/multi_task/train_multi_task.py` | Main training loop, CLI entry point |
| `mtl-moo/multi_task/min_norm_solvers.py` | Core MGDA solver (PyTorch) |
| `mtl-moo/multi_task/models/` | LeNet (MultiMNIST), ResNet18 (CelebA), SegNet/PSPNet (Cityscapes) |
| `ParetoMTL/multiMNIST/` | Preference-vector based Pareto MTL experiments |
| `anh/process.ipynb` | Active research notebook with MTL variations |
| `minh/MTL/` | Drug review and MultiMNIST MTL Jupyter experiments |
| `minh/MOP/` | Synthetic multi-objective problem notebooks |

### Subdirectory Roles

- **`mtl-moo/`** — Production MGDA framework with TensorBoard logging
- **`ParetoMTL/`** — NeurIPS 2019 reference implementation (two-phase: init + optimization)
- **`anh/`** — Research variations: MinMax, non-monotone line search, toy problems
- **`minh/`** — Core experiments: drug review (TextCNN + NLP), MultiMNIST, result visualization

### Gradient Normalization (`normalization_type`)

- `none` — Raw task gradients
- `l2` — Normalize each by L2 norm
- `loss` — Normalize by loss value
- `loss+` — Normalize by `loss × L2` (recommended in sample.json)

## Notes

- Codebase uses old PyTorch `Variable` API (deprecated in 1.0+) — expect deprecation warnings, but code still runs
- Hardcoded paths are common; check `configs.json` and local notebook paths before running
- The `data/` directory is gitignored; datasets must be downloaded separately
