# Spacespresso — Advanced Deep Learning Project 2025/2026

Pixel-level anomaly detection for **Mission: Spacepresso**, the Advanced Deep Learning (ADL) challenge at Politecnico di Milano, A.Y. 2025/2026.

📄 **[Final Report (PDF)](report/Avengers_Don't_Lose_Report.pdf)** - team *Avengers Don't Lose*.

## Course Information

| | |
| --- | --- |
| **University** | Politecnico di Milano |
| **Course** | Advanced Deep Learning (ADL) |
| **Academic Year** | 2025/2026 |
| **Task** | Pixel-level anomaly detection |
| **Metric** | Pixel-Level Average Precision (AP) |

## Team Members

- Jasmin Spinetto — [@JasminSpinetto](https://github.com/JasminSpinetto)
- Camilo A. Martínez-Mejía — [@camiloa2m](https://github.com/camiloa2m)
- Juan Martín Sánchez Bardellini — [@jmsb505](https://github.com/jmsb505)
- Reinaldo Toledo — [@Rey7910](https://github.com/Rey7910)

## Project Overview

> *Milan Space Center, 14 days before the rocket launch. The Italian Space Agency is ready to go, but the logistics department has made a catastrophic mess. We've just received our final shipment of supplies for the mission to Mars: mechanical gears, electronic components, and — most importantly — our survival stash of coffee beans and Bronte pistachios for the onboard gelato machine. The problem? The manufacturing facility sent us a batch riddled with defects. If even one bad coffee bean hits the grinder, the crew will revolt. Your task is to build an anomaly detection system to filter out every single faulty piece before we blast off. Identify the broken components, save the gelato, and ensure the espresso is impeccable!*

This repository contains our development and implementation of the ADL anomaly detection challenge for A.Y. 2025/2026. The goal is a **pixel-level anomaly detection system** that identifies defective items (mechanical gears, electronic components, coffee beans, and Bronte pistachios) in a mixed batch of manufacturing products. Only a small number of labeled anomalies are provided per class, so the setting is effectively **few-shot / cold-start anomaly detection**.

Models are evaluated on **Pixel-Level Average Precision (AP)** and must be **fully reproducible on Google Colab** (T4 GPU).

The codebase is organized so that notebooks stay lightweight while the reusable implementation lives in `src/`. Every method follows a common plug-and-play interface (see [Method Interface](#method-interface)), which makes it easy to swap approaches, tune them, and combine them into ensembles.

## Methods Implemented

We explored a broad range of anomaly-detection families. All methods live in `src/methods/` and share the same interface, with per-experiment settings in `configs/`.

| Family | Method | File |
| --- | --- | --- |
| Memory bank / feature distance | **PatchCore Lite** (coreset memory bank + kNN) | `patchcore_lite.py` |
| Memory bank / feature distance | Pretrained-feature scoring | `pretrained_features.py` |
| Memory bank / feature distance | EfficientAD with DINOv2 features | `efficientad_dinov2.py` |
| Reconstruction | Convolutional Autoencoder | `autoencoder.py` |
| Reconstruction + synthetic anomalies | DRAEM | `draem.py` |
| Normalizing flows | FastFlow | `fast_flow.py` |
| Knowledge distillation | Student–Teacher | `student_teacher.py` |
| Knowledge distillation | Reverse Distillation | `reverse_distillation.py` |
| Supervised (few-shot) | Patch-level MLP | `supervised_patch_mlp.py` |
| Supervised (few-shot) | DINOv2 patch MLP | `supervised_dinov2_mlp.py` |
| Synthetic supervision | U-Net on synthetic defects | `synthetic_unet.py` |
| Combination | Ensemble of anomaly maps | `ensemble.py` |
| Reference | Dummy baseline | `dummy.py` |

### Final submission

The final, reproducible configuration is **PatchCore Lite** with `wide_resnet50_2` backbone (see [`configs/final/final.yaml`](configs/final/final.yaml)). Full method comparison, ablations, and results are in the [report](report/Avengers_Don't_Lose_Report.pdf). Reproduce it end-to-end from [`notebooks/FINAL_reproduction.ipynb`](notebooks/FINAL_reproduction.ipynb).

## Installation and Environment Setup

We recommend a dedicated virtual environment (or Conda) for reproducibility.

### Clone the repository

```bash
git clone https://github.com/JasminSpinetto/Spacespresso.git
cd Spacespresso
```

### Install dependencies

```bash
pip install -r requirements.txt
```

A Conda environment file is also provided:

```bash
conda env create -f environment.yml
conda activate spacespresso
```

### Local NVIDIA GPU (optional)

If you have a local GPU, install a CUDA build of PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

CUDA 11.8 is compatible with many NVIDIA GPUs. If installation fails, check your driver with `nvidia-smi` and see the [official PyTorch install page](https://pytorch.org/get-started/locally/) for the right build. We suggest developing locally when a GPU is available, then validating on Colab.

## Reproducibility on Google Colab

All training and inference must be reproducible on **Google Colab** with a **T4 GPU runtime** (`Runtime → Change runtime type → T4 GPU`).

### Key constraints

- **~15 GB usable VRAM**: models must fit within this budget.
- **Ephemeral storage**: the filesystem resets between sessions. Mount Google Drive or re-download the dataset at the start of each session.
- **Session limits**: free-tier sessions can disconnect after inactivity or extended use; save checkpoints frequently.

### Staying within VRAM

- Use mixed precision (`torch.cuda.amp`) to roughly halve memory usage with minimal accuracy loss.
- Reduce batch size and accumulate gradients if you hit out-of-memory errors.
- Free unused tensors explicitly with `del tensor` and `torch.cuda.empty_cache()`.
- Prefer smaller backbones or quantized models for larger architectures.

## Dataset Structure

The dataset is organized per object class. Each class follows this layout:

```text
class_XX/
├── train/
│   ├── good/                # Clean training images
│   └── anomaly_YY/          # One labeled anomalous example per anomaly type
├── ground_truth_train/
│   └── anomaly_YY/          # Pixel-level masks for the labeled anomaly examples
└── test/                    # Unlabeled leaderboard images
anomaly_descriptions.csv     # Textual descriptions of each anomaly type
```

Each source sample is exported as five separate files (views) sharing the same `sample_id`; each view is treated as an independent `ImageSample`. On Colab, the configs expect the dataset at:

```text
/content/data/spacepresso
```

## Evaluation Metric

Models are evaluated with **Pixel-Level Average Precision (AP)**:

```text
AP = Σ_n (R_n − R_{n−1}) · P_n
```

where `P_n` and `R_n` are precision and recall at the `n`-th threshold. AP measures how well the model ranks anomalous pixels above normal pixels across all thresholds.

## Rules and Constraints

| Item | Allowed |
| --- | --- |
| Pre-trained models (standard, lecture-covered, or peer-reviewed) | Yes |
| Synthetic data generated from the provided dataset | Yes |
| Manual annotation of the test set | No |
| External datasets from the same domain | No |
| Using test images for training or self-supervised pre-training | No |

## Repository Structure

```text
src/
├── common/           # shared utilities: data, config, training runner, evaluation,
│                     # validation, tuning, augmentation, checkpointing, q8rle,
│                     # submission, ranking, visualization, seeding
└── methods/          # plug-and-play method implementations (see Methods Implemented)

configs/              # experiment- and member-specific YAML configs, grouped by method
                      # (configs/final/final.yaml is the final submission)
notebooks/            # example, per-member experiment, tuning, and reproduction notebooks
data/spacepresso/     # dataset mount point (contents git-ignored)
outputs/              # per-member local outputs (git-ignored, .gitkeep only)
submissions/          # per-member CSV submissions (git-ignored, .gitkeep only)
report/               # final report and figures
experiments.csv       # experiment tracker
requirements.txt      # pip dependencies
environment.yml       # Conda environment
```

Generated outputs, model weights, checkpoints, and CSV submissions are git-ignored; `.gitkeep` files preserve the folder structure.

## Running an Experiment

Every method is global and runnable from a notebook or a short script. Example with PatchCore Lite:

```python
from src.common.config import load_config
from src.common.data import SpacepressoDataModule
from src.common.submission import SubmissionWriter
from src.common.training import ExperimentRunner
from src.methods import get_method_class

config = load_config("configs/patchcore_lite/juan_baseline.yaml")
dm = SpacepressoDataModule(**config["data"])
train_good = dm.load_train_good()
test = dm.load_test()

Method = get_method_class("patchcore_lite")
runner = ExperimentRunner(Method(config), config)
runner.fit(train_good)
predictions = runner.predict(test)
SubmissionWriter(config["submission"]["output_path"]).write(predictions)
```

### PatchCore Lite memory knobs

PatchCore Lite (`src/methods/patchcore_lite.py`) uses a pretrained `timm` feature extractor (default `wide_resnet50_2`, feature maps from `out_indices: [2, 3]`), a bounded normal-patch pool, greedy coreset sampling, chunked nearest-neighbor distances, and upsampled/smoothed anomaly maps normalized to `[0, 1]`. If you hit memory limits, tune, in order:

- `batch_size`: lower this first if feature extraction crashes.
- `image_size`: lower for quick local checks.
- `candidate_pool_size`: max normal-patch candidates kept before coreset selection.
- `max_coreset_size`: max final memory-bank size for nearest-neighbor search.

## Validation and Tuning

Validate on the **labeled training anomalies**, never on the leaderboard test set. The shared validation utilities split clean `train/good` images by `sample_id` and add the labeled `train/anomaly_YY` samples (with masks) as validation positives.

## Method Interface

All methods are global and plug-and-play. Every module in `src/methods/` exposes:

```python
class Method(BaseMethod):
    def fit(self, train_data, val_data=None):
        ...

    def predict(self, test_data) -> dict[str, np.ndarray]:
        ...
```

`predict` returns one 2D float anomaly map in `[0, 1]` per image id:

```python
{
    "image_id": anomaly_map,  # 2D np.ndarray, values in [0, 1]
}
```
