# Spacespresso - Advanced Deep Learning Project - 2025/2026

### Course Information
- **University:** Politecnico di Milano
- **Course:** Advanced Deep Learning (ADL)
- **Academic Year:** 2025/2026

### Team Members
- Jasmin Spinetto ([Jasmin Spinetto](https://github.com/JasminSpinetto))
- Camilo A. Martínez-Mejía ([Camilo A. Martínez-Mejía] (https://github.com/camiloa2m))

---

## Project Overview

> *Milan Space Center, 14 days before the rocket launch.*

This repository contains the development and implementation of the ADL anomaly detection challenge for A.Y. 2025/2026 — **Mission: Spacepresso**. The task is to build a pixel-level anomaly detection system capable of identifying defective items (mechanical gears, electronic components, coffee beans, and Bronte pistachios) in a mixed batch of manufacturing products. The system is evaluated using **Pixel-Level Average Precision (AP)** and must be fully reproducible on Google Colab.

---

## Installation and Environment Setup

To ensure reproducibility, we recommend using a virtual environment. Follow these steps to set up the workspace:

### Clone the Repository
```bash
git clone https://github.com/<your-org>/<your-repo>
cd <your-repo>
```

### Create a Conda virtual environment
```bash
conda env create -f environment.yml
conda activate adl_project
```

### Hardware-Specific PyTorch Installation

> Run **only one** of the following sections depending on your hardware.

#### A. Users with NVIDIA GPU
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
> CUDA 11.8 is compatible with the vast majority of NVIDIA GPUs. If installation fails, check your driver version with `nvidia-smi` and visit the [official PyTorch page](https://pytorch.org/get-started/locally/) for alternatives.

#### B. Users without a GPU (CPU only)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
> Running deep learning models on CPU is significantly slower. If you do not have a compatible NVIDIA GPU, we strongly recommend using Google Colab with a GPU runtime.

## Dataset Structure

The dataset is organized per object class. Each class follows this layout:

```
class_XX/
├── train/
│   ├── good/                    # Clean training images
│   └── anomaly_YY/              # One labeled anomalous example per anomaly type
├── ground_truth_train/
│   └── anomaly_YY/              # Pixel-level masks for the labeled anomaly examples
└── test/                        # Unlabeled leaderboard images (clean + anomalous)
anomaly_descriptions.csv         # Textual descriptions of each anomaly type
```

> Each source sample is exported as **five separate files** sharing the same `sample_id`.

---

## Evaluation Metric

Models are evaluated using **Pixel-Level Average Precision (AP)**:

$$AP = \sum_n (R_n - R_{n-1}) \cdot P_n$$

where $P_n$ and $R_n$ are the precision and recall at the $n$-th threshold. This measures how well the model ranks anomalous pixels above normal pixels across all thresholds.

---
## Rules & Constraints

| | Allowed |
|--|:-------:|
| Pre-trained models (standard, lecture-covered, or peer-reviewed) | ✅ |
| Synthetic data generated from the provided dataset | ✅ |
| Manual annotation of the test set | ❌ |
| External datasets from the same domain | ❌ |
| Using test images for training / self-supervised pre-training | ❌ |
