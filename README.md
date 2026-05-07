# Spacespresso - Advanced Deep Learning Project - 2025/2026

## Course Information

- **University:** Politecnico di Milano
- **Course:** Advanced Deep Learning (ADL)
- **Academic Year:** 2025/2026

## Team Members

- Jasmin Spinetto ([Jasmin Spinetto](https://github.com/JasminSpinetto))
- Camilo A. Martínez-Mejía ([Camilo A. Martínez-Mejía](https://github.com/camiloa2m))
- Juan Martin Sanchez Bardellini ([Juan Martin Sanchez Bardellini](https://github.com/jmsb505))
- Reinaldo Toledo ([Reinaldo Toledo](https://github.com/Rey7910))

## Project Overview

> Milan Space Center, 14 days before the rocket launch. Milan Space Center, 14 days before the rocket launch. The Italian Space Agency is ready to go, but the logistics department has made a catastrophic mess. We’ve just received our final shipment of supplies for the mission to Mars: mechanical gears, electronic components, and most importantly our survival stash of coffee beans and Bronte pistachios for the onboard gelato machine. The problem? The manufacturing facility sent us a batch riddled with defects. If even one bad coffee bean hits the grinder, the crew will revolt. The launch is in 14 days. Your task is to build an anomaly detection system to filter out every single faulty piece before we blast off. Identify the broken components, save the gelato, and ensure the espresso is impeccable!

This repository contains the development and implementation of the ADL anomaly detection challenge for A.Y. 2025/2026: **Mission: Spacepresso**. The task is to build a pixel-level anomaly detection system capable of identifying defective items, including mechanical gears, electronic components, coffee beans, and Bronte pistachios, in a mixed batch of manufacturing products. The system is evaluated using **Pixel-Level Average Precision (AP)** and must be fully reproducible on Google Colab.

The repository is organized so notebooks stay lightweight and reusable implementation lives in `src/`.

## Installation and Environment Setup

To ensure reproducibility, we recommend using a virtual environment.

### Clone the Repository

```bash
pip install -r requirements.txt
```

### Hardware Specifications

All training and inference must be fully reproducible on **Google Colab** using a **T4 GPU runtime** (`Runtime -> Change runtime type -> T4 GPU`).

| Component | Spec |
| --- | --- |
| GPU | NVIDIA Tesla T4 |
| Architecture | Turing (TU104) |
| CUDA Cores | 2,560 |
| Tensor Cores | 320, multi-precision: FP32 / FP16 / INT8 / INT4 |
| RT Cores | 40 |
| VRAM | 16 GB GDDR6, around 15 GB usable with 1 GB reserved for ECC |
| Memory Bandwidth | 320 GB/s |
| Memory Interface | 256-bit |
| GPU Clock (boost) | up to 1590 MHz |
| FP32 Performance | 8.1 TFLOPS |
| FP16 / Mixed Precision | around 65 TFLOPS |
| TDP | 70 W |
| System RAM | around 12-13 GB |
| CPU | Intel Xeon, 2 vCPUs |
| Disk | around 70 GB, ephemeral |

### Key Constraints

- Around 15 GB usable VRAM: models must fit within this budget.
- Ephemeral storage: the runtime filesystem resets between sessions. Mount Google Drive or re-download the dataset at the start of each session.
- Session time limits: free-tier sessions can disconnect after periods of inactivity or extended use. Save checkpoints frequently.
- GPU is not guaranteed: free-tier users may occasionally be assigned an older GPU such as a K80. Colab Pro offers more consistent T4 access.

For these reasons, we suggest developing locally if a GPU is available, then testing on Colab.

### Tips for Staying Within VRAM

- Use mixed precision with `torch.cuda.amp` to halve memory usage with minimal accuracy loss.
- Reduce batch size and accumulate gradients if you hit out-of-memory errors.
- Clear unused tensors explicitly with `del tensor` and `torch.cuda.empty_cache()`.
- Prefer smaller backbones or quantized models when working with larger architectures.

### Local NVIDIA GPU

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

CUDA 11.8 is compatible with many NVIDIA GPUs. If installation fails, check your driver version with `nvidia-smi` and visit the official PyTorch installation page for alternatives.

## Dataset Structure

The dataset is organized per object class. Each class follows this layout:

```text
class_XX/
├── train/
│   ├── good/                    # Clean training images
│   └── anomaly_YY/              # One labeled anomalous example per anomaly type
├── ground_truth_train/
│   └── anomaly_YY/              # Pixel-level masks for the labeled anomaly examples
└── test/                        # Unlabeled leaderboard images
anomaly_descriptions.csv         # Textual descriptions of each anomaly type
```

Each source sample is exported as five separate files sharing the same `sample_id`.

In Colab, the baseline configs expect:

```text
/content/data/spacepresso
```

Each image/view is treated as an independent `ImageSample` for the first baseline.

## Evaluation Metric

Models are evaluated using **Pixel-Level Average Precision (AP)**:

```text
AP = sum_n (R_n - R_{n-1}) * P_n
```

where `P_n` and `R_n` are the precision and recall at the `n`-th threshold. This measures how well the model ranks anomalous pixels above normal pixels across all thresholds.

## Rules and Constraints

| Item | Allowed |
| --- | --- |
| Pre-trained models: standard, lecture-covered, or peer-reviewed | Yes |
| Synthetic data generated from the provided dataset | Yes |
| Manual annotation of the test set | No |
| External datasets from the same domain | No |
| Using test images for training or self-supervised pre-training | No |

## Repository Structure

```text
notebooks/        # one experiment notebook per team member
src/common/       # shared data, q8rle, submission, config, visualization, runner utilities
src/methods/      # global plug-and-play method implementations
configs/          # experiment-specific and member-specific YAML configs
outputs/          # member-specific local outputs
submissions/      # member-specific CSV submissions
report/           # report material
experiments.csv   # experiment tracker
```

Methods are global. Configs are experiment-specific and may be member-specific. Outputs and submissions are member-specific.

## Running A Member Notebook

Each team member owns a notebook under `notebooks/` and a config under `configs/`.

Typical flow:

1. Load a YAML config.
2. Set the seed.
3. Initialize `SpacepressoDataModule`.
4. Load `train/good`, optional labeled anomalies for sanity checks, and `test`.
5. Instantiate the method with `get_method_class(config["method"]["name"])`.
6. Run `ExperimentRunner.fit(...)` and `ExperimentRunner.predict(...)`.
7. Write a CSV with `SubmissionWriter`.

From notebooks, run imports from the repo root:

```python
from src.common.config import load_config
from src.common.data import SpacepressoDataModule
from src.methods import get_method_class
```

## PatchCore Lite Baseline

The first baseline is `src/methods/patchcore_lite.py`, adapted from the uploaded MVTec AD reference notebook into a reusable method.

It uses:

- a pretrained timm feature extractor, default `wide_resnet50_2`
- feature maps from `out_indices: [2, 3]`
- clean training images only
- a bounded normal patch candidate pool
- greedy coreset sampling
- a configurable memory-bank cap via `max_coreset_size`
- chunked nearest-neighbor distances
- upsampled, smoothed anomaly maps normalized to `[0, 1]`

PatchCore memory knobs:

- `batch_size`: lower this first if feature extraction crashes.
- `candidate_pool_size`: maximum number of normal patch candidates kept before coreset selection.
- `max_coreset_size`: maximum final memory-bank size used for nearest-neighbor search.
- `image_size`: lower this for quick local checks.

Run it from `notebooks/patchcore_lite_baseline_example.ipynb`, `notebooks/juan_experiments.ipynb`, or with:

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

## Validation and Tuning

Use the labeled training anomalies for local validation, not the leaderboard test set. The shared validation utilities split clean `train/good` images by `sample_id` and add the labeled `train/anomaly_YY` samples with masks as validation positives.

PatchCore Lite tuning example:

```text
notebooks/patchcore_lite_optuna_example.ipynb
configs/patchcore_lite/juan_tuning.yaml
```

The Optuna objective maximizes `pixel_ap` by default and writes study outputs to:

```text
outputs/juan/patchcore_lite_tuning/optuna/
```

Important tuning artifacts:

- `outputs/validation_rankings.csv`: shared validation leaderboard across baseline runs, Optuna trials, and future experiments.
- `ranking.csv`: trials sorted by Pixel-Level AP, with `rank`, `trial_id`, score, and sampled hyperparameters.
- `best_trial.json`: best trial id, score, params, and path to the best config.
- `best_config.yaml`: full resolved config for reproducing the best trial.
- `trial_configs/*.yaml`: full resolved config for every completed trial.

Use `notebooks/view_validation_rankings.ipynb` to view the shared validation leaderboard without running training.

## Method Interface

All methods are global and plug-and-play from notebooks. Every method module in `src/methods/` exposes:

```python
class Method(BaseMethod):
    def fit(self, train_data, val_data=None):
        ...

    def predict(self, test_data) -> dict[str, np.ndarray]:
        ...
```

`predict` must return one 2D float anomaly map in `[0, 1]` per image id:

```python
{
    "image_id": anomaly_map,
}
```

## Submissions

Submission CSVs use:

```text
ID,Label
```

`Label` is q8rle-encoded from a 2D anomaly score map. Use `src/common/submission.py`; do not duplicate q8rle or CSV-writing logic in notebooks.

Member submission folders:

```text
submissions/juan/
submissions/jasmin/
submissions/camilo/
submissions/reinaldo/
submissions/final/
```

Member output folders:

```text
outputs/juan/
outputs/jasmin/
outputs/camilo/
outputs/reinaldo/
outputs/final/
```

Generated outputs, model weights, checkpoints, and CSV submissions are ignored by git. `.gitkeep` files keep the folder structure.

## Project Rules

- Put reusable utilities in `src/common/`.
- Put method implementations in `src/methods/`.
- Keep notebooks orchestration-only.
- Do not duplicate data loading, q8rle, submission creation, or PatchCore logic in notebooks.
- Track experiments in `experiments.csv`.
