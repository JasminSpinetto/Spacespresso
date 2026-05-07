from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.common.sample import ImageSample


@dataclass(slots=True)
class EvaluationResult:
    pixel_ap: float
    image_ap: float
    pixel_auroc: float | None
    image_auroc: float | None
    n_images: int
    n_anomaly_pixels: int

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "pixel_ap": self.pixel_ap,
            "image_ap": self.image_ap,
            "pixel_auroc": self.pixel_auroc,
            "image_auroc": self.image_auroc,
            "n_images": self.n_images,
            "n_anomaly_pixels": self.n_anomaly_pixels,
        }


def evaluate_predictions(
    samples: list[ImageSample],
    predictions: dict[str, np.ndarray],
) -> EvaluationResult:
    pixel_targets: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    image_targets: list[int] = []
    image_scores: list[float] = []

    for sample in samples:
        if sample.image_id not in predictions:
            raise KeyError(f"Missing prediction for image_id={sample.image_id}")

        pred = _as_prediction_map(predictions[sample.image_id])
        mask = load_sample_mask(sample, pred.shape)
        target = (mask > 0).astype(np.uint8)

        pixel_targets.append(target.reshape(-1))
        pixel_scores.append(pred.reshape(-1))
        image_targets.append(int(target.max() > 0 or sample.label == 1))
        image_scores.append(float(pred.max()))

    y_pixel = np.concatenate(pixel_targets)
    s_pixel = np.concatenate(pixel_scores)
    y_image = np.asarray(image_targets, dtype=np.uint8)
    s_image = np.asarray(image_scores, dtype=np.float32)

    pixel_ap = _average_precision(y_pixel, s_pixel)
    image_ap = _average_precision(y_image, s_image)
    pixel_auroc = _roc_auc(y_pixel, s_pixel)
    image_auroc = _roc_auc(y_image, s_image)

    return EvaluationResult(
        pixel_ap=pixel_ap,
        image_ap=image_ap,
        pixel_auroc=pixel_auroc,
        image_auroc=image_auroc,
        n_images=len(samples),
        n_anomaly_pixels=int(y_pixel.sum()),
    )


def load_sample_mask(sample: ImageSample, shape: tuple[int, int]) -> np.ndarray:
    if sample.mask_path is None:
        return np.zeros(shape, dtype=np.uint8)

    mask_path = Path(sample.mask_path)
    if not mask_path.exists():
        return np.zeros(shape, dtype=np.uint8)

    with Image.open(mask_path) as image:
        image = image.convert("L")
        if image.size != (shape[1], shape[0]):
            image = image.resize((shape[1], shape[0]), resample=Image.NEAREST)
        mask = np.asarray(image, dtype=np.uint8)
    return (mask > 0).astype(np.uint8)


def _as_prediction_map(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Prediction must be a 2D map, got shape {arr.shape}")
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_positive = int(np.sum(y_true))
    if n_positive == 0:
        return 0.0

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    score_sorted = y_score[order]

    distinct = np.where(np.diff(score_sorted))[0]
    threshold_idxs = np.r_[distinct, y_true.size - 1]
    tps = np.cumsum(y_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / n_positive

    previous_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous_recall) * precision))


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_positive = int(np.sum(y_true))
    n_negative = int(y_true.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return None

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, y_score.size + 1)

    sorted_scores = y_score[order]
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = float(np.mean(np.arange(start + 1, end + 1)))
        start = end

    positive_rank_sum = float(np.sum(ranks[y_true == 1]))
    auc = (positive_rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    return float(auc)
