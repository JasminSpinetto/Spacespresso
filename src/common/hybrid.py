from __future__ import annotations

from typing import Any

import numpy as np


def robust_normalize_map(
    anomaly_map: np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
    eps: float = 1e-8,
) -> np.ndarray:
    arr = np.asarray(anomaly_map, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    if arr.size == 0:
        return arr.copy()

    lo = float(np.percentile(arr, lower_percentile))
    hi = float(np.percentile(arr, upper_percentile))
    if hi - lo <= eps:
        lo = float(arr.min())
        hi = float(arr.max())
    if hi - lo <= eps:
        return np.zeros_like(arr, dtype=np.float32)

    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def patchcore_unet_refinement(
    patchcore_map: np.ndarray,
    unet_map: np.ndarray,
    *,
    strength: float = 0.25,
    unet_threshold: float = 0.5,
    unet_gamma: float = 1.0,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """Use U-Net as a spatial prior while preserving PatchCore as the base score."""

    patchcore = np.asarray(patchcore_map, dtype=np.float32)
    unet = robust_normalize_map(unet_map, upper_percentile=upper_percentile)
    if patchcore.shape != unet.shape:
        raise ValueError(f"PatchCore and U-Net maps must share shape, got {patchcore.shape} and {unet.shape}")

    threshold = float(unet_threshold)
    if threshold > 0.0:
        if threshold >= 1.0:
            prior = (unet >= threshold).astype(np.float32)
        else:
            prior = np.clip((unet - threshold) / (1.0 - threshold), 0.0, 1.0)
    else:
        prior = unet

    gamma = max(float(unet_gamma), 1e-6)
    prior = np.power(prior, gamma).astype(np.float32)
    refined = patchcore * (1.0 + float(strength) * prior)
    return np.clip(refined, 0.0, 1.0).astype(np.float32)


def build_patchcore_unet_refinement_predictions(
    patchcore_predictions: dict[str, np.ndarray],
    unet_predictions: dict[str, np.ndarray],
    params: dict[str, Any],
) -> dict[str, np.ndarray]:
    common_ids = sorted(set(patchcore_predictions) & set(unet_predictions))
    return {
        image_id: patchcore_unet_refinement(
            patchcore_predictions[image_id],
            unet_predictions[image_id],
            strength=float(params.get("strength", 0.25)),
            unet_threshold=float(params.get("unet_threshold", 0.5)),
            unet_gamma=float(params.get("unet_gamma", 1.0)),
            upper_percentile=float(params.get("upper_percentile", 99.5)),
        )
        for image_id in common_ids
    }
