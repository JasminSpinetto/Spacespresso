from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage


def normalize_prediction_processing_config(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(config or {})
    background = dict(cfg.get("background_suppression", {}))
    components = dict(cfg.get("connected_components", {}))
    return {
        "background_suppression": {
            "enabled": bool(background.get("enabled", False)),
            "threshold": float(background.get("threshold", 0.20)),
            "dilation": int(background.get("dilation", 16)),
            "threshold_per_class": dict(background.get("threshold_per_class", {})),
        },
        "connected_components": {
            "enabled": bool(components.get("enabled", False)),
            "min_component_area": int(components.get("min_component_area", 16)),
            "strong_component_threshold": float(components.get("strong_component_threshold", 0.15)),
            "mean_component_threshold": float(components.get("mean_component_threshold", 0.05)),
            "binary_threshold": float(components.get("binary_threshold", 0.01)),
            "gaussian_sigma": float(components.get("gaussian_sigma", 0.0)),
            "connectivity": int(components.get("connectivity", 2)),
        },
    }


def process_prediction_maps(
    samples,
    predictions: dict[str, np.ndarray],
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    processing = normalize_prediction_processing_config(config.get("prediction_processing", {}))
    output = {str(image_id): np.asarray(pred).copy() for image_id, pred in predictions.items()}
    if not output:
        return output

    background = processing["background_suppression"]
    if background["enabled"]:
        image_size = _image_size(config)
        id_to_sample = {str(sample.image_id): sample for sample in samples}
        for image_id, pred in list(output.items()):
            sample = id_to_sample.get(str(image_id))
            if sample is None:
                raise KeyError(f"Missing sample metadata for prediction image_id={image_id}")
            threshold = background["threshold_per_class"].get(
                sample.class_name,
                background["threshold"],
            )
            mask = foreground_mask_from_sample(
                sample,
                image_size=image_size,
                threshold=float(threshold),
                dilation=int(background["dilation"]),
            )
            output[image_id] = pred * mask.astype(np.float32)

    components = processing["connected_components"]
    if components["enabled"]:
        for image_id, pred in list(output.items()):
            output[image_id] = post_process_prediction_map(
                pred,
                min_component_area=components["min_component_area"],
                strong_component_threshold=components["strong_component_threshold"],
                mean_component_threshold=components["mean_component_threshold"],
                binary_threshold=components["binary_threshold"],
                gaussian_sigma=components["gaussian_sigma"],
                connectivity=components["connectivity"],
            )

    return output


def foreground_mask_from_sample(
    sample,
    image_size: tuple[int, int],
    threshold: float,
    dilation: int,
) -> np.ndarray:
    from scipy.ndimage import binary_dilation, binary_fill_holes

    if sample.image is not None:
        image = np.asarray(sample.image, dtype=np.float32)
        if image.shape[:2] != image_size:
            image = _load_image(sample.image_path, image_size)
    else:
        image = _load_image(sample.image_path, image_size)
    gray = image.mean(axis=2)
    fg = binary_fill_holes(gray > float(threshold))
    if dilation > 0:
        fg = binary_dilation(fg, iterations=int(dilation))
    return fg.astype(np.float32)


def post_process_prediction_map(
    arr: np.ndarray,
    min_component_area: int = 16,
    strong_component_threshold: float = 0.15,
    mean_component_threshold: float = 0.05,
    binary_threshold: float = 0.01,
    gaussian_sigma: float = 0.0,
    connectivity: int = 2,
) -> np.ndarray:
    from scipy.ndimage import (
        gaussian_filter,
        generate_binary_structure,
        label as ndi_label,
        maximum as ndi_max,
        mean as ndi_mean,
        sum as ndi_sum,
    )

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    out = arr.copy()
    smoothed = gaussian_filter(out, sigma=gaussian_sigma) if gaussian_sigma > 0.0 else out
    mask = smoothed > binary_threshold
    if not mask.any():
        return np.clip(out, 0.0, 1.0).astype(arr.dtype)

    struct = generate_binary_structure(2, 1 if connectivity == 1 else 2)
    labels, nlabels = ndi_label(mask, structure=struct)
    if nlabels == 0:
        return np.clip(out, 0.0, 1.0).astype(arr.dtype)

    index = np.arange(1, nlabels + 1)
    areas = ndi_sum(mask, labels, index)
    maxs = ndi_max(out, labels, index)
    means = ndi_mean(out, labels, index)
    should_remove = (maxs < strong_component_threshold) | (
        (areas < min_component_area) & (means < mean_component_threshold)
    )

    remove_mask_lookup = np.concatenate(([False], should_remove))
    out[remove_mask_lookup[labels]] = 0.0
    return np.clip(out, 0.0, 1.0).astype(arr.dtype)


def _image_size(config: dict[str, Any]) -> tuple[int, int]:
    image_size = config.get("data", {}).get("image_size", 224)
    if isinstance(image_size, int):
        return (image_size, image_size)
    if len(image_size) != 2:
        raise ValueError("image_size must be an int or a pair of ints")
    return (int(image_size[0]), int(image_size[1]))


def _load_image(path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    with PILImage.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size[1], image_size[0]), resample=PILImage.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0
