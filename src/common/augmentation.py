from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL import ImageEnhance, ImageFilter, ImageOps


_BILINEAR = (
    PILImage.Resampling.BILINEAR if hasattr(PILImage, "Resampling") else PILImage.BILINEAR
)
_AFFINE = PILImage.Transform.AFFINE if hasattr(PILImage, "Transform") else PILImage.AFFINE


def deterministic_seed(seed: int, *parts: object) -> int:
    payload = "::".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def normalize_augmentation_config(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(config or {})
    copies_per_image = max(0, int(cfg.get("copies_per_image", 0)))
    include_original = bool(cfg.get("include_original", True))
    enabled = bool(cfg.get("enabled", False)) and copies_per_image > 0

    if not include_original and copies_per_image == 0:
        include_original = True

    return {
        "enabled": enabled,
        "copies_per_image": copies_per_image,
        "include_original": include_original,
        "horizontal_flip_probability": float(cfg.get("horizontal_flip_probability", 0.0)),
        "vertical_flip_probability": float(cfg.get("vertical_flip_probability", 0.0)),
        "max_rotation_degrees": float(cfg.get("max_rotation_degrees", 0.0)),
        "max_translate_fraction": float(cfg.get("max_translate_fraction", 0.0)),
        "scale_range": _float_pair(cfg.get("scale_range", (1.0, 1.0))),
        "brightness": max(0.0, float(cfg.get("brightness", 0.0))),
        "contrast": max(0.0, float(cfg.get("contrast", 0.0))),
        "gaussian_noise_std": max(0.0, float(cfg.get("gaussian_noise_std", 0.0))),
        "blur_probability": float(cfg.get("blur_probability", 0.0)),
        "blur_radius_range": _float_pair(cfg.get("blur_radius_range", (0.1, 0.8))),
        "fill_color": _rgb_tuple(cfg.get("fill_color", (0, 0, 0))),
    }


def augmented_sample_count(n_samples: int, config: dict[str, Any] | None) -> int:
    cfg = normalize_augmentation_config(config)
    if not cfg["enabled"]:
        return int(n_samples)
    variants_per_sample = int(cfg["copies_per_image"]) + int(bool(cfg["include_original"]))
    return int(n_samples) * max(1, variants_per_sample)


def apply_image_augmentation(
    image: np.ndarray,
    config: dict[str, Any] | None,
    seed: int,
) -> np.ndarray:
    cfg = normalize_augmentation_config(config)
    rng = np.random.default_rng(int(seed))
    pil_image = _to_pil_rgb(image)
    fill_color = cfg["fill_color"]

    if rng.random() < cfg["horizontal_flip_probability"]:
        pil_image = ImageOps.mirror(pil_image)
    if rng.random() < cfg["vertical_flip_probability"]:
        pil_image = ImageOps.flip(pil_image)

    angle = _uniform_symmetric(rng, cfg["max_rotation_degrees"])
    if abs(angle) > 1e-6:
        pil_image = pil_image.rotate(angle, resample=_BILINEAR, fillcolor=fill_color)

    scale_low, scale_high = cfg["scale_range"]
    scale = float(rng.uniform(scale_low, scale_high))
    if abs(scale - 1.0) > 1e-6:
        pil_image = _scale_about_center(pil_image, scale, fill_color)

    max_translate = cfg["max_translate_fraction"]
    if max_translate > 0.0:
        width, height = pil_image.size
        dx = int(round(rng.uniform(-max_translate, max_translate) * width))
        dy = int(round(rng.uniform(-max_translate, max_translate) * height))
        if dx != 0 or dy != 0:
            pil_image = pil_image.transform(
                pil_image.size,
                _AFFINE,
                (1, 0, -dx, 0, 1, -dy),
                resample=_BILINEAR,
                fillcolor=fill_color,
            )

    if cfg["brightness"] > 0.0:
        pil_image = ImageEnhance.Brightness(pil_image).enhance(
            _positive_factor(rng, cfg["brightness"])
        )
    if cfg["contrast"] > 0.0:
        pil_image = ImageEnhance.Contrast(pil_image).enhance(_positive_factor(rng, cfg["contrast"]))

    blur_probability = min(1.0, max(0.0, cfg["blur_probability"]))
    if blur_probability > 0.0 and rng.random() < blur_probability:
        low, high = cfg["blur_radius_range"]
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(low, high))))

    out = np.asarray(pil_image, dtype=np.float32) / 255.0
    noise_std = cfg["gaussian_noise_std"]
    if noise_std > 0.0:
        out = out + rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _to_pil_rgb(image: np.ndarray) -> PILImage.Image:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected an RGB image array, got shape {arr.shape}")
    arr = np.clip(arr, 0.0, 1.0)
    return PILImage.fromarray((arr * 255.0).round().astype(np.uint8), mode="RGB")


def _scale_about_center(
    image: PILImage.Image,
    scale: float,
    fill_color: tuple[int, int, int],
) -> PILImage.Image:
    width, height = image.size
    scale = max(1e-3, float(scale))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = image.resize((resized_width, resized_height), resample=_BILINEAR)

    if resized_width >= width and resized_height >= height:
        left = (resized_width - width) // 2
        top = (resized_height - height) // 2
        return resized.crop((left, top, left + width, top + height))

    canvas = PILImage.new("RGB", (width, height), fill_color)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _positive_factor(rng: np.random.Generator, amount: float) -> float:
    return max(0.0, 1.0 + _uniform_symmetric(rng, amount))


def _uniform_symmetric(rng: np.random.Generator, amount: float) -> float:
    amount = max(0.0, float(amount))
    if amount == 0.0:
        return 0.0
    return float(rng.uniform(-amount, amount))


def _float_pair(value: Any) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        first = second = float(value)
    else:
        first, second = value
        first = float(first)
        second = float(second)
    return (min(first, second), max(first, second))


def _rgb_tuple(value: Any) -> tuple[int, int, int]:
    if isinstance(value, (int, float)):
        x = int(value)
        return (x, x, x)
    red, green, blue = value
    return (
        int(np.clip(red, 0, 255)),
        int(np.clip(green, 0, 255)),
        int(np.clip(blue, 0, 255)),
    )
