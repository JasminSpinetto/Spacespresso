from __future__ import annotations

import numpy as np


def _display_image(sample):
    if sample.image is not None:
        return np.clip(sample.image, 0, 1)
    if sample.image_path is None:
        return None

    try:
        from PIL import Image

        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            return np.asarray(image, dtype=np.float32) / 255.0
    except Exception:
        return None


def show_samples(samples, n: int = 5):
    import matplotlib.pyplot as plt

    shown = list(samples)[:n]
    if not shown:
        print("No samples to show.")
        return None

    fig, axes = plt.subplots(1, len(shown), figsize=(4 * len(shown), 4))
    axes = np.atleast_1d(axes)
    for ax, sample in zip(axes, shown):
        image = _display_image(sample)
        if image is None:
            ax.text(0.5, 0.5, "image not loaded", ha="center", va="center")
        else:
            ax.imshow(image)
        ax.set_title(f"{sample.class_name}\n{sample.image_id}")
        ax.axis("off")
    fig.tight_layout()
    return fig


def show_predictions(samples, predictions: dict[str, np.ndarray], n: int = 5):
    import matplotlib.pyplot as plt

    shown = [sample for sample in list(samples) if sample.image_id in predictions][:n]
    if not shown:
        print("No matching predictions to show.")
        return None

    fig, axes = plt.subplots(len(shown), 2, figsize=(8, 4 * len(shown)))
    axes = np.asarray(axes).reshape(len(shown), 2)
    for row, sample in zip(axes, shown):
        image_ax, map_ax = row
        image = _display_image(sample)
        if image is None:
            image_ax.text(0.5, 0.5, "image not loaded", ha="center", va="center")
        else:
            image_ax.imshow(image)
        image_ax.set_title(sample.image_id)
        image_ax.axis("off")

        anomaly_map = predictions[sample.image_id]
        map_ax.imshow(anomaly_map, cmap="magma", vmin=0, vmax=1)
        map_ax.set_title("anomaly map")
        map_ax.axis("off")
    fig.tight_layout()
    return fig
