from __future__ import annotations

import numpy as np

from src.methods.base import BaseMethod


def _configured_shape(config: dict) -> tuple[int, int]:
    image_size = config.get("data", {}).get("image_size", 224)
    if isinstance(image_size, int):
        return (image_size, image_size)
    return (int(image_size[0]), int(image_size[1]))


class Method(BaseMethod):
    def fit(self, train_data, val_data=None):
        return self

    def predict(self, test_data) -> dict[str, np.ndarray]:
        default_shape = _configured_shape(self.config)
        predictions: dict[str, np.ndarray] = {}
        for sample in test_data:
            if sample.image is not None:
                shape = sample.image.shape[:2]
            else:
                shape = default_shape
            predictions[sample.image_id] = np.zeros(shape, dtype=np.float32)
        return predictions

