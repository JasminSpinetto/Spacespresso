from __future__ import annotations

import numpy as np

from src.methods.base import BaseMethod


class Method(BaseMethod):
    def fit(self, train_data, val_data=None):
        return self

    def predict(self, test_data) -> dict[str, np.ndarray]:
        raise NotImplementedError("TODO: implement the autoencoder method.")

