from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseMethod(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def fit(self, train_data, val_data=None):
        """Fit or initialize the method and return self."""

    @abstractmethod
    def predict(self, test_data) -> dict[str, np.ndarray]:
        """Return a mapping from image_id to a 2D anomaly map in [0, 1]."""

    def save(self, output_dir: str | Path):
        raise NotImplementedError(f"{self.__class__.__name__}.save is not implemented")

    def load(self, checkpoint_path: str | Path):
        raise NotImplementedError(f"{self.__class__.__name__}.load is not implemented")

