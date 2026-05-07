from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class ImageSample:
    image_id: str
    sample_id: str
    class_name: str
    view_id: str
    split: str
    image_path: Path
    image: np.ndarray | None = None
    label: int | None = None
    mask_path: Path | None = None
    anomaly_type: str | None = None


@dataclass(slots=True)
class MultiViewSample:
    sample_id: str
    class_name: str
    views: list[ImageSample]

