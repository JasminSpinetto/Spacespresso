from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

from src.common.sample import ImageSample


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_image_filename(path: str | Path) -> tuple[str, str, str]:
    stem = Path(path).stem
    matches = list(re.finditer(r"(?i)(?:^|[_\-.])?(view\d+)(?=$|[_\-.])", stem))
    if not matches:
        return stem, stem, "view0"

    match = matches[-1]
    view_id = match.group(1).lower()
    sample_id = (stem[: match.start()] + stem[match.end() :]).strip("_-. ")
    sample_id = re.sub(r"[_\-.]+$", "", sample_id)
    if not sample_id:
        sample_id = stem
    return stem, sample_id, view_id


class SpacepressoDataModule:
    def __init__(
        self,
        data_root: str | Path | None = None,
        image_size: int | tuple[int, int] = 224,
        load_images: bool = True,
        root: str | Path | None = None,
    ):
        if data_root is None:
            data_root = root
        if data_root is None:
            raise ValueError("SpacepressoDataModule requires data_root or root")
        self.data_root = Path(data_root)
        self.image_size = self._normalize_image_size(image_size)
        self.load_images = bool(load_images)

    @staticmethod
    def _normalize_image_size(image_size: int | tuple[int, int] | list[int]) -> tuple[int, int]:
        if isinstance(image_size, int):
            return (image_size, image_size)
        if len(image_size) != 2:
            raise ValueError("image_size must be an int or a pair of ints")
        return (int(image_size[0]), int(image_size[1]))

    def list_classes(self) -> list[str]:
        if not self.data_root.exists():
            return []
        return sorted(p.name for p in self.data_root.glob("class_*") if p.is_dir())

    def load_train_good(self, class_name: str | None = None) -> list[ImageSample]:
        samples: list[ImageSample] = []
        for cls in self._iter_classes(class_name):
            image_dir = self.data_root / cls / "train" / "good"
            for path in self._iter_images(image_dir):
                samples.append(
                    self._make_sample(
                        path=path,
                        class_name=cls,
                        split="train_good",
                        label=0,
                        anomaly_type=None,
                        mask_path=None,
                    )
                )
        return samples

    def load_train_anomalies(self, class_name: str | None = None) -> list[ImageSample]:
        samples: list[ImageSample] = []
        for cls in self._iter_classes(class_name):
            train_dir = self.data_root / cls / "train"
            for anomaly_dir in sorted(train_dir.glob("anomaly_*")):
                if not anomaly_dir.is_dir():
                    continue
                anomaly_type = anomaly_dir.name
                mask_dir = self.data_root / cls / "ground_truth_train" / anomaly_type
                for path in self._iter_images(anomaly_dir):
                    samples.append(
                        self._make_sample(
                            path=path,
                            class_name=cls,
                            split="train_anomaly",
                            label=1,
                            anomaly_type=anomaly_type,
                            mask_path=self._find_mask_path(mask_dir, path),
                        )
                    )
        return samples

    def load_test(self, class_name: str | None = None) -> list[ImageSample]:
        samples: list[ImageSample] = []
        for cls in self._iter_classes(class_name):
            image_dir = self.data_root / cls / "test"
            for path in self._iter_images(image_dir):
                samples.append(
                    self._make_sample(
                        path=path,
                        class_name=cls,
                        split="test",
                        label=None,
                        anomaly_type=None,
                        mask_path=None,
                    )
                )
        return samples

    def load_anomaly_descriptions(self) -> pd.DataFrame | None:
        path = self.data_root / "anomaly_descriptions.csv"
        if not path.exists():
            return None
        return pd.read_csv(path)

    def _iter_classes(self, class_name: str | None) -> Iterable[str]:
        if class_name is None:
            yield from self.list_classes()
            return
        yield class_name

    @staticmethod
    def _iter_images(directory: Path) -> list[Path]:
        if not directory.exists():
            return []
        return sorted(
            p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _make_sample(
        self,
        path: Path,
        class_name: str,
        split: str,
        label: int | None,
        mask_path: Path | None,
        anomaly_type: str | None,
    ) -> ImageSample:
        image_id, sample_id, view_id = parse_image_filename(path)
        image = self.load_image(path) if self.load_images else None
        return ImageSample(
            image_id=image_id,
            sample_id=sample_id,
            class_name=class_name,
            view_id=view_id,
            split=split,
            image_path=path,
            image=image,
            label=label,
            mask_path=mask_path,
            anomaly_type=anomaly_type,
        )

    def load_image(self, path: str | Path) -> np.ndarray:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = image.resize(self.image_size[::-1], resample=Image.BILINEAR)
            arr = np.asarray(image, dtype=np.float32) / 255.0
        return arr

    @staticmethod
    def _find_mask_path(mask_dir: Path, image_path: Path) -> Path | None:
        if not mask_dir.exists():
            return None
        stem = image_path.stem
        for ext in IMAGE_EXTENSIONS:
            candidate = mask_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        matches = sorted(
            p for p in mask_dir.glob(f"{stem}*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        return matches[0] if matches else None
