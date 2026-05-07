from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.q8rle import float_matrix_to_q8rle, q8rle_to_float_matrix


def _prepare_prediction_map(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Prediction must be a 2D numpy array, got shape {arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    min_value = float(arr.min()) if arr.size else 0.0
    max_value = float(arr.max()) if arr.size else 0.0
    if min_value < 0.0 or max_value > 1.0:
        dynamic_range = max_value - min_value
        if dynamic_range > 1e-12:
            arr = (arr - min_value) / dynamic_range
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


class SubmissionWriter:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)

    def write(self, predictions: dict[str, np.ndarray]) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for image_id in sorted(predictions):
            anomaly_map = _prepare_prediction_map(predictions[image_id])
            rows.append({"ID": image_id, "Label": float_matrix_to_q8rle(anomaly_map)})
        pd.DataFrame(rows, columns=["ID", "Label"]).to_csv(self.output_path, index=False)
        return self.output_path


def validate_submission(path: str | Path, expected_shape: tuple[int, int] | None = None) -> bool:
    path = Path(path)
    df = pd.read_csv(path)
    if list(df.columns) != ["ID", "Label"]:
        raise ValueError(f"Submission columns must be ['ID', 'Label'], got {list(df.columns)}")
    for i, label in enumerate(df["Label"]):
        if not isinstance(label, str) or not label.startswith("q8rle"):
            raise ValueError(f"Row {i} label does not start with q8rle")
        decoded = q8rle_to_float_matrix(label)
        if expected_shape is not None and decoded.shape != expected_shape:
            raise ValueError(
                f"Row {i} decoded shape {decoded.shape} does not match {expected_shape}"
            )
    return True

