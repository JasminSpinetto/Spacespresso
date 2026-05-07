from __future__ import annotations

import re

import numpy as np


_HEADER = "q8rle"


def _as_unit_float_matrix(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"q8rle expects a 2D matrix, got shape {arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def float_matrix_to_q8rle(x: np.ndarray) -> str:
    """Encode a 2D float map in [0, 1] as quantized 8-bit RLE.

    The matrix is quantized to uint8 and flattened column-wise via
    ``flat = q.T.reshape(-1)`` as required by the competition format.
    The emitted format is:

        q8rle <height> <width> <value_0> <run_0> <value_1> <run_1> ...
    """

    arr = _as_unit_float_matrix(x)
    h, w = arr.shape
    q = np.rint(arr * 255.0).astype(np.uint8)
    flat = q.T.reshape(-1)

    if flat.size == 0:
        return f"{_HEADER} {h} {w}"

    runs: list[str] = []
    prev = int(flat[0])
    count = 1
    for value in flat[1:]:
        value_int = int(value)
        if value_int == prev:
            count += 1
        else:
            runs.extend([str(prev), str(count)])
            prev = value_int
            count = 1
    runs.extend([str(prev), str(count)])

    return " ".join([_HEADER, str(h), str(w), *runs])


def q8rle_to_float_matrix(s: str) -> np.ndarray:
    """Decode a q8rle string produced by ``float_matrix_to_q8rle``."""

    if not isinstance(s, str) or not s.startswith(_HEADER):
        raise ValueError("q8rle string must start with 'q8rle'")

    # Accept either whitespace or colon separators to be tolerant while keeping
    # the encoder simple and CSV-friendly.
    tokens = [t for t in re.split(r"[\s:]+", s.strip()) if t]
    if len(tokens) < 3 or tokens[0] != _HEADER:
        raise ValueError("Invalid q8rle header")

    h = int(tokens[1])
    w = int(tokens[2])
    payload = tokens[3:]
    if len(payload) % 2 != 0:
        raise ValueError("q8rle payload must contain value/run pairs")

    values: list[np.ndarray] = []
    total = 0
    for value_token, run_token in zip(payload[0::2], payload[1::2]):
        value = int(value_token)
        run = int(run_token)
        if value < 0 or value > 255:
            raise ValueError(f"q8rle value out of uint8 range: {value}")
        if run < 0:
            raise ValueError(f"q8rle run length must be non-negative: {run}")
        if run:
            values.append(np.full(run, value, dtype=np.uint8))
            total += run

    expected = h * w
    if total != expected:
        raise ValueError(f"Decoded q8rle length {total} does not match shape {h}x{w}")

    flat = np.concatenate(values) if values else np.array([], dtype=np.uint8)
    q = flat.reshape(w, h).T
    return q.astype(np.float32) / 255.0

