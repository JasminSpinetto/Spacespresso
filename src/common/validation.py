from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.common.sample import ImageSample


@dataclass(slots=True)
class ValidationSplit:
    train_good: list[ImageSample]
    val_samples: list[ImageSample]


def _group_by_class_and_sample(samples: Iterable[ImageSample]) -> dict[str, dict[str, list[ImageSample]]]:
    grouped: dict[str, dict[str, list[ImageSample]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        grouped[sample.class_name][sample.sample_id].append(sample)
    return grouped


def _fold_ids(sample_ids: list[str], n_splits: int, fold: int, seed: int) -> set[str]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if fold < 0 or fold >= n_splits:
        raise ValueError(f"fold must be in [0, {n_splits}), got {fold}")

    rng = np.random.default_rng(seed)
    shuffled = np.array(sorted(sample_ids), dtype=object)
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, n_splits)
    return set(str(x) for x in folds[fold])


def make_validation_split(
    train_good: list[ImageSample],
    train_anomalies: list[ImageSample],
    n_splits: int = 5,
    fold: int = 0,
    seed: int = 42,
    anomaly_fraction: float = 1.0,
    good_fraction: float = 1.0,
) -> ValidationSplit:
    """Create a sample-id based validation split.

    Clean training samples are split by ``sample_id`` within each class. Labeled
    training anomalies are never used for fitting and are included in validation.
    ``anomaly_fraction`` and ``good_fraction`` can reduce expensive validation
    during early tuning.
    """

    train_samples: list[ImageSample] = []
    val_good: list[ImageSample] = []

    for class_name, by_sample_id in _group_by_class_and_sample(train_good).items():
        val_ids = _fold_ids(list(by_sample_id), n_splits=n_splits, fold=fold, seed=seed)
        for sample_id, views in by_sample_id.items():
            if sample_id in val_ids:
                val_good.extend(views)
            else:
                train_samples.extend(views)

    val_good = _select_good_validation_views(val_good, good_fraction=good_fraction, seed=seed)
    val_anomalies = _select_anomalies(train_anomalies, anomaly_fraction=anomaly_fraction, seed=seed)
    return ValidationSplit(train_good=train_samples, val_samples=[*val_good, *val_anomalies])


def make_cross_validation_splits(
    train_good: list[ImageSample],
    train_anomalies: list[ImageSample],
    n_splits: int = 5,
    seed: int = 42,
    anomaly_fraction: float = 1.0,
    good_fraction: float = 1.0,
) -> list[ValidationSplit]:
    return [
        make_validation_split(
            train_good=train_good,
            train_anomalies=train_anomalies,
            n_splits=n_splits,
            fold=fold,
            seed=seed,
            anomaly_fraction=anomaly_fraction,
            good_fraction=good_fraction,
        )
        for fold in range(n_splits)
    ]


def _select_good_validation_views(
    val_good: list[ImageSample],
    good_fraction: float,
    seed: int,
) -> list[ImageSample]:
    if good_fraction >= 1.0:
        return list(val_good)
    if good_fraction <= 0.0:
        return []

    by_key: dict[tuple[str, str], list[ImageSample]] = defaultdict(list)
    for sample in val_good:
        by_key[(sample.class_name, sample.sample_id)].append(sample)

    keys = sorted(by_key)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_keep = max(1, int(round(len(keys) * good_fraction)))
    keep = set(keys[:n_keep])

    selected: list[ImageSample] = []
    for key in sorted(keep):
        selected.extend(by_key[key])
    return selected


def _select_anomalies(
    train_anomalies: list[ImageSample],
    anomaly_fraction: float,
    seed: int,
) -> list[ImageSample]:
    if anomaly_fraction >= 1.0:
        return list(train_anomalies)
    if anomaly_fraction <= 0.0:
        return []

    by_key: dict[tuple[str, str, str], list[ImageSample]] = defaultdict(list)
    for sample in train_anomalies:
        key = (sample.class_name, sample.anomaly_type or "anomaly", sample.sample_id)
        by_key[key].append(sample)

    keys = sorted(by_key)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_keep = max(1, int(round(len(keys) * anomaly_fraction)))
    keep = set(keys[:n_keep])

    selected: list[ImageSample] = []
    for key in sorted(keep):
        selected.extend(by_key[key])
    return selected
