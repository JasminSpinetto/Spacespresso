from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


RANKING_COLUMNS = [
    "rank",
    "model_id",
    "experiment_name",
    "parent_experiment",
    "owner",
    "method",
    "source",
    "metric",
    "score",
    "pixel_ap",
    "image_ap",
    "pixel_auroc",
    "image_auroc",
    "n_images",
    "n_anomaly_pixels",
    "validation_n_splits",
    "validation_folds",
    "validation_good_fraction",
    "validation_anomaly_fraction",
    "config_path",
    "params_json",
    "notes",
    "created_at",
]


def validation_ranking_record(
    config: dict[str, Any],
    metrics: dict[str, Any],
    validation_config: dict[str, Any] | None = None,
    model_id: str | None = None,
    source: str = "experiment",
    config_path: str | Path | None = None,
    params: dict[str, Any] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    validation_config = validation_config or {}
    experiment = config.get("experiment", {})
    method = config.get("method", {})
    metric_name = str(validation_config.get("metric", "pixel_ap"))
    score = metrics.get(metric_name)
    if score is None:
        raise ValueError(f"Metric {metric_name} is missing from validation metrics")

    experiment_name = experiment.get("name", "experiment")
    record = {
        "model_id": model_id or experiment_name,
        "experiment_name": experiment_name,
        "parent_experiment": experiment.get("parent", ""),
        "owner": experiment.get("owner", ""),
        "method": method.get("name", ""),
        "source": source,
        "metric": metric_name,
        "score": float(score),
        "pixel_ap": _optional_float(metrics.get("pixel_ap")),
        "image_ap": _optional_float(metrics.get("image_ap")),
        "pixel_auroc": _optional_float(metrics.get("pixel_auroc")),
        "image_auroc": _optional_float(metrics.get("image_auroc")),
        "n_images": metrics.get("n_images"),
        "n_anomaly_pixels": metrics.get("n_anomaly_pixels"),
        "validation_n_splits": validation_config.get("n_splits", ""),
        "validation_folds": _folds_to_string(validation_config),
        "validation_good_fraction": validation_config.get("good_fraction", ""),
        "validation_anomaly_fraction": validation_config.get("anomaly_fraction", ""),
        "config_path": str(config_path) if config_path is not None else "",
        "params_json": json.dumps(params if params is not None else method, sort_keys=True),
        "notes": notes,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return record


class RankingWriter:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)

    def upsert(self, record: dict[str, Any]) -> pd.DataFrame:
        return self.upsert_many([record])

    def upsert_many(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        existing = self._read_existing()
        incoming = pd.DataFrame(records)
        df = pd.concat([existing, incoming], ignore_index=True)
        if not df.empty and "model_id" in df.columns:
            df = df.drop_duplicates(subset=["model_id"], keep="last")
        df = rank_dataframe(df)
        df.to_csv(self.output_path, index=False)
        return df

    def _read_existing(self) -> pd.DataFrame:
        if not self.output_path.exists():
            return pd.DataFrame(columns=RANKING_COLUMNS)
        return pd.read_csv(self.output_path)


def rank_dataframe(df: pd.DataFrame, metric_column: str = "score") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=RANKING_COLUMNS)
    ranked = df.copy()
    ranked[metric_column] = pd.to_numeric(ranked[metric_column], errors="coerce")
    ranked = ranked.sort_values(metric_column, ascending=False, na_position="last").reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    for column in RANKING_COLUMNS:
        if column not in ranked.columns:
            ranked[column] = ""
    return ranked[RANKING_COLUMNS]


def load_rankings(root: str | Path = ".", extra_paths: list[str | Path] | None = None) -> pd.DataFrame:
    root = Path(root)
    paths = [root / "outputs" / "validation_rankings.csv"]
    paths.extend(root.glob("outputs/**/validation_rankings.csv"))
    if extra_paths:
        paths.extend(Path(path) for path in extra_paths)

    frames = []
    seen: set[Path] = set()
    for path in paths:
        path = path.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        frames.append(pd.read_csv(path))

    if not frames:
        return pd.DataFrame(columns=RANKING_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    if "model_id" in df.columns:
        df = df.drop_duplicates(subset=["model_id"], keep="last")
    return rank_dataframe(df)


def _optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _folds_to_string(validation_config: dict[str, Any]) -> str:
    if "folds" in validation_config:
        return ",".join(str(x) for x in validation_config["folds"])
    if "fold" in validation_config:
        return str(validation_config["fold"])
    return ""

