from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.evaluation import evaluate_predictions
from src.common.prediction_processing import process_prediction_maps
from src.common.training import ExperimentRunner
from src.common.validation import make_validation_split
from src.methods import get_method_class
from src.methods.base import BaseMethod


class Method(BaseMethod):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        method_config = config.get("method", {})
        self.ensemble_config = method_config
        self.base_method_config = copy.deepcopy(method_config.get("base_method", {}))
        if not self.base_method_config:
            raise ValueError("Ensemble method requires method.base_method")
        if self.base_method_config.get("name") == "ensemble":
            raise ValueError("Nested ensemble base_method is not supported")

        self.seed = int(config.get("seed", 42))
        self.n_splits = int(method_config.get("n_splits", 5))
        self.select_top_k = int(method_config.get("select_top_k", 3))
        self.metric = str(method_config.get("metric", "pixel_ap"))
        self.aggregation = str(method_config.get("aggregation", "mean"))
        self.good_fraction = float(method_config.get("good_fraction", 1.0))
        self.anomaly_fraction = float(method_config.get("anomaly_fraction", 1.0))
        self.save_checkpoints = bool(method_config.get("save_checkpoints", True))
        predict_chunk_size = method_config.get("predict_chunk_size")
        self.predict_chunk_size = (
            int(predict_chunk_size) if predict_chunk_size is not None else None
        )
        self.fold_records: list[dict[str, Any]] = []
        self.selected_records: list[dict[str, Any]] = []
        self.selected_runners: list[ExperimentRunner] = []

    def fit(self, train_data, val_data=None):
        if val_data is None:
            raise ValueError(
                "Ensemble fit requires train anomalies as val_data: runner.fit(train_good, train_anomalies)"
            )
        train_good = list(train_data)
        train_anomalies = list(val_data)
        if not train_anomalies:
            raise ValueError("Ensemble fold selection requires at least one labeled anomaly sample")
        if self.n_splits < 2:
            raise ValueError("method.n_splits must be at least 2 for k-fold ensemble selection")

        self.fold_records = []
        self.selected_records = []
        self.selected_runners = []

        for fold in range(self.n_splits):
            split = make_validation_split(
                train_good=train_good,
                train_anomalies=train_anomalies,
                n_splits=self.n_splits,
                fold=fold,
                seed=self.seed,
                anomaly_fraction=self.anomaly_fraction,
                good_fraction=self.good_fraction,
            )
            fold_config = self._base_config_for_fold(fold)
            method_cls = get_method_class(fold_config["method"]["name"])
            runner = ExperimentRunner(method_cls(fold_config), fold_config)
            runner.fit(split.train_good)
            predictions = runner.predict(split.val_samples)
            predictions = process_prediction_maps(split.val_samples, predictions, fold_config)
            metrics = evaluate_predictions(split.val_samples, predictions).as_dict()
            score = metrics.get(self.metric)
            if score is None:
                raise ValueError(f"Metric {self.metric} is undefined for ensemble fold {fold}")

            record = {
                "fold": fold,
                "score": float(score),
                "metric": self.metric,
                "n_train_good": len(split.train_good),
                "n_val_samples": len(split.val_samples),
                "selected": False,
                **metrics,
            }
            self.fold_records.append({"record": record, "runner": runner})
            print(f"Ensemble fold {fold}: {self.metric}={float(score):.4f}")

        ranked = sorted(
            self.fold_records,
            key=lambda item: (-float(item["record"]["score"]), int(item["record"]["fold"])),
        )
        n_select = max(1, min(self.select_top_k, len(ranked)))
        selected = ranked[:n_select]
        selected_folds = {int(item["record"]["fold"]) for item in selected}

        for item in self.fold_records:
            item["record"]["selected"] = int(item["record"]["fold"]) in selected_folds

        self.selected_runners = [item["runner"] for item in selected]
        self.selected_records = [copy.deepcopy(item["record"]) for item in selected]
        print(f"Selected ensemble folds: {sorted(selected_folds)}")

        output_dir = self.config.get("experiment", {}).get("output_dir")
        if output_dir:
            self.save(output_dir)
        return self

    def predict(self, test_data) -> dict[str, np.ndarray]:
        if not self.selected_runners:
            raise RuntimeError("Ensemble has not been fitted yet")
        if str(self.aggregation).lower() != "mean":
            raise ValueError(f"Unsupported ensemble aggregation: {self.aggregation}")

        samples = list(test_data)
        if not samples:
            return {}

        chunk_size = self.predict_chunk_size or len(samples)
        chunk_size = max(1, int(chunk_size))
        output: dict[str, np.ndarray] = {}
        n_members = float(len(self.selected_runners))

        for start in range(0, len(samples), chunk_size):
            chunk_samples = samples[start : start + chunk_size]
            expected_ids = {str(sample.image_id) for sample in chunk_samples}
            chunk_sum: dict[str, np.ndarray] | None = None

            for runner in self.selected_runners:
                member_predictions = runner.predict(chunk_samples)
                if set(member_predictions) != expected_ids:
                    missing = sorted(expected_ids - set(member_predictions))
                    extra = sorted(set(member_predictions) - expected_ids)
                    raise ValueError(
                        "Ensemble member prediction IDs mismatch; "
                        f"missing={missing[:3]}, extra={extra[:3]}"
                    )

                if chunk_sum is None:
                    chunk_sum = {
                        image_id: np.asarray(prediction, dtype=np.float32).copy()
                        for image_id, prediction in member_predictions.items()
                    }
                else:
                    for image_id, prediction in member_predictions.items():
                        chunk_sum[image_id] += np.asarray(prediction, dtype=np.float32)

            chunk_predictions = {
                image_id: (prediction_sum / n_members).astype(np.float32)
                for image_id, prediction_sum in (chunk_sum or {}).items()
            }
            chunk_predictions = process_prediction_maps(chunk_samples, chunk_predictions, self.config)
            output.update(chunk_predictions)

        return output

    def save(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        records = [copy.deepcopy(item["record"]) for item in self.fold_records]
        if records:
            pd.DataFrame(records).sort_values(["selected", "score"], ascending=[False, False]).to_csv(
                output_dir / "fold_metrics.csv",
                index=False,
            )

        metadata = {
            "method": "ensemble",
            "base_method": self.base_method_config,
            "n_splits": self.n_splits,
            "select_top_k": self.select_top_k,
            "metric": self.metric,
            "aggregation": self.aggregation,
            "predict_chunk_size": self.predict_chunk_size,
            "selected_folds": [int(record["fold"]) for record in self.selected_records],
            "selected_records": self.selected_records,
        }

        if self.save_checkpoints and self.selected_runners:
            checkpoint_paths = []
            for record, runner in zip(self.selected_records, self.selected_runners):
                fold = int(record["fold"])
                checkpoint_dir = output_dir / "selected_models" / f"fold_{fold:02d}"
                checkpoint_path = runner.method.save(checkpoint_dir)
                checkpoint_paths.append(str(checkpoint_path))
            metadata["checkpoint_paths"] = checkpoint_paths

        with (output_dir / "ensemble_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return output_dir

    def load(self, checkpoint_path: str | Path):
        path = Path(checkpoint_path)
        metadata_path = path / "ensemble_metadata.json" if path.is_dir() else path
        if not metadata_path.exists():
            raise FileNotFoundError(f"Ensemble metadata not found: {metadata_path}")

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.base_method_config = copy.deepcopy(metadata["base_method"])
        self.n_splits = int(metadata.get("n_splits", self.n_splits))
        self.select_top_k = int(metadata.get("select_top_k", self.select_top_k))
        self.metric = str(metadata.get("metric", self.metric))
        self.aggregation = str(metadata.get("aggregation", self.aggregation))
        self.predict_chunk_size = metadata.get("predict_chunk_size", self.predict_chunk_size)
        self.predict_chunk_size = (
            None if self.predict_chunk_size is None else int(self.predict_chunk_size)
        )
        self.selected_records = [copy.deepcopy(record) for record in metadata.get("selected_records", [])]

        fold_metrics_path = metadata_path.parent / "fold_metrics.csv"
        if fold_metrics_path.exists():
            records = pd.read_csv(fold_metrics_path).to_dict(orient="records")
            self.fold_records = [{"record": record, "runner": None} for record in records]
        else:
            self.fold_records = [{"record": record, "runner": None} for record in self.selected_records]

        checkpoint_paths = metadata.get("checkpoint_paths", [])
        if len(checkpoint_paths) != len(self.selected_records):
            checkpoint_paths = [
                str(metadata_path.parent / "selected_models" / f"fold_{int(record['fold']):02d}" / "patchcore_lite.pt")
                for record in self.selected_records
            ]

        self.selected_runners = []
        for record, checkpoint in zip(self.selected_records, checkpoint_paths):
            fold = int(record["fold"])
            checkpoint_file = _resolve_checkpoint_path(checkpoint, metadata_path.parent)
            fold_config = self._base_config_for_fold(fold)
            method_cls = get_method_class(fold_config["method"]["name"])
            method = method_cls(fold_config)
            method.load(checkpoint_file)
            self.selected_runners.append(ExperimentRunner(method, fold_config))

        if not self.selected_runners:
            raise ValueError(f"No selected ensemble checkpoints loaded from {metadata_path}")
        print(f"Loaded ensemble folds: {[int(record['fold']) for record in self.selected_records]}")
        return self

    def _base_config_for_fold(self, fold: int) -> dict[str, Any]:
        fold_config = copy.deepcopy(self.config)
        fold_config["method"] = copy.deepcopy(self.base_method_config)
        fold_config.setdefault("experiment", {})
        parent_name = self.config.get("experiment", {}).get("name", "ensemble")
        fold_config["experiment"]["name"] = f"{parent_name}_fold_{fold:02d}"
        output_dir = self.config.get("experiment", {}).get("output_dir")
        if output_dir:
            fold_config["experiment"]["output_dir"] = str(Path(output_dir) / "folds" / f"fold_{fold:02d}")
        return fold_config


def _aggregate_predictions(
    prediction_sets: list[dict[str, np.ndarray]],
    aggregation: str,
) -> dict[str, np.ndarray]:
    if not prediction_sets:
        return {}
    aggregation = str(aggregation).lower()
    image_ids = sorted(prediction_sets[0])
    expected_ids = set(image_ids)
    for i, predictions in enumerate(prediction_sets[1:], start=1):
        if set(predictions) != expected_ids:
            missing = sorted(expected_ids - set(predictions))
            extra = sorted(set(predictions) - expected_ids)
            raise ValueError(f"Prediction set {i} IDs mismatch; missing={missing[:3]}, extra={extra[:3]}")

    output: dict[str, np.ndarray] = {}
    for image_id in image_ids:
        stack = np.stack([np.asarray(predictions[image_id], dtype=np.float32) for predictions in prediction_sets])
        if aggregation == "mean":
            output[image_id] = np.mean(stack, axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unsupported ensemble aggregation: {aggregation}")
    return output


def _resolve_checkpoint_path(path: str | Path, metadata_dir: Path) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.exists():
        return checkpoint_path
    if checkpoint_path.is_absolute():
        raise FileNotFoundError(f"Ensemble checkpoint not found: {checkpoint_path}")

    candidate = metadata_dir / checkpoint_path.name
    if candidate.exists():
        return candidate

    fold_candidate = metadata_dir / "selected_models" / checkpoint_path.parent.name / checkpoint_path.name
    if fold_candidate.exists():
        return fold_candidate

    raise FileNotFoundError(f"Ensemble checkpoint not found: {path}")
