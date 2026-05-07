from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from src.common.evaluation import evaluate_predictions
from src.common.ranking import RankingWriter, rank_dataframe, validation_ranking_record
from src.common.training import ExperimentRunner
from src.common.validation import make_validation_split
from src.methods import get_method_class


TrialConfigFn = Callable[[Any, dict[str, Any]], dict[str, Any]]


class OptunaTuner:
    def __init__(
        self,
        base_config: dict[str, Any],
        train_good,
        train_anomalies,
        suggest_config: TrialConfigFn,
    ):
        self.base_config = copy.deepcopy(base_config)
        self.train_good = list(train_good)
        self.train_anomalies = list(train_anomalies)
        self.suggest_config = suggest_config
        self.tuning_config = self.base_config.get("tuning", {})
        self.trial_records: list[dict[str, Any]] = []

    def run(self):
        try:
            import optuna
        except Exception as exc:
            raise RuntimeError("Optuna is required for tuning. Install requirements.txt.") from exc

        direction = self.tuning_config.get("direction", "maximize")
        study_name = self.tuning_config.get("study_name")
        storage = self.tuning_config.get("storage")
        load_if_exists = bool(self.tuning_config.get("load_if_exists", True))

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists,
            direction=direction,
        )
        study.optimize(self.objective, n_trials=int(self.tuning_config.get("n_trials", 10)))
        self._write_outputs(study)
        return study

    def objective(self, trial) -> float:
        config = self.suggest_config(trial, copy.deepcopy(self.base_config))
        trial_id = self._trial_id(trial.number)
        config.setdefault("experiment", {})["name"] = trial_id
        config["experiment"]["parent"] = self.base_config.get("experiment", {}).get("name")
        trial.set_user_attr("trial_id", trial_id)
        trial.set_user_attr("resolved_method_config", copy.deepcopy(config.get("method", {})))

        folds = self.tuning_config.get("folds", [0])
        n_splits = int(self.tuning_config.get("n_splits", 5))
        seed = int(config.get("seed", 42))
        anomaly_fraction = float(self.tuning_config.get("anomaly_fraction", 1.0))
        good_fraction = float(self.tuning_config.get("good_fraction", 1.0))
        metric_name = str(self.tuning_config.get("metric", "pixel_ap"))

        scores: list[float] = []
        fold_metrics: list[dict[str, Any]] = []
        for fold in folds:
            split = make_validation_split(
                train_good=self.train_good,
                train_anomalies=self.train_anomalies,
                n_splits=n_splits,
                fold=int(fold),
                seed=seed,
                anomaly_fraction=anomaly_fraction,
                good_fraction=good_fraction,
            )

            method_cls = get_method_class(config["method"]["name"])
            runner = ExperimentRunner(method_cls(config), config)
            runner.fit(split.train_good)
            predictions = runner.predict(split.val_samples)
            result = evaluate_predictions(split.val_samples, predictions)
            metrics = result.as_dict()
            fold_metrics.append(metrics)
            value = metrics[metric_name]
            if value is None:
                raise ValueError(f"Metric {metric_name} is undefined for fold {fold}")
            scores.append(float(value))
            trial.set_user_attr(f"fold_{fold}_{metric_name}", float(value))
            for key, metric_value in metrics.items():
                if metric_value is not None:
                    trial.set_user_attr(f"fold_{fold}_{key}", metric_value)

        mean_score = float(sum(scores) / len(scores))
        mean_metrics = self._mean_metrics(fold_metrics)
        trial.set_user_attr("metric_name", metric_name)
        trial.set_user_attr("mean_score", mean_score)
        for key, value in mean_metrics.items():
            if value is not None:
                trial.set_user_attr(f"mean_{key}", value)
        self.trial_records.append(
            {
                "trial_id": trial_id,
                "trial_number": trial.number,
                "metric_name": metric_name,
                "score": mean_score,
                "metrics": mean_metrics,
                "params": copy.deepcopy(trial.params),
                "method": copy.deepcopy(config.get("method", {})),
                "config": self._strip_tuning_runtime(config),
            }
        )
        return mean_score

    def _write_outputs(self, study) -> None:
        output_dir = self.tuning_config.get("output_dir")
        if not output_dir:
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        trials_df = study.trials_dataframe()
        trials_df.to_csv(output_path / "trials.csv", index=False)
        ranking = self._ranking_dataframe(study)
        ranking.to_csv(output_path / "ranking.csv", index=False)
        self._write_trial_configs(output_path, study)
        validation_ranking = self._validation_ranking_dataframe(study, output_path)
        validation_ranking.to_csv(output_path / "validation_rankings.csv", index=False)
        self._write_global_ranking(validation_ranking)

        best_config = self._best_config(study)
        best = {
            "trial_id": self._trial_id(study.best_trial.number),
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial": study.best_trial.number,
            "metric_name": self.tuning_config.get("metric", "pixel_ap"),
            "best_config_path": str(output_path / "best_config.yaml"),
        }
        with (output_path / "best_trial.json").open("w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)
        with (output_path / "best_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(best_config, f, sort_keys=False)

    def _trial_id(self, trial_number: int) -> str:
        study_name = self.tuning_config.get("study_name") or self.base_config.get("experiment", {}).get(
            "name", "study"
        )
        return f"{study_name}_trial_{trial_number:04d}"

    def _ranking_dataframe(self, study) -> pd.DataFrame:
        rows = []
        metric_name = self.tuning_config.get("metric", "pixel_ap")
        for trial in study.trials:
            if trial.value is None:
                continue
            row = {
                "trial_id": trial.user_attrs.get("trial_id", self._trial_id(trial.number)),
                "trial_number": trial.number,
                "metric": metric_name,
                "score": float(trial.value),
                "state": str(trial.state),
            }
            row.update({f"param_{key}": value for key, value in trial.params.items()})
            rows.append(row)

        rows = sorted(rows, key=lambda x: x["score"], reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank

        columns = ["rank", "trial_id", "trial_number", "metric", "score", "state"]
        param_columns = sorted(k for row in rows for k in row if k.startswith("param_"))
        return pd.DataFrame(rows, columns=[*columns, *param_columns])

    def _validation_ranking_dataframe(self, study, output_path: Path) -> pd.DataFrame:
        records = []
        record_by_trial = {record["trial_number"]: record for record in self.trial_records}
        metric_name = self.tuning_config.get("metric", "pixel_ap")
        validation_config = {
            "metric": metric_name,
            "n_splits": self.tuning_config.get("n_splits", ""),
            "folds": self.tuning_config.get("folds", [0]),
            "good_fraction": self.tuning_config.get("good_fraction", ""),
            "anomaly_fraction": self.tuning_config.get("anomaly_fraction", ""),
        }

        for trial in study.trials:
            if trial.value is None:
                continue

            trial_id = trial.user_attrs.get("trial_id", self._trial_id(trial.number))
            record = record_by_trial.get(trial.number)
            config = record["config"] if record is not None else self._config_from_trial(trial)
            metrics = record["metrics"] if record is not None else self._metrics_from_trial(trial)
            metrics.setdefault(metric_name, float(trial.value))
            metrics.setdefault("n_images", trial.user_attrs.get("mean_n_images"))
            metrics.setdefault("n_anomaly_pixels", trial.user_attrs.get("mean_n_anomaly_pixels"))

            records.append(
                validation_ranking_record(
                    config=config,
                    metrics=metrics,
                    validation_config=validation_config,
                    model_id=trial_id,
                    source="optuna",
                    config_path=output_path / "trial_configs" / f"{trial_id}.yaml",
                    params=trial.params,
                    notes=f"Optuna trial {trial.number}",
                )
            )

        if not records:
            return rank_dataframe(pd.DataFrame())
        return rank_dataframe(pd.DataFrame(records))

    def _write_trial_configs(self, output_path: Path, study) -> None:
        config_dir = output_path / "trial_configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        records = {record["trial_number"]: record for record in self.trial_records}
        for trial in study.trials:
            if trial.value is None:
                continue
            record = records.get(trial.number)
            config = record["config"] if record is not None else self._config_from_trial(trial)
            trial_id = trial.user_attrs.get("trial_id", self._trial_id(trial.number))
            with (config_dir / f"{trial_id}.yaml").open("w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False)

    def _best_config(self, study) -> dict[str, Any]:
        records = {record["trial_number"]: record for record in self.trial_records}
        record = records.get(study.best_trial.number)
        if record is not None:
            return record["config"]
        return self._config_from_trial(study.best_trial)

    def _write_global_ranking(self, validation_ranking: pd.DataFrame) -> None:
        ranking_path = self.base_config.get("ranking", {}).get("output_path")
        if not ranking_path:
            return
        if validation_ranking.empty:
            return
        records = validation_ranking.drop(columns=["rank"], errors="ignore").to_dict(orient="records")
        RankingWriter(ranking_path).upsert_many(records)

    @staticmethod
    def _mean_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        if not fold_metrics:
            return {}
        keys = sorted({key for metrics in fold_metrics for key in metrics})
        mean_metrics: dict[str, Any] = {}
        for key in keys:
            values = [metrics.get(key) for metrics in fold_metrics if metrics.get(key) is not None]
            if not values:
                mean_metrics[key] = None
            elif key in {"n_images", "n_anomaly_pixels"}:
                mean_metrics[key] = int(round(sum(float(x) for x in values) / len(values)))
            else:
                mean_metrics[key] = float(sum(float(x) for x in values) / len(values))
        return mean_metrics

    @staticmethod
    def _metrics_from_trial(trial) -> dict[str, Any]:
        metrics = {}
        for key, value in trial.user_attrs.items():
            if key.startswith("mean_"):
                metrics[key.removeprefix("mean_")] = value
        return metrics

    def _config_from_trial(self, trial) -> dict[str, Any]:
        config = copy.deepcopy(self.base_config)
        config.setdefault("experiment", {})["name"] = trial.user_attrs.get(
            "trial_id", self._trial_id(trial.number)
        )
        config["experiment"]["parent"] = self.base_config.get("experiment", {}).get("name")
        method_config = trial.user_attrs.get("resolved_method_config")
        if method_config is not None:
            config["method"] = method_config
        return self._strip_tuning_runtime(config)

    @staticmethod
    def _strip_tuning_runtime(config: dict[str, Any]) -> dict[str, Any]:
        config = copy.deepcopy(config)
        tuning = config.get("tuning")
        if isinstance(tuning, dict):
            tuning.pop("storage", None)
        return config


def suggest_patchcore_lite_config(trial, config: dict[str, Any]) -> dict[str, Any]:
    method = config.setdefault("method", {})
    search = config.get("tuning", {}).get("search_space", {})

    method["coreset_fraction"] = _suggest_float(
        trial, "coreset_fraction", search, default_low=0.002, default_high=0.02, log=True
    )
    method["max_coreset_size"] = _suggest_int(
        trial, "max_coreset_size", search, default_low=500, default_high=2500, step=250
    )
    method["candidate_pool_size"] = _suggest_int(
        trial, "candidate_pool_size", search, default_low=2000, default_high=10000, step=1000
    )
    method["projection_dim"] = _suggest_categorical(
        trial, "projection_dim", search, default_choices=[128, 256, 512]
    )
    method["sigma"] = _suggest_float(
        trial, "sigma", search, default_low=1.0, default_high=6.0, log=False
    )
    out_indices = _suggest_categorical(
        trial, "out_indices", search, default_choices=["2,3", "1,2,3"]
    )
    method["out_indices"] = [int(x) for x in str(out_indices).split(",")]
    return config


def _suggest_float(
    trial,
    name: str,
    search: dict[str, Any],
    default_low: float,
    default_high: float,
    log: bool,
) -> float:
    spec = search.get(name, {})
    return float(
        trial.suggest_float(
            name,
            float(spec.get("low", default_low)),
            float(spec.get("high", default_high)),
            log=bool(spec.get("log", log)),
        )
    )


def _suggest_int(
    trial,
    name: str,
    search: dict[str, Any],
    default_low: int,
    default_high: int,
    step: int,
) -> int:
    spec = search.get(name, {})
    return int(
        trial.suggest_int(
            name,
            int(spec.get("low", default_low)),
            int(spec.get("high", default_high)),
            step=int(spec.get("step", step)),
        )
    )


def _suggest_categorical(
    trial,
    name: str,
    search: dict[str, Any],
    default_choices: list[Any],
):
    spec = search.get(name, {})
    choices = spec.get("choices", default_choices)
    return trial.suggest_categorical(name, choices)
