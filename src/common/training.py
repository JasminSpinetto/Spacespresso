from __future__ import annotations

from typing import Any


class ExperimentRunner:
    def __init__(self, method: Any, config: dict[str, Any]):
        self.method = method
        self.config = config

    def fit(self, train_data, val_data=None):
        training_config = self.config.get("training", {})
        enabled = bool(training_config.get("enabled", True))
        if enabled:
            return self.method.fit(train_data, val_data)
        return self.method

    def predict(self, test_data):
        return self.method.predict(test_data)

