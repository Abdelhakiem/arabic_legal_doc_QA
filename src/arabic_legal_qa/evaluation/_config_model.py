"""Tiny pyfunc wrapper to make a config artifact registerable as an MLflow model."""
from mlflow.pyfunc import PythonModel


class ConfigBundleModel(PythonModel):
    """A minimal MLflow PythonModel that emits the immutable config JSON."""

    def __init__(self, config_json: str) -> None:
        self.config_json = config_json

    def predict(self, context, model_input, params=None):  # pragma: no cover - exercised by MLflow
        import pandas as pd

        return pd.DataFrame({"config_json": [self.config_json] * len(model_input)})
