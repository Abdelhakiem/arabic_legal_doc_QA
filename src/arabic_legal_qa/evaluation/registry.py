"""Register and promote a tested RAG configuration in MLflow."""
from __future__ import annotations

import logging
from dataclasses import asdict

from helpers.config import RegisteredRAGConfig
from helpers.mlflow_registry import CONFIG_TAG, config_to_json

logger = logging.getLogger(__name__)


def log_rag_config_model(config: RegisteredRAGConfig) -> str:
    """Log the config artifact and registerable pyfunc model in the active run."""
    import mlflow
    import pandas as pd
    from mlflow.models import infer_signature

    from arabic_legal_qa.evaluation._config_model import ConfigBundleModel

    payload = asdict(config)
    payload_json = config_to_json(config)
    input_example = pd.DataFrame({"request": ["return approved rag configuration"]})
    output_example = pd.DataFrame({"config_json": [payload_json]})
    mlflow.log_dict(payload, "rag_config.json")
    model_info = mlflow.pyfunc.log_model(
        name="rag_config",
        python_model=ConfigBundleModel(payload_json),
        input_example=input_example,
        signature=infer_signature(input_example, output_example),
    )
    return model_info.model_uri


def register_logged_rag_config(
    config: RegisteredRAGConfig,
    *,
    model_uri: str,
    model_name: str,
    tracking_uri: str,
    run_id: str,
    alias: str | None = None,
) -> str:
    """Register a candidate logged by an evaluation run and optionally promote it."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    model_version = mlflow.register_model(model_uri, model_name)
    client.set_model_version_tag(
        model_name,
        model_version.version,
        CONFIG_TAG,
        config_to_json(config),
    )
    client.set_model_version_tag(model_name, model_version.version, "source_run_id", run_id)
    if alias:
        client.set_registered_model_alias(model_name, alias, model_version.version)
    logger.info(
        "Registered RAG configuration",
        extra={
            "event": "mlflow.registry.config_registered",
            "model_name": model_name,
            "model_version": str(model_version.version),
            "model_alias": alias,
        },
    )
    return str(model_version.version)
