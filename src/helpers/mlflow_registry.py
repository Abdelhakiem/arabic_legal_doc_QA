"""Read the approved RAG configuration from MLflow Model Registry."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

from helpers.config import RegisteredRAGConfig, Settings

logger = logging.getLogger(__name__)
CONFIG_TAG = "rag_config_json"


def load_registered_rag_config(
    settings: Settings,
) -> tuple[RegisteredRAGConfig, str] | None:
    """Resolve the configured model alias; return ``None`` when registry is off.

    The model version carries the validated config in a tag, so serving does
    not download or execute a model artifact just to read chunk/retrieval
    settings. A real MLflow model is still logged and registered by evaluation
    so every config has run/artifact provenance.
    """

    if not settings.mlflow_tracking_uri:
        return None
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    version = client.get_model_version_by_alias(
        settings.mlflow_config_model,
        settings.mlflow_config_alias,
    )
    raw = version.tags.get(CONFIG_TAG)
    if not raw:
        raise ValueError(
            f"Registry version {settings.mlflow_config_model}@{settings.mlflow_config_alias} "
            f"does not contain required tag {CONFIG_TAG!r}"
        )
    data: Any = json.loads(raw)
    config = RegisteredRAGConfig(**data)
    identity = f"{settings.mlflow_config_model}@{settings.mlflow_config_alias}#{version.version}"
    logger.info(
        "Loaded approved RAG configuration from MLflow Registry",
        extra={
            "event": "mlflow.registry.config_loaded",
            "model_name": settings.mlflow_config_model,
            "model_alias": settings.mlflow_config_alias,
            "model_version": str(version.version),
        },
    )
    return config, identity


def config_to_json(config: RegisteredRAGConfig) -> str:
    """Stable, JSON-safe representation used as the model-version tag."""

    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
