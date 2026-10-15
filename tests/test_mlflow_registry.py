import json
import sys
import types

import pytest

from helpers.config import RegisteredRAGConfig, Settings
from helpers.mlflow_registry import (
    CONFIG_TAG,
    config_to_json,
    load_registered_rag_config,
)


def test_registered_config_round_trips_as_stable_json() -> None:
    config = RegisteredRAGConfig(
        chunk_tokens=384, chunk_overlap=32, retrieval_k=5, prefetch_k=24
    )

    assert json.loads(config_to_json(config)) == {
        "chunk_tokens": 384,
        "chunk_overlap": 32,
        "embedding_model": "intfloat/multilingual-e5-small",
        "embedding_revision": None,
        "retrieval_k": 5,
        "prefetch_k": 24,
        "min_evidence": 1,
    }


@pytest.mark.parametrize(
    "values",
    [
        {"chunk_tokens": 0},
        {"chunk_tokens": 32, "chunk_overlap": 32},
        {"retrieval_k": 25, "prefetch_k": 24},
    ],
)
def test_registered_config_rejects_invalid_values(values) -> None:
    with pytest.raises(ValueError):
        RegisteredRAGConfig(**values)


def test_registry_alias_loads_version_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = RegisteredRAGConfig(chunk_tokens=384, chunk_overlap=16)

    class FakeClient:
        def __init__(self, tracking_uri):
            assert tracking_uri == "http://mlflow:5000"

        def get_model_version_by_alias(self, name, alias):
            assert (name, alias) == ("arabic-legal-rag-config", "champion")
            return types.SimpleNamespace(
                version="7", tags={CONFIG_TAG: config_to_json(expected)}
            )

    mlflow_module = types.ModuleType("mlflow")
    mlflow_module.__path__ = []
    tracking_module = types.ModuleType("mlflow.tracking")
    tracking_module.MlflowClient = FakeClient
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_module)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking_module)

    settings = Settings(
        _env_file=None,
        mlflow_tracking_uri="http://mlflow:5000",
        mlflow_config_model="arabic-legal-rag-config",
        mlflow_config_alias="champion",
    )
    loaded, identity = load_registered_rag_config(settings)

    assert loaded == expected
    assert identity == "arabic-legal-rag-config@champion#7"


def test_registry_lookup_is_disabled_without_tracking_uri() -> None:
    assert load_registered_rag_config(Settings(_env_file=None)) is None
