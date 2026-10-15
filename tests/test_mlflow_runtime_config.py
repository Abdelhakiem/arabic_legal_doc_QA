import pytest

from helpers.config import RegisteredRAGConfig, Settings
from arabic_legal_qa.rag.orchestrator import create_rag


def test_create_rag_applies_registry_config_to_live_pipeline(tmp_path, monkeypatch) -> None:
    settings = Settings(_env_file=None, mlflow_tracking_uri="http://mlflow:5000")
    registered = RegisteredRAGConfig(
        chunk_tokens=384,
        chunk_overlap=24,
        retrieval_k=7,
        prefetch_k=32,
        min_evidence=2,
    )
    monkeypatch.setattr("arabic_legal_qa.rag.orchestrator.get_settings", lambda *_args: settings)
    monkeypatch.setattr(
        "helpers.mlflow_registry.load_registered_rag_config",
        lambda _settings: (registered, "arabic-legal-rag-config@champion#3"),
    )

    rag = create_rag(tmp_path)

    assert rag.config.chunk_tokens == 384
    assert rag.config.chunk_overlap == 24
    assert rag.config.retrieval_k == 7
    assert rag.config.prefetch_k == 32
    assert rag.config.min_evidence == 2
    assert rag.registry_config_version == "arabic-legal-rag-config@champion#3"


def test_required_registry_config_fails_startup_if_alias_unavailable(tmp_path, monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        mlflow_tracking_uri="http://mlflow:5000",
        mlflow_config_required=True,
    )
    monkeypatch.setattr("arabic_legal_qa.rag.orchestrator.get_settings", lambda *_args: settings)
    monkeypatch.setattr(
        "helpers.mlflow_registry.load_registered_rag_config",
        lambda _settings: None,
    )

    with pytest.raises(RuntimeError, match="config is required"):
        create_rag(tmp_path)
