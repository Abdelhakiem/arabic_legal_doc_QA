from __future__ import annotations

import pytest

from arabic_legal_qa.rag import llms


@pytest.fixture(autouse=True)
def clear_settings_cache():
    llms.get_settings.cache_clear()
    yield
    llms.get_settings.cache_clear()


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LLM_PROVIDER",
        "GROQ_TOKEN",
        "GROQ_API_KEY",
        "QROQ_MODEL",
        "OLLAMA_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_auto_prefers_groq_when_token_and_adapter_are_available(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("GROQ_TOKEN", "test-secret")
    monkeypatch.setenv("QROQ_MODEL", "test/groq-model")
    monkeypatch.setattr(llms, "_adapter_available", lambda name: name == "langchain_groq")

    config = llms.resolve_llm_config()

    assert config.provider == "groq"
    assert config.model_name == "test/groq-model"
    assert config.groq_token == "test-secret"
    assert "test-secret" not in repr(config)


def test_auto_falls_back_to_ollama_without_groq_token(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OLLAMA_MODEL", "test/local-model")
    monkeypatch.setattr(llms, "_adapter_available", lambda name: name == "langchain_ollama")

    config = llms.resolve_llm_config()

    assert config.provider == "ollama"
    assert config.model_name == "test/local-model"


def test_auto_falls_back_to_ollama_when_groq_adapter_is_unavailable(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("GROQ_TOKEN", "test-secret")
    monkeypatch.setenv("OLLAMA_MODEL", "test/local-model")
    monkeypatch.setattr(llms, "_adapter_available", lambda name: name == "langchain_ollama")

    config = llms.resolve_llm_config()

    assert config.provider == "ollama"
    assert config.model_name == "test/local-model"


def test_auto_fails_clearly_when_no_provider_is_available(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr(llms, "_adapter_available", lambda _name: False)

    with pytest.raises(RuntimeError, match="No LLM provider is available"):
        llms.resolve_llm_config()


def test_explicit_groq_requires_a_token(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr(llms, "_adapter_available", lambda _name: True)

    with pytest.raises(RuntimeError, match="GROQ_TOKEN is not configured"):
        llms.resolve_llm_config(llms.LLMConfig(provider="groq"))
