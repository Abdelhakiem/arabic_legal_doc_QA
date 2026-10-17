from __future__ import annotations

import pytest

from arabic_legal_qa.rag import llms
from helpers.config import Settings


@pytest.fixture(autouse=True)
def clear_settings_cache(monkeypatch: pytest.MonkeyPatch):
    cached_get_settings = llms.get_settings
    cached_get_settings.cache_clear()
    monkeypatch.setattr(llms, "get_settings", lambda: Settings(_env_file=None))
    yield
    cached_get_settings.cache_clear()


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LLM_PROVIDER",
        "GROQ_TOKEN",
        "GROQ_API_KEY",
        "QROQ_MODEL",
        "OLLAMA_MODEL",
        "OLLAMA_BASE_URL",
        "OLLAMA_PROBE_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)


def test_auto_prefers_available_ollama_model_over_groq(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("GROQ_TOKEN", "test-secret")
    monkeypatch.setenv("QROQ_MODEL", "test/groq-model")
    monkeypatch.setenv("OLLAMA_MODEL", "test/local-model")
    monkeypatch.setattr(llms, "_adapter_available", lambda _name: True)
    monkeypatch.setattr(llms, "_ollama_model_available", lambda _config: True)

    config = llms.resolve_llm_config()

    assert config.provider == "ollama"
    assert config.model_name == "test/local-model"
    assert config.base_url == "http://localhost:11434"
    assert "test-secret" not in repr(config)


def test_auto_falls_back_to_groq_when_ollama_model_is_unavailable(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("GROQ_TOKEN", "test-secret")
    monkeypatch.setenv("QROQ_MODEL", "test/groq-model")
    monkeypatch.setattr(llms, "_adapter_available", lambda _name: True)
    monkeypatch.setattr(llms, "_ollama_model_available", lambda _config: False)

    config = llms.resolve_llm_config()

    assert config.provider == "groq"
    assert config.model_name == "test/groq-model"
    assert config.groq_token == "test-secret"


def test_auto_uses_ollama_when_available_without_groq_credentials(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OLLAMA_MODEL", "test/local-model")
    monkeypatch.setattr(llms, "_adapter_available", lambda _name: True)
    monkeypatch.setattr(llms, "_ollama_model_available", lambda _config: True)

    config = llms.resolve_llm_config()

    assert config.provider == "ollama"
    assert config.model_name == "test/local-model"


def test_auto_fails_clearly_when_ollama_and_groq_are_unavailable(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr(
        llms, "_adapter_available", lambda name: name == "langchain_ollama"
    )
    monkeypatch.setattr(llms, "_ollama_model_available", lambda _config: False)

    with pytest.raises(RuntimeError, match="No LLM provider is available"):
        llms.resolve_llm_config()


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


def test_ollama_probe_checks_the_configured_model(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"models":[{"name":"qwen2.5:7b","model":"qwen2.5:7b"}]}'

    requested = {}

    def fake_urlopen(url, timeout):
        requested["url"] = url
        requested["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(llms, "urlopen", fake_urlopen)
    config = llms.LLMConfig(
        provider="auto",
        ollama_model="qwen2.5:7b",
        base_url="http://ollama:11434/",
        ollama_probe_timeout_seconds=0.75,
    )

    assert llms._ollama_model_available(config)
    assert requested == {
        "url": "http://ollama:11434/api/tags",
        "timeout": 0.75,
    }


def test_ollama_probe_returns_false_when_service_is_unreachable(monkeypatch):
    def fail_urlopen(_url, timeout):
        raise TimeoutError(f"probe timed out after {timeout}s")

    monkeypatch.setattr(llms, "urlopen", fail_urlopen)

    assert not llms._ollama_model_available(llms.LLMConfig())
