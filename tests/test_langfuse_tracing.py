"""Tests for opt-in, privacy-conscious Langfuse request tracing."""

from contextlib import contextmanager
from typing import Any

from langchain_core.documents import Document

from arabic_legal_qa.rag.orchestrator import RAG
from helpers.config import RAGConfig, Settings
from helpers.langfuse import create_langfuse_client


class FakeObservation:
    def __init__(self, details: dict[str, Any]) -> None:
        self.details = details
        self.updates: dict[str, Any] = {}

    def update(self, **kwargs: Any) -> None:
        self.updates.update(kwargs)


class FakeLangfuse:
    def __init__(self) -> None:
        self.observations: list[FakeObservation] = []

    @contextmanager
    def start_as_current_observation(self, **kwargs: Any):
        observation = FakeObservation(kwargs)
        self.observations.append(observation)
        yield observation


class FakeChatModel:
    def invoke(self, prompt: str) -> str:
        return "Supported by Article 147."


def test_langfuse_is_disabled_by_default() -> None:
    assert create_langfuse_client(Settings(_env_file=None)) is None


def test_query_records_spans_without_capturing_question_or_prompt(
    tmp_path, monkeypatch
) -> None:
    rag = RAG(RAGConfig(root=tmp_path))
    langfuse = FakeLangfuse()
    rag.langfuse_client = langfuse
    monkeypatch.setattr(
        rag,
        "retrieve",
        lambda *_args, **_kwargs: [
            Document(
                page_content="A contract may not be modified except by agreement.",
                metadata={
                    "article_number": 147,
                    "language": "en",
                    "citation": "Egyptian Civil Code, Article 147",
                },
            )
        ],
    )
    monkeypatch.setattr(rag, "_load_llm", lambda: FakeChatModel())

    result = rag.query("When can a contract be modified?")

    assert [item.details["name"] for item in langfuse.observations] == [
        "rag.query",
        "qdrant.hybrid_search",
        "answer-generation",
    ]
    root, retrieval, generation = langfuse.observations
    assert root.details["input"] is None
    assert root.updates["output"] == {
        "evidence_count": 1,
        "sources": ["Egyptian Civil Code, Article 147"],
    }
    assert retrieval.details["input"] == {"k": 5}
    assert retrieval.updates["output"]["articles"] == [
        "Egyptian Civil Code, Article 147"
    ]
    assert generation.details["input"] is None
    assert generation.updates["output"] == {"generated": True}
    assert result["answer"] == "Supported by Article 147."


def test_enabled_langfuse_without_keys_is_disabled_gracefully() -> None:
    settings = Settings(_env_file=None, langfuse_enabled=True)

    assert create_langfuse_client(settings) is None
