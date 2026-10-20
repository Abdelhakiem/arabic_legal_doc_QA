"""Tests for opt-in, privacy-conscious Langfuse request tracing."""

from contextlib import contextmanager
from typing import Any

import pytest
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


def test_required_langfuse_without_keys_fails_startup() -> None:
    settings = Settings(_env_file=None, langfuse_enabled=True, langfuse_required=True)

    with pytest.raises(RuntimeError, match="API keys are missing"):
        create_langfuse_client(settings)


def test_required_query_fails_closed_if_no_trace_client(tmp_path) -> None:
    rag = RAG(RAGConfig(root=tmp_path), langfuse_required=True)

    with pytest.raises(RuntimeError, match="client is unavailable"):
        rag.query("What does the code say?")


def test_query_can_set_a_trace_id_for_later_score_linking(tmp_path, monkeypatch):
    rag = RAG(RAGConfig(root=tmp_path))
    langfuse = FakeLangfuse()
    rag.langfuse_client = langfuse
    monkeypatch.setattr(
        rag,
        "retrieve",
        lambda *_args, **_kwargs: [
            Document(
                page_content="A contract may not be modified except by agreement.",
                metadata={"article_number": 147, "language": "en"},
            )
        ],
    )
    monkeypatch.setattr(rag, "_load_llm", lambda: FakeChatModel())
    trace_id = "ab" * 16

    result = rag.query(
        "When can a contract be modified?",
        include_trace=True,
        trace_id=trace_id,
        trace_metadata={"question_id": "monitor-1", "source": "ragas_monitor"},
    )

    root_observation = langfuse.observations[0]
    assert root_observation.details["trace_context"] == {"trace_id": trace_id}
    assert root_observation.details["metadata"]["question_id"] == "monitor-1"
    assert result["trace_id"] == trace_id
