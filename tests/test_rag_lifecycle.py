"""Unit tests for startup preparation without loading external model services."""

from types import SimpleNamespace

import pytest

from helpers.config import RAGConfig
from arabic_legal_qa.rag.orchestrator import RAG


class FakeIndexClient:
    def __init__(self, count: int) -> None:
        self.indexed = count
        self.count_calls = 0

    def count(self, _collection_name: str, exact: bool = False) -> SimpleNamespace:
        self.count_calls += 1
        assert exact is True
        return SimpleNamespace(count=self.indexed)


def test_initialize_prepares_clients_once_and_health_reads_index_count(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rag = RAG(RAGConfig(root=tmp_path))
    index = FakeIndexClient(count=15)
    calls = {"embeddings": 0, "sparse": 0, "llm": 0}

    def get_client():
        rag.qdrant_client = index
        return index

    monkeypatch.setattr(rag, "retrieval_client", get_client)
    monkeypatch.setattr(rag, "_load_embeddings", lambda: calls.__setitem__("embeddings", calls["embeddings"] + 1))
    monkeypatch.setattr(
        "arabic_legal_qa.rag.orchestrator.prepare_retrieval",
        lambda _config: calls.__setitem__("sparse", calls["sparse"] + 1),
    )
    monkeypatch.setattr(rag, "_load_llm", lambda: calls.__setitem__("llm", calls["llm"] + 1))

    assert rag.initialize() == 15
    assert rag.health() == 15
    assert calls == {"embeddings": 1, "sparse": 1, "llm": 1}
    assert index.count_calls == 2


def test_initialize_rejects_an_empty_index(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    rag = RAG(RAGConfig(root=tmp_path))
    index = FakeIndexClient(count=0)
    monkeypatch.setattr(rag, "retrieval_client", lambda: index)
    monkeypatch.setattr(rag, "_load_embeddings", lambda: None)
    monkeypatch.setattr("arabic_legal_qa.rag.orchestrator.prepare_retrieval", lambda _config: None)
    monkeypatch.setattr(rag, "_load_llm", lambda: None)

    with pytest.raises(RuntimeError, match="collection is empty"):
        rag.initialize()
