"""Tests for the user-query-only retrieval path."""
from langchain_core.documents import Document

from helpers.config import RAGConfig
from arabic_legal_qa.rag.orchestrator import RAG


class FakeChatModel:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "The supported answer is Article 147."


def test_query_retrieves_once_with_original_question_and_generates_once(tmp_path, monkeypatch):
    question = "هل يجوز تعديل العقد من طرف واحد؟"
    rag = RAG(RAGConfig(root=tmp_path))
    llm = FakeChatModel()
    retrieved_queries: list[str] = []

    def retrieve(query: str, k=None, prefetch_k=None):
        retrieved_queries.append(query)
        return [Document(
            page_content="The contract may not be revoked or modified except by agreement.",
            metadata={
                "article_number": 147,
                "language": "en",
                "citation": "Egyptian Civil Code, Article 147",
                "chunk_id": "chunk-147",
            },
        )]

    monkeypatch.setattr(rag, "retrieve", retrieve)
    monkeypatch.setattr(rag, "_load_llm", lambda: llm)

    result = rag.query(question)

    assert retrieved_queries == [question]
    assert len(llm.prompts) == 1
    assert question in llm.prompts[0]
    assert "The contract may not be revoked" in llm.prompts[0]
    assert result["evidence_count"] == 1
    assert result["references"][0]["article_number"] == 147
    assert "queries" not in result


def test_query_abstains_after_single_original_question_retrieval(tmp_path, monkeypatch):
    rag = RAG(RAGConfig(root=tmp_path))
    retrieved_queries: list[str] = []

    def retrieve(query: str, k=None, prefetch_k=None):
        retrieved_queries.append(query)
        return []

    monkeypatch.setattr(rag, "retrieve", retrieve)
    monkeypatch.setattr(
        rag,
        "_load_llm",
        lambda: (_ for _ in ()).throw(AssertionError("LLM should not run on abstention")),
    )

    result = rag.query("ما أثر العقد؟")

    assert retrieved_queries == ["ما أثر العقد؟"]
    assert result["evidence_count"] == 0
    assert result["references"] == []
    assert "لا توجد أدلة كافية" in result["answer"]
