"""Contract tests for the scheduled RAGAS monitor's scoring boundaries."""

from __future__ import annotations

import json

import pytest

from arabic_legal_qa.evaluation.monitor import (
    RAGAS_METRICS,
    _aggregate_scores,
    _attach_langfuse,
    _load_questions,
    _question_trace_id,
)
from arabic_legal_qa.rag.orchestrator import RAG
from helpers.config import RAGConfig


def _score_row(question_id: str, language: str, score: float) -> dict[str, object]:
    return {
        "question_id": question_id,
        "language": language,
        "llm_context_precision_with_reference": score,
        "faithfulness": score,
        "context_recall": score,
        "answer_correctness": score,
    }


def test_monitor_loads_at_least_fifty_questions_with_references(tmp_path) -> None:
    path = tmp_path / "questions.json"
    rows = [
        {
            "id": f"q-{index}",
            "language": "ar" if index % 2 else "en",
            "question": f"Question {index}",
            "reference_answer": f"Reference {index}",
        }
        for index in range(50)
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")

    assert len(_load_questions(path)) == 50


def test_monitor_rejects_less_than_fifty_questions(tmp_path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([{"id": "one"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="at least 50"):
        _load_questions(path)


def test_aggregate_requires_and_returns_all_four_ragas_metrics() -> None:
    rows = [_score_row("ar-1", "ar", 0.8), _score_row("en-1", "en", 0.6)]

    overall, by_language = _aggregate_scores(rows)

    assert set(overall) == set(RAGAS_METRICS)
    assert overall == {name: 0.7 for name in RAGAS_METRICS}
    assert by_language["faithfulness"] == {"ar": 0.8, "en": 0.6}


def test_aggregate_rejects_a_missing_metric_score() -> None:
    row = _score_row("ar-1", "ar", 0.8)
    row.pop("context_recall")

    with pytest.raises(RuntimeError, match="did not return context_recall"):
        _aggregate_scores([row])


def test_trace_id_is_32_hex_and_unique_per_run_and_question() -> None:
    first = _question_trace_id("a" * 32, "q-1")

    assert len(first) == 32
    int(first, 16)
    assert _question_trace_id("a" * 32, "q-1") == first
    assert _question_trace_id("a" * 32, "q-2") != first
    assert _question_trace_id("b" * 32, "q-1") != first


def test_monitor_rag_requires_langfuse_tracing(tmp_path) -> None:
    rag = RAG(RAGConfig(root=tmp_path))
    client = object()

    _attach_langfuse(rag, client, capture_content=False)

    assert rag.langfuse_client is client
    assert rag.langfuse_required is True
    assert rag.langfuse_capture_content is False
