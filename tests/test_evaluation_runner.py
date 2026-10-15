import pytest

from arabic_legal_qa.evaluation.runner import (
    DEFAULT_CANDIDATES,
    DEFAULT_MLFLOW_TRACKING_URI,
    _article_retrieval_metrics,
    _cache_read,
    _cache_write,
    _quick_questions,
    evaluate_candidates,
)
from helpers.config import Settings


def test_default_experiment_sweep_has_six_chunk_overlap_and_k_candidates() -> None:
    assert len(DEFAULT_CANDIDATES) == 6
    assert len(set(DEFAULT_CANDIDATES)) == 6
    assert {tokens for tokens, _overlap, _k in DEFAULT_CANDIDATES} == {256, 384, 480}
    assert {overlap for _tokens, overlap, _k in DEFAULT_CANDIDATES} == {0, 32}
    assert {k for _tokens, _overlap, k in DEFAULT_CANDIDATES} == {5, 8}


def test_article_retrieval_metrics_score_expected_article_recall_and_hit() -> None:
    rows = [
        {"expected_articles": [147], "retrieved_article_numbers": [147, 148]},
        {"expected_articles": [54, 55], "retrieved_article_numbers": [55]},
    ]

    assert _article_retrieval_metrics(rows) == {
        "article_hit_rate_at_k": 1.0,
        "article_recall_at_k": 0.75,
    }


def test_quick_mode_selects_balanced_arabic_and_english_examples() -> None:
    rows = [
        {"id": "ar1", "language": "ar"},
        {"id": "en1", "language": "en"},
        {"id": "ar2", "language": "ar"},
        {"id": "en2", "language": "en"},
        {"id": "ar3", "language": "ar"},
        {"id": "en3", "language": "en"},
    ]

    selected = _quick_questions(rows)

    assert len(selected) == 4
    assert [row["id"] for row in selected] == ["ar1", "ar2", "en1", "en2"]


def test_evaluation_stage_cache_round_trips_json(tmp_path) -> None:
    cache_file = tmp_path / "answers" / "answer.json"
    expected = {"answer": "الإجابة", "scores": [0.5, 1.0]}

    assert _cache_read(cache_file) is None
    _cache_write(cache_file, expected)
    assert _cache_read(cache_file) == expected


def test_heldout_split_cannot_be_used_for_a_configuration_sweep(
    tmp_path, monkeypatch
) -> None:
    settings = Settings(_env_file=None, mlflow_tracking_uri="http://localhost:5000")
    monkeypatch.setattr(
        "arabic_legal_qa.evaluation.runner.get_settings", lambda *_args: settings
    )

    with pytest.raises(ValueError, match="validate-champion"):
        evaluate_candidates(root=tmp_path, split="heldout")


def test_evaluator_uses_local_mlflow_default_when_uri_is_unset() -> None:
    settings = Settings(_env_file=None)

    assert settings.mlflow_tracking_uri is None
    assert DEFAULT_MLFLOW_TRACKING_URI == "http://localhost:5000"
