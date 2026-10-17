"""Validate the bilingual production-monitoring probes against the corpus."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from helpers.config import get_settings, project_root


def _load_monitoring_data() -> tuple[list[dict], dict[int, dict]]:
    root = project_root()
    settings = get_settings()
    question_path = root / "data/evaluation/monitoring_questions.json"
    articles_path = settings.processed_dir.expanduser()
    if not articles_path.is_absolute():
        articles_path = root / articles_path
    articles_path /= settings.canonical_articles_filename

    questions = json.loads(question_path.read_text(encoding="utf-8"))
    article_records = json.loads(articles_path.read_text(encoding="utf-8"))
    return questions, {row["article_number"]: row for row in article_records}


def test_monitoring_set_has_50_unique_balanced_bilingual_questions() -> None:
    questions, _articles = _load_monitoring_data()

    assert len(questions) == 50
    assert len({row["id"] for row in questions}) == 50
    assert len({row["question"] for row in questions}) == 50
    assert Counter(row["language"] for row in questions) == {"en": 25, "ar": 25}

    groups: dict[str, list[dict]] = defaultdict(list)
    for question in questions:
        groups[question["group"]].append(question)
    assert len(groups) == 25
    for rows in groups.values():
        assert len(rows) == 2
        assert {row["language"] for row in rows} == {"en", "ar"}
        assert rows[0]["expected_articles"] == rows[1]["expected_articles"]
        assert rows[0]["answerable"] is rows[1]["answerable"]


def test_monitoring_expected_articles_and_answerability_match_corpus() -> None:
    questions, articles = _load_monitoring_data()

    for question in questions:
        expected = question["expected_articles"]
        assert expected == list(dict.fromkeys(expected)), question["id"]
        assert question["answerable"] is bool(expected), question["id"]
        assert expected == [] or set(expected) <= articles.keys(), question["id"]
        assert question["reference_answer"].strip(), question["id"]

        if expected:
            repeal_flags = {articles[number]["is_repealed"] for number in expected}
            assert repeal_flags == {question["repeal_case"]}, question["id"]

    assert sum(question["repeal_case"] for question in questions) == 4
    assert sum(not question["answerable"] for question in questions) == 4


def test_monitoring_set_covers_multiple_legal_categories() -> None:
    questions, _articles = _load_monitoring_data()

    assert len({question["category"] for question in questions}) >= 8
