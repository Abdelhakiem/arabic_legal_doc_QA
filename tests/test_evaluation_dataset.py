"""Check that the compact evaluation set is balanced and grounded in the corpus."""

from __future__ import annotations

import json
from collections import Counter

from helpers.config import get_settings, project_root


def _load_evaluation_data() -> tuple[list[dict], dict[int, dict]]:
    root = project_root()
    settings = get_settings()
    question_path = root / "data/evaluation/foundation_questions.json"
    articles_path = settings.processed_dir.expanduser()
    if not articles_path.is_absolute():
        articles_path = root / articles_path
    articles_path /= settings.canonical_articles_filename

    questions = json.loads(question_path.read_text(encoding="utf-8"))
    article_records = json.loads(articles_path.read_text(encoding="utf-8"))
    return questions, {row["article_number"]: row for row in article_records}


def test_foundation_set_has_ten_unique_bilingual_questions_and_disjoint_splits() -> (
    None
):
    questions, _articles = _load_evaluation_data()

    assert len(questions) == 10
    assert len({row["id"] for row in questions}) == 10
    assert Counter(row["language"] for row in questions) == {"en": 5, "ar": 5}
    assert Counter(row["split"] for row in questions) == {"dev": 6, "heldout": 4}
    assert {row["language"] for row in questions if row["split"] == "dev"} == {
        "en",
        "ar",
    }
    assert {row["language"] for row in questions if row["split"] == "heldout"} == {
        "en",
        "ar",
    }

    groups_by_split = {
        split: {row["group"] for row in questions if row["split"] == split}
        for split in ("dev", "heldout")
    }
    assert groups_by_split["dev"].isdisjoint(groups_by_split["heldout"])


def test_expected_articles_exist_and_repeal_labels_match_canonical_articles() -> None:
    questions, articles = _load_evaluation_data()

    for question in questions:
        expected = question["expected_articles"]
        assert len(expected) == len(set(expected)), question["id"]
        assert question["answerable"] is bool(expected), question["id"]
        assert set(expected) <= articles.keys(), question["id"]

        if question["repeal_case"]:
            assert expected, question["id"]
            assert all(articles[number]["is_repealed"] for number in expected), (
                question["id"]
            )
        else:
            assert all(not articles[number]["is_repealed"] for number in expected), (
                question["id"]
            )


def test_dataset_includes_multi_article_questions_and_out_of_scope_cases() -> None:
    questions, _articles = _load_evaluation_data()

    assert sum(len(row["expected_articles"]) > 1 for row in questions) >= 4
    assert sum(not row["answerable"] for row in questions) == 2
