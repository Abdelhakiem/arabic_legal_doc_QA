"""Acceptance checks against the canonical, processed Civil Code corpus."""

from __future__ import annotations

import json

import pytest

from helpers.config import Settings, get_settings, project_root
from arabic_legal_qa.rag.pdf_loader import validate_articles


@pytest.fixture(scope="module")
def corpus() -> tuple[list[dict], Settings]:
    settings = get_settings()
    root = project_root()
    processed_dir = settings.processed_dir.expanduser()
    if not processed_dir.is_absolute():
        processed_dir = root / processed_dir
    articles_path = processed_dir / settings.canonical_articles_filename
    if not articles_path.is_file():
        pytest.fail(
            f"Canonical corpus not found at {articles_path}; run corpus extraction first."
        )
    records = json.loads(articles_path.read_text(encoding="utf-8"))
    assert isinstance(records, list), f"Expected a JSON list in {articles_path}"
    return records, settings


def test_article_numbers_are_contiguous(corpus: tuple[list[dict], Settings]) -> None:
    records, settings = corpus
    article_numbers = [record["article_number"] for record in records]
    expected = list(range(settings.expected_article_start, settings.expected_article_end + 1))

    assert len(article_numbers) == len(set(article_numbers)), "Duplicate article numbers found"
    assert sorted(article_numbers) == expected, "Canonical article numbers have gaps or unexpected values"


def test_every_record_has_nonempty_arabic_text(corpus: tuple[list[dict], Settings]) -> None:
    records, _ = corpus
    empty_articles = [
        record.get("article_number")
        for record in records
        if not isinstance(record.get("text_ar"), str) or not record["text_ar"].strip()
    ]

    assert not empty_articles, f"Articles with empty Arabic text: {empty_articles}"


def test_article_text_does_not_exceed_sane_length(corpus: tuple[list[dict], Settings]) -> None:
    records, settings = corpus
    oversized = [
        (record.get("article_number"), field, len(record[field]))
        for record in records
        for field in ("text_ar", "text_en")
        if isinstance(record.get(field), str)
        and len(record[field]) > settings.article_text_max_length
    ]

    assert not oversized, (
        f"Article text exceeds {settings.article_text_max_length} characters: {oversized}"
    )


def test_repeal_ranges_are_flagged_and_corpus_passes_validator(
    corpus: tuple[list[dict], Settings],
) -> None:
    records, settings = corpus
    incorrect_flags = []
    for record in records:
        number = record["article_number"]
        should_be_repealed = any(
            start <= number <= end for start, end in settings.repeal_ranges
        )
        if record.get("is_repealed") is not should_be_repealed:
            incorrect_flags.append(number)

    assert not incorrect_flags, f"Incorrect repeal flags: {incorrect_flags}"
    assert validate_articles(records, settings) == []
