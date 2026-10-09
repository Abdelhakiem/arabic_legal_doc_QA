"""Focused unit tests for the corpus validation gate."""

from __future__ import annotations

import pytest

from helpers.config import Settings
from arabic_legal_qa.rag.pdf_loader import validate_articles


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        expected_article_start=1,
        expected_article_end=3,
        repeal_ranges=((2, 2),),
        article_text_max_length=20,
    )


def valid_articles() -> list[dict]:
    return [
        {"article_number": 1, "text_ar": "نص المادة الأولى", "is_repealed": False},
        {"article_number": 2, "text_ar": "نص المادة الثانية", "is_repealed": True},
        {"article_number": 3, "text_ar": "نص المادة الثالثة", "is_repealed": False},
    ]


def test_article_numbers_are_contiguous(settings: Settings) -> None:
    articles = valid_articles()
    assert validate_articles(articles, settings) == []

    articles.pop(1)
    errors = validate_articles(articles, settings)
    assert any("not contiguous" in error and "[2]" in error for error in errors)


def test_every_article_has_nonempty_arabic_text(settings: Settings) -> None:
    articles = valid_articles()
    articles[0]["text_ar"] = "  \n"

    errors = validate_articles(articles, settings)

    assert "Article 1 has empty Arabic text" in errors


@pytest.mark.parametrize("field", ["text_ar", "text_en"])
def test_article_text_stays_within_sane_length(settings: Settings, field: str) -> None:
    articles = valid_articles()
    articles[0][field] = "ط" * (settings.article_text_max_length + 1)

    errors = validate_articles(articles, settings)

    language = "Arabic" if field == "text_ar" else "English"
    assert any(f"Article 1 {language} text exceeds 20 characters" in error for error in errors)


def test_repealed_articles_are_flagged(settings: Settings) -> None:
    articles = valid_articles()
    articles[1]["is_repealed"] = False
    articles[2]["is_repealed"] = True

    errors = validate_articles(articles, settings)

    assert "Article 2 should be flagged as repealed" in errors
    assert "Article 3 should be flagged as active" in errors
