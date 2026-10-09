"""Unit tests for the API request and response contracts; no endpoints involved."""

import pytest
from pydantic import ValidationError

from api.schema import AskRequest, AskResponse


def test_ask_request_trims_question() -> None:
    payload = AskRequest(question="  ما آثار العقد؟  ")

    assert payload.question == "ما آثار العقد؟"


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_ask_request_rejects_empty_or_whitespace_question(question: str) -> None:
    with pytest.raises(ValidationError, match="question"):
        AskRequest(question=question)


def test_ask_response_accepts_canonical_article_citations() -> None:
    response = AskResponse(
        answer="The rule is described in Article 147.",
        sources=["Egyptian Civil Code, Article 147"],
    )

    assert response.sources == ["Egyptian Civil Code, Article 147"]


def test_ask_response_rejects_chunk_ids_and_duplicate_citations() -> None:
    with pytest.raises(ValidationError, match="canonical citations"):
        AskResponse(answer="Answer", sources=["internal-chunk-id"])

    with pytest.raises(ValidationError, match="duplicate"):
        AskResponse(
            answer="Answer",
            sources=[
                "Egyptian Civil Code, Article 147",
                "Egyptian Civil Code, Article 147",
            ],
        )


def test_ask_response_allows_empty_sources_for_abstention() -> None:
    response = AskResponse(answer="There is not enough evidence.", sources=[])

    assert response.sources == []
