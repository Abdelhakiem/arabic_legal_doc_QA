"""Pydantic contracts for the HTTP API."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


_ARTICLE_CITATION = re.compile(r"^Egyptian Civil Code, Article [1-9][0-9]*$")


def is_article_citation(value: str) -> bool:
    """Return whether a value is a canonical Civil Code article citation."""

    return bool(_ARTICLE_CITATION.fullmatch(value))


class AskRequest(BaseModel):
    """Question payload accepted by a future ``POST /ask`` endpoint."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, description="Question in Arabic or English")

    @field_validator("question")
    @classmethod
    def question_must_contain_non_whitespace(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("question must not be empty or whitespace-only")
        return question


class AskResponse(BaseModel):
    """Grounded answer and canonical article citations."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    sources: list[str] = Field(
        description="Unique article citations; never chunk IDs. Empty when there are no sources."
    )

    @field_validator("sources")
    @classmethod
    def sources_must_be_unique_article_citations(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if not is_article_citation(value)]
        if invalid:
            raise ValueError(
                "sources must contain canonical citations like "
                "'Egyptian Civil Code, Article 147'"
            )
        if len(values) != len(set(values)):
            raise ValueError("sources must not contain duplicate article citations")
        return values


class HealthResponse(BaseModel):
    """Service readiness and local vector-index size."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["healthy", "unhealthy"]
    documents_indexed: int = Field(ge=0)
    errors: list[str] | None = None
