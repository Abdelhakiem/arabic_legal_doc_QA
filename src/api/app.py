"""FastAPI application for grounded Egyptian Civil Code Q&A."""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from helpers.config import get_settings
from helpers.logging_config import configure_logging
from arabic_legal_qa.rag.orchestrator import RAG, create_rag
from api.schema import AskRequest, AskResponse, HealthResponse, is_article_citation

logger = logging.getLogger(__name__)


def get_rag(request: Request) -> RAG:
    """Return the process-scoped RAG instance only after readiness succeeds."""

    if not request.app.state.ready or request.app.state.rag is None:
        raise HTTPException(
            status_code=503,
            detail={"errors": request.app.state.startup_errors or ["RAG service is not ready"]},
        )
    return request.app.state.rag


def _source_citations(result: dict[str, Any]) -> list[str]:
    """Extract unique canonical citations, dropping all non-citation metadata."""

    citations: list[str] = []
    seen: set[str] = set()
    for reference in result.get("references", []):
        citation = reference.get("citation")
        if isinstance(citation, str) and is_article_citation(citation) and citation not in seen:
            seen.add(citation)
            citations.append(citation)
    return citations


def create_app(rag_factory: Callable[[], RAG] | None = None) -> FastAPI:
    """Create the API app with process-scoped startup and shutdown resources."""

    factory = rag_factory or create_rag

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.rag = None
        app.state.ready = False
        app.state.documents_indexed = 0
        app.state.startup_errors = []
        rag = None
        try:
            configure_logging(get_settings().log_level)
            rag = factory()
            app.state.rag = rag
            app.state.documents_indexed = rag.initialize()
            app.state.ready = True
            logger.info(
                "FastAPI RAG service is ready",
                extra={
                    "event": "api.startup.completed",
                    "operation": "startup",
                    "document_count": app.state.documents_indexed,
                },
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            app.state.startup_errors = [error]
            logger.exception(
                "FastAPI RAG service failed readiness checks",
                extra={"event": "api.startup.failed", "operation": "startup"},
            )
        try:
            yield
        finally:
            if rag is not None:
                rag.close()

    app = FastAPI(
        title="Arabic Legal QA",
        description="Grounded Q&A over the Egyptian Civil Code.",
        lifespan=lifespan,
    )

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest, rag: RAG = Depends(get_rag)) -> AskResponse:
        result = rag.query(payload.question)
        return AskResponse(
            answer=result["answer"],
            sources=_source_citations(result),
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        response_model_exclude_none=True,
    )
    def health(request: Request) -> HealthResponse | JSONResponse:
        if not request.app.state.ready or request.app.state.rag is None:
            return JSONResponse(
                status_code=503,
                content=HealthResponse(
                    status="unhealthy",
                    documents_indexed=0,
                    errors=request.app.state.startup_errors or ["RAG service is not ready"],
                ).model_dump(exclude_none=True),
            )
        try:
            count = request.app.state.rag.health()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            request.app.state.ready = False
            request.app.state.startup_errors = [error]
            logger.exception(
                "FastAPI RAG health check failed",
                extra={"event": "api.health.failed", "operation": "health"},
            )
            return JSONResponse(
                status_code=503,
                content=HealthResponse(
                    status="unhealthy",
                    documents_indexed=0,
                    errors=[error],
                ).model_dump(exclude_none=True),
            )
        return HealthResponse(status="healthy", documents_indexed=count)

    return app


app = create_app()

__all__ = ["app", "create_app", "get_rag"]
