"""FastAPI application for grounded Egyptian Civil Code Q&A."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from api.schema import AskRequest, AskResponse, HealthResponse, is_article_citation
from arabic_legal_qa.rag.orchestrator import RAG, create_rag
from helpers.config import get_settings
from helpers.langfuse import create_langfuse_client, shutdown_langfuse
from helpers.logging_config import configure_logging
from helpers.metrics import create_http_metrics

logger = logging.getLogger(__name__)


def get_rag(request: Request) -> RAG:
    """Return the process-scoped RAG instance only after readiness succeeds."""

    if not request.app.state.ready or request.app.state.rag is None:
        raise HTTPException(
            status_code=503,
            detail={
                "errors": request.app.state.startup_errors
                or ["RAG service is not ready"]
            },
        )
    return request.app.state.rag


def _source_citations(result: dict[str, Any]) -> list[str]:
    """Extract unique canonical citations, dropping all non-citation metadata."""

    citations: list[str] = []
    seen: set[str] = set()
    for reference in result.get("references", []):
        citation = reference.get("citation")
        if (
            isinstance(citation, str)
            and is_article_citation(citation)
            and citation not in seen
        ):
            seen.add(citation)
            citations.append(citation)
    return citations


def create_app(rag_factory: Callable[[], RAG] | None = None) -> FastAPI:
    """Create the API app with process-scoped startup and shutdown resources."""

    factory = rag_factory or create_rag
    http_metrics = create_http_metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.rag = None
        app.state.ready = False
        app.state.documents_indexed = 0
        app.state.startup_errors = []
        app.state.http_metrics = http_metrics
        rag = None
        langfuse_client = None
        try:
            settings = get_settings()
            configure_logging(settings.log_level)
            langfuse_client = create_langfuse_client(settings)
            rag = factory()
            if isinstance(rag, RAG):
                rag.langfuse_client = langfuse_client
                rag.langfuse_capture_content = settings.langfuse_capture_content
                rag.langfuse_required = settings.langfuse_required
            elif settings.langfuse_required:
                raise RuntimeError(
                    "Langfuse is required but the configured RAG factory does not "
                    "support request tracing"
                )
            app.state.rag = rag
            app.state.langfuse_client = langfuse_client
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
            try:
                if rag is not None:
                    rag.close()
            finally:
                shutdown_langfuse(langfuse_client)

    app = FastAPI(
        title="Arabic Legal QA",
        description="Grounded Q&A over the Egyptian Civil Code.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def record_http_metrics(request: Request, call_next):
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            route_name = getattr(route, "path", "unmatched")
            http_metrics.observe(
                request.method,
                route_name,
                status,
                time.perf_counter() - started,
            )

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        body, content_type = http_metrics.render()
        return Response(content=body, media_type=content_type)

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
                    errors=request.app.state.startup_errors
                    or ["RAG service is not ready"],
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
