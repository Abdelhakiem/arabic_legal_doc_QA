"""Structured JSON logging for the Arabic Legal QA application.

Configure once at an executable boundary (CLI/API startup). Pipeline modules
should use ``logging.getLogger(__name__)`` and put safe structured metadata in
``extra`` or :func:`log_context`; do not log raw questions, retrieved text,
answers, prompts, or credentials.
"""
from __future__ import annotations

import contextvars
import datetime as dt
import json
import logging
import os
import sys
from contextlib import contextmanager
from typing import Any, Iterator


_DEFAULT_LEVEL = "INFO"
_CONTEXT_FIELDS = (
    "request_id",
    "trace_id",
    "operation",
    "stage",
    "corpus_version",
    "index_version",
    "model_name",
    "duration_ms",
    "article_count",
    "document_count",
    "chunk_count",
    "retrieved_count",
    "query_count",
    "reference_count",
    "query_count",
    "reference_count",
    "grounded",
    "error_code",
)
_log_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "arabic_legal_qa_log_context", default={}
)


class JsonFormatter(logging.Formatter):
    """Render one stable, machine-readable JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": _log_context.get().get("request_id", "-"),
        }

        # Context is applied first, then per-event fields can override it.
        for field in _CONTEXT_FIELDS:
            value = _log_context.get().get(field)
            if value is not None:
                payload[field] = value
            if hasattr(record, field):
                payload[field] = getattr(record, field)

        if "request_id" in payload:
            payload["correlation_id"] = payload["request_id"]

        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = event
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Temporarily bind safe RAG/request fields to logs in this context.

    Example: ``with log_context(request_id=rid, operation="query"): ...``
    ContextVar keeps concurrent async requests isolated.
    """

    unknown = set(fields) - set(_CONTEXT_FIELDS)
    if unknown:
        raise ValueError(f"Unsupported logging context fields: {sorted(unknown)}")
    updated = {**_log_context.get(), **fields}
    token = _log_context.set(updated)
    try:
        yield
    finally:
        _log_context.reset(token)


def configure_logging(level: str | int | None = None) -> None:
    """Configure the root logger to emit JSON lines to stdout.

    ``level`` overrides ``ARABIC_LEGAL_QA_LOG_LEVEL``. This function owns the
    root handler configuration; call it once when starting the CLI or API.
    Repeated calls replace the handler rather than duplicating log output.
    """

    configured_level = level if level is not None else os.getenv(
        "ARABIC_LEGAL_QA_LOG_LEVEL", _DEFAULT_LEVEL
    )
    if isinstance(configured_level, str):
        resolved_level = logging.getLevelName(configured_level.upper())
        if not isinstance(resolved_level, int):
            raise ValueError(
                f"Invalid logging level {configured_level!r}; use DEBUG, INFO, "
                "WARNING, ERROR, or CRITICAL"
            )
    elif isinstance(configured_level, int):
        resolved_level = configured_level
    else:
        raise TypeError("level must be a logging level name, integer, or None")

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)
    root_logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root_logger.addHandler(handler)


__all__ = ["JsonFormatter", "configure_logging", "log_context"]
