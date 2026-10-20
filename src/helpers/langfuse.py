"""Best-effort Langfuse lifecycle and observation helpers."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import Any, Iterator

from helpers.config import Settings

logger = logging.getLogger(__name__)


def create_langfuse_client(settings: Settings) -> Any | None:
    """Initialize Langfuse when enabled and credentials are configured.

    Observability is optional by default. ``langfuse_required`` makes startup
    fail closed if keys are missing, the SDK cannot initialize, or auth fails.
    """

    if not settings.langfuse_enabled:
        if settings.langfuse_required:
            raise RuntimeError("Langfuse tracing is required but disabled")
        return None
    secret_key = (
        settings.langfuse_secret_key.get_secret_value()
        if settings.langfuse_secret_key
        else ""
    )
    if not settings.langfuse_public_key or not secret_key:
        if settings.langfuse_required:
            raise RuntimeError(
                "Langfuse tracing is required but public/secret API keys are missing"
            )
        logger.warning(
            "Langfuse tracing is enabled but API keys are missing; tracing disabled",
            extra={"event": "langfuse.initialization.skipped"},
        )
        return None

    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=secret_key,
            base_url=settings.langfuse_base_url,
            environment=settings.langfuse_environment,
        )
    except Exception as exc:
        if settings.langfuse_required:
            raise RuntimeError("Langfuse client initialization failed") from exc
        logger.warning(
            "Langfuse initialization failed; continuing without tracing",
            exc_info=True,
            extra={"event": "langfuse.initialization.failed"},
        )
        return None

    if settings.langfuse_required:
        try:
            authorized = client.auth_check()
        except Exception as exc:
            client.shutdown()
            raise RuntimeError("Langfuse authentication check failed") from exc
        if not authorized:
            client.shutdown()
            raise RuntimeError(
                "Langfuse rejected the configured API keys; check the project keys"
            )

    logger.info(
        "Langfuse tracing initialized",
        extra={
            "event": "langfuse.initialization.completed",
            "environment": settings.langfuse_environment,
            "base_url": settings.langfuse_base_url,
            "content_capture": settings.langfuse_capture_content,
        },
    )
    return client


@contextmanager
def observation(
    client: Any | None, *, required: bool = False, **kwargs: Any
) -> Iterator[Any | None]:
    """Create a nested observation, optionally failing if tracing is required."""

    if client is None:
        if required:
            raise RuntimeError(
                "Langfuse observation required but client is unavailable"
            )
        yield None
        return

    try:
        context = client.start_as_current_observation(**kwargs)
        span = context.__enter__()
    except Exception as exc:
        if required:
            raise RuntimeError("Could not start required Langfuse observation") from exc
        logger.warning(
            "Could not start Langfuse observation; continuing without this span",
            exc_info=True,
            extra={"event": "langfuse.observation.start_failed"},
        )
        yield None
        return

    try:
        yield span
    except BaseException:
        try:
            context.__exit__(*sys.exc_info())
        except Exception:
            logger.warning(
                "Could not close failed Langfuse observation",
                exc_info=True,
                extra={"event": "langfuse.observation.close_failed"},
            )
        raise
    else:
        try:
            context.__exit__(None, None, None)
        except Exception:
            if required:
                raise RuntimeError(
                    "Could not close required Langfuse observation"
                ) from None
            logger.warning(
                "Could not close Langfuse observation; RAG result is unaffected",
                exc_info=True,
                extra={"event": "langfuse.observation.close_failed"},
            )


def update_observation(span: Any | None, **kwargs: Any) -> None:
    """Update a span if tracing is active, suppressing telemetry-only failures."""

    if span is None:
        return
    try:
        span.update(**kwargs)
    except Exception:
        logger.warning(
            "Could not update Langfuse observation; RAG result is unaffected",
            exc_info=True,
            extra={"event": "langfuse.observation.update_failed"},
        )


def shutdown_langfuse(client: Any | None) -> None:
    """Flush and stop the process-scoped Langfuse client during API shutdown."""

    if client is None:
        return
    try:
        client.shutdown()
    except Exception:
        logger.warning(
            "Could not flush Langfuse observations during shutdown",
            exc_info=True,
            extra={"event": "langfuse.shutdown.failed"},
        )
