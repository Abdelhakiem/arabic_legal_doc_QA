FROM python:3.13-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

# Create the runtime user before copying files so ownership is assigned at
# copy time rather than by recursively copying the large dependency tree.
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --create-home appuser \
    && mkdir -p /app/data/model_cache \
    && mkdir -p /mlflow \
    && chown appuser:appuser /app/data/model_cache /mlflow

# Install runtime dependencies first so source edits do not invalidate this layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --extra ollama --no-install-project \
    && uv cache clean

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser data/raw/ ./data/raw/
COPY --chown=appuser:appuser data/processed/ ./data/processed/
COPY --chown=appuser:appuser data/vector_store/qdrant/ ./data/vector_store/qdrant/
COPY --chown=appuser:appuser data/evaluation/monitoring_questions.json ./data/evaluation/monitoring_questions.json

RUN test -f data/raw/egyptian_civil_law.pdf \
    && test -f data/processed/articles.json \
    && test -f data/vector_store/qdrant/manifest.json \
    && test -f data/evaluation/monitoring_questions.json

RUN uv sync --locked --no-dev --extra ollama \
    && uv cache clean

USER appuser

# Keep RAGAS/Judge dependencies out of the serving image; this target is only
# used by the scheduled monitoring service.
FROM base AS monitor
USER root
RUN apt-get update \
    && apt-get install --no-install-recommends -y git \
    && rm -rf /var/lib/apt/lists/* \
    && uv sync --locked --no-dev --extra ollama --extra evaluation \
    && uv cache clean
USER appuser
CMD ["arabic-legal-qa-monitor", "--interval-hours", "24"]

# Preserve the slim runtime image as the default target for the API and MLflow.
FROM base AS runtime
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
