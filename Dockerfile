FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

# Install runtime dependencies first so source edits do not invalidate this layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY src/ ./src/
COPY data/raw/ ./data/raw/
COPY data/processed/ ./data/processed/
COPY data/vector_store/qdrant/ ./data/vector_store/qdrant/

RUN test -f data/raw/egyptian_civil_law.pdf \
    && test -f data/processed/articles.json \
    && test -f data/vector_store/qdrant/manifest.json \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p data/model_cache \
    && chown -R appuser:appuser /app

RUN uv sync --locked --no-dev

USER appuser
EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
