FROM python:3.13-slim

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
    && chown appuser:appuser /app/data/model_cache

# Install runtime dependencies first so source edits do not invalidate this layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project \
    && uv cache clean

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser data/raw/ ./data/raw/
COPY --chown=appuser:appuser data/processed/ ./data/processed/
COPY --chown=appuser:appuser data/vector_store/qdrant/ ./data/vector_store/qdrant/

RUN test -f data/raw/egyptian_civil_law.pdf \
    && test -f data/processed/articles.json \
    && test -f data/vector_store/qdrant/manifest.json

RUN uv sync --locked --no-dev \
    && uv cache clean

USER appuser
EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
