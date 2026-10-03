"""Central project configuration for the RAG pipeline.

Keep values that define the corpus or model contract here so indexing and
retrieval cannot silently drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


EXTRACTOR_VERSION = "geometry-glyphs-v1"
EXPECTED_ARTICLES = set(range(1, 1150))
REPEAL_RANGES = ((54, 80), (389, 417))
# E5-small is multilingual (including Arabic), uses 384-dimensional vectors,
# and is substantially lighter than BGE-M3.  Its encoder accepts 512 tokens.
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_EMBEDDING_LIMIT = 512


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for the shared multilingual embedding model."""

    model_name: str = DEFAULT_EMBEDDING_MODEL
    model_revision: str | None = None
    embedding_limit: int = DEFAULT_EMBEDDING_LIMIT
    device: str = "cpu"
    batch_size: int = 4
    normalize_embeddings: bool = True
    # E5 is trained with task prefixes; keep them in the shared model contract
    # so indexing and query-time retrieval cannot silently disagree.
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "


def project_root(start: Path | None = None) -> Path:
    """Find the repository root from ``start`` or the current directory."""

    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "data").exists():
            return candidate
    raise RuntimeError("Could not find project root containing pyproject.toml and data/")


def model_cache_dir(root: Path | None = None) -> Path:
    return project_root(root) / "data/model_cache"
