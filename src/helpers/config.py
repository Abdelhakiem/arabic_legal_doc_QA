"""Central project settings and typed configuration contracts.

All application defaults and environment-backed settings live here. Feature
modules consume the typed configs below rather than reading ``.env`` or
defining their own configuration classes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# Fixed source-edition contract and extraction geometry.
DEFAULT_EXTRACTOR_VERSION = "geometry-glyphs-v1"
DEFAULT_EXPECTED_ARTICLE_START = 1
DEFAULT_EXPECTED_ARTICLE_END = 1149
DEFAULT_REPEAL_RANGES = ((54, 80), (389, 417))
DEFAULT_PDF_PAGE_COUNT = 170
DEFAULT_ARTICLE_TEXT_MAX_LENGTH = 40_000
DEFAULT_PDF_PROGRESS_INTERVAL_PAGES = 25
DEFAULT_EXTRACTION_ROWS_FILENAME = "extraction_rows.json"
DEFAULT_ARTICLE_CANDIDATES_FILENAME = "article_candidates.json"
DEFAULT_EXTRACTION_REPORT_FILENAME = "extraction_report.json"
DEFAULT_VALIDATION_STATUS_FILENAME = "validation_status.json"
DEFAULT_CANONICAL_ARTICLES_FILENAME = "articles.json"
PDF_TABLE_LEFT_X = 30.6
PDF_TABLE_MIDDLE_X = 297.65
PDF_TABLE_RIGHT_X = 564.82
PDF_TABLE_SNAP_TOLERANCE = 4
PDF_TABLE_INTERSECTION_TOLERANCE = 5
PDF_ARTICLE_452_REPAIR_PAGE = 59

# Pipeline and model defaults.
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_EMBEDDING_ONNX_FILENAME = "onnx/model.onnx"
DEFAULT_EMBEDDING_LIMIT = 512
DEFAULT_EMBEDDING_BATCH_SIZE = 4
DEFAULT_EMBEDDING_NORMALIZE = True
DEFAULT_E5_QUERY_PREFIX = "query: "
DEFAULT_E5_PASSAGE_PREFIX = "passage: "
DEFAULT_CHUNK_TOKENS = 480
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_LLM_PROVIDER = "auto"
DEFAULT_LLM_TEMPERATURE = 0.0
DEFAULT_LLM_CONTEXT_TOKENS = 8192
DEFAULT_QDRANT_COLLECTION = "egyptian_civil_code"
DEFAULT_QDRANT_VECTOR_SIZE = 384
DEFAULT_QDRANT_BATCH_SIZE = 64
DEFAULT_QDRANT_SPARSE_MODEL = "Qdrant/bm25"
DEFAULT_QDRANT_EXACT_SEARCH = True
DEFAULT_QDRANT_MANIFEST = "manifest.json"
DEFAULT_RETRIEVAL_K = 5
DEFAULT_PREFETCH_K = 24
DEFAULT_MAX_QUERIES = 3
DEFAULT_MIN_EVIDENCE = 1
DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 8000
DEFAULT_LOG_LEVEL = "INFO"

# Backward-compatible constants for existing parser consumers.
EXTRACTOR_VERSION = DEFAULT_EXTRACTOR_VERSION
EXPECTED_ARTICLES = set(range(DEFAULT_EXPECTED_ARTICLE_START, DEFAULT_EXPECTED_ARTICLE_END + 1))
REPEAL_RANGES = DEFAULT_REPEAL_RANGES


@dataclass(frozen=True)
class EmbeddingConfig:
    """ONNX embedding model contract shared by index and retrieval."""

    model_name: str = DEFAULT_EMBEDDING_MODEL
    model_revision: str | None = None
    onnx_model_filename: str = DEFAULT_EMBEDDING_ONNX_FILENAME
    embedding_limit: int = DEFAULT_EMBEDDING_LIMIT
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    normalize_embeddings: bool = DEFAULT_EMBEDDING_NORMALIZE
    query_prefix: str = DEFAULT_E5_QUERY_PREFIX
    passage_prefix: str = DEFAULT_E5_PASSAGE_PREFIX


@dataclass(frozen=True)
class LLMConfig:
    """Provider-neutral LLM configuration; provider resolution lives in rag.llms."""

    provider: str = DEFAULT_LLM_PROVIDER
    model_name: str | None = None
    groq_token: str | None = field(default=None, repr=False)
    groq_model: str = DEFAULT_GROQ_MODEL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    gemini_model: str = DEFAULT_GEMINI_MODEL
    temperature: float = DEFAULT_LLM_TEMPERATURE
    num_ctx: int = DEFAULT_LLM_CONTEXT_TOKENS


@dataclass(frozen=True)
class ChunkingConfig:
    """Token-budget configuration for article-aware chunking."""

    embedding_limit: int = DEFAULT_EMBEDDING_LIMIT
    chunk_tokens: int | None = DEFAULT_CHUNK_TOKENS
    overlap: int = DEFAULT_CHUNK_OVERLAP


@dataclass(frozen=True)
class QdrantConfig:
    """Local Qdrant collection and indexing configuration."""

    path: Path = Path("data/vector_store/qdrant")
    collection_name: str = DEFAULT_QDRANT_COLLECTION
    vector_size: int = DEFAULT_QDRANT_VECTOR_SIZE
    batch_size: int = DEFAULT_QDRANT_BATCH_SIZE
    sparse_model: str = DEFAULT_QDRANT_SPARSE_MODEL
    exact_search: bool = DEFAULT_QDRANT_EXACT_SEARCH
    manifest_name: str = DEFAULT_QDRANT_MANIFEST


@dataclass(frozen=True)
class RAGConfig:
    """Filesystem and inference configuration for one RAG instance."""

    root: Path
    pdf_path: Path | None = None
    processed_dir: Path | None = None
    model_cache_dir: Path | None = None
    qdrant_path: Path | None = None
    collection_name: str = DEFAULT_QDRANT_COLLECTION
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    exact_search: bool = DEFAULT_QDRANT_EXACT_SEARCH
    qdrant_vector_size: int = DEFAULT_QDRANT_VECTOR_SIZE
    qdrant_batch_size: int = DEFAULT_QDRANT_BATCH_SIZE
    qdrant_sparse_model: str = DEFAULT_QDRANT_SPARSE_MODEL
    qdrant_manifest_name: str = DEFAULT_QDRANT_MANIFEST
    retrieval_k: int = DEFAULT_RETRIEVAL_K
    prefetch_k: int = DEFAULT_PREFETCH_K
    max_queries: int = DEFAULT_MAX_QUERIES
    min_evidence: int = DEFAULT_MIN_EVIDENCE

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        object.__setattr__(self, "root", root)
        for field_name, default_path in (
            ("pdf_path", Path("data/raw/egyptian_civil_law.pdf")),
            ("processed_dir", Path("data/processed")),
            ("model_cache_dir", Path("data/model_cache")),
            ("qdrant_path", Path("data/vector_store/qdrant")),
        ):
            value = getattr(self, field_name) or default_path
            resolved = Path(value).expanduser()
            if not resolved.is_absolute():
                resolved = root / resolved
            object.__setattr__(self, field_name, resolved)

    def qdrant_config(self) -> QdrantConfig:
        return QdrantConfig(
            path=self.qdrant_path,
            collection_name=self.collection_name,
            vector_size=self.qdrant_vector_size,
            batch_size=self.qdrant_batch_size,
            sparse_model=self.qdrant_sparse_model,
            exact_search=self.exact_search,
            manifest_name=self.qdrant_manifest_name,
        )


class Settings(BaseSettings):
    """Environment-backed project settings with sensible RAG defaults.

    Environment variables use the uppercase field names shown in ``.env.example``.
    Existing process environment values take precedence over the project ``.env``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
    )

    # Source corpus and extraction.
    extractor_version: str = DEFAULT_EXTRACTOR_VERSION
    expected_article_start: int = DEFAULT_EXPECTED_ARTICLE_START
    expected_article_end: int = DEFAULT_EXPECTED_ARTICLE_END
    repeal_ranges: tuple[tuple[int, int], ...] = DEFAULT_REPEAL_RANGES
    pdf_page_count: int = DEFAULT_PDF_PAGE_COUNT
    article_text_max_length: int = DEFAULT_ARTICLE_TEXT_MAX_LENGTH
    pdf_progress_interval_pages: int = DEFAULT_PDF_PROGRESS_INTERVAL_PAGES
    pdf_table_left_x: float = PDF_TABLE_LEFT_X
    pdf_table_middle_x: float = PDF_TABLE_MIDDLE_X
    pdf_table_right_x: float = PDF_TABLE_RIGHT_X
    pdf_table_snap_tolerance: int = PDF_TABLE_SNAP_TOLERANCE
    pdf_table_intersection_tolerance: int = PDF_TABLE_INTERSECTION_TOLERANCE
    pdf_article_452_repair_page: int = PDF_ARTICLE_452_REPAIR_PAGE
    extraction_rows_filename: str = DEFAULT_EXTRACTION_ROWS_FILENAME
    article_candidates_filename: str = DEFAULT_ARTICLE_CANDIDATES_FILENAME
    extraction_report_filename: str = DEFAULT_EXTRACTION_REPORT_FILENAME
    validation_status_filename: str = DEFAULT_VALIDATION_STATUS_FILENAME
    canonical_articles_filename: str = DEFAULT_CANONICAL_ARTICLES_FILENAME

    # Paths and persisted artifacts.
    pdf_path: Path = Path("data/raw/egyptian_civil_law.pdf")
    processed_dir: Path = Path("data/processed")
    model_cache_path: Path = Path("data/model_cache")
    qdrant_path: Path = Path("data/vector_store/qdrant")

    # Embedding and chunking.
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_revision: str | None = Field(
        default=None, validation_alias=AliasChoices("E5_SMALL_REVISION", "EMBEDDING_REVISION")
    )
    embedding_onnx_filename: str = DEFAULT_EMBEDDING_ONNX_FILENAME
    embedding_limit: int = DEFAULT_EMBEDDING_LIMIT
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    normalize_embeddings: bool = DEFAULT_EMBEDDING_NORMALIZE
    query_prefix: str = DEFAULT_E5_QUERY_PREFIX
    passage_prefix: str = DEFAULT_E5_PASSAGE_PREFIX
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    # Chat model selection and provider configuration.
    llm_provider: str = DEFAULT_LLM_PROVIDER
    groq_token: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GROQ_TOKEN", "GROQ_API_KEY")
    )
    qroq_model: str = DEFAULT_GROQ_MODEL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    gemini_model: str = DEFAULT_GEMINI_MODEL
    llm_temperature: float = DEFAULT_LLM_TEMPERATURE
    llm_num_ctx: int = DEFAULT_LLM_CONTEXT_TOKENS

    # Qdrant, retrieval, and answer policy.
    qdrant_collection: str = DEFAULT_QDRANT_COLLECTION
    qdrant_vector_size: int = DEFAULT_QDRANT_VECTOR_SIZE
    qdrant_batch_size: int = DEFAULT_QDRANT_BATCH_SIZE
    qdrant_sparse_model: str = DEFAULT_QDRANT_SPARSE_MODEL
    qdrant_exact_search: bool = DEFAULT_QDRANT_EXACT_SEARCH
    qdrant_manifest_name: str = DEFAULT_QDRANT_MANIFEST
    retrieval_k: int = DEFAULT_RETRIEVAL_K
    prefetch_k: int = DEFAULT_PREFETCH_K
    max_queries: int = DEFAULT_MAX_QUERIES
    min_evidence: int = DEFAULT_MIN_EVIDENCE

    # Serving and logs.
    host: str = DEFAULT_API_HOST
    port: int = DEFAULT_API_PORT
    log_level: str = DEFAULT_LOG_LEVEL

    def embedding_config(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            model_name=self.embedding_model,
            model_revision=self.embedding_revision,
            onnx_model_filename=self.embedding_onnx_filename,
            embedding_limit=self.embedding_limit,
            batch_size=self.embedding_batch_size,
            normalize_embeddings=self.normalize_embeddings,
            query_prefix=self.query_prefix,
            passage_prefix=self.passage_prefix,
        )

    def llm_config(self) -> LLMConfig:
        token = self.groq_token.get_secret_value() if self.groq_token else None
        return LLMConfig(
            provider=self.llm_provider,
            groq_token=token or None,
            groq_model=self.qroq_model,
            ollama_model=self.ollama_model,
            base_url=self.ollama_base_url,
            gemini_model=self.gemini_model,
            temperature=self.llm_temperature,
            num_ctx=self.llm_num_ctx,
        )

    def chunking_config(self) -> ChunkingConfig:
        return ChunkingConfig(
            embedding_limit=self.embedding_limit,
            chunk_tokens=self.chunk_tokens,
            overlap=self.chunk_overlap,
        )

    def qdrant_config(self, root: Path | None = None) -> QdrantConfig:
        selected_root = project_root(root) if root is None else Path(root).expanduser().resolve()
        path = self.qdrant_path.expanduser()
        if not path.is_absolute():
            path = selected_root / path
        return QdrantConfig(
            path=path,
            collection_name=self.qdrant_collection,
            vector_size=self.qdrant_vector_size,
            batch_size=self.qdrant_batch_size,
            sparse_model=self.qdrant_sparse_model,
            exact_search=self.qdrant_exact_search,
            manifest_name=self.qdrant_manifest_name,
        )

    def rag_config(self, root: Path | None = None) -> RAGConfig:
        selected_root = project_root(root) if root is None else Path(root).expanduser().resolve()
        return RAGConfig(
            root=selected_root,
            pdf_path=self.pdf_path,
            processed_dir=self.processed_dir,
            model_cache_dir=self.model_cache_path,
            qdrant_path=self.qdrant_path,
            collection_name=self.qdrant_collection,
            chunk_tokens=self.chunk_tokens,
            chunk_overlap=self.chunk_overlap,
            exact_search=self.qdrant_exact_search,
            qdrant_vector_size=self.qdrant_vector_size,
            qdrant_batch_size=self.qdrant_batch_size,
            qdrant_sparse_model=self.qdrant_sparse_model,
            qdrant_manifest_name=self.qdrant_manifest_name,
            retrieval_k=self.retrieval_k,
            prefetch_k=self.prefetch_k,
            max_queries=self.max_queries,
            min_evidence=self.min_evidence,
        )

    @property
    def expected_articles(self) -> set[int]:
        return set(range(self.expected_article_start, self.expected_article_end + 1))


def project_root(start: Path | None = None) -> Path:
    """Find the repository root from ``start`` or the current directory."""

    current = Path(start or Path.cwd()).expanduser().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "data").exists():
            return candidate
    raise RuntimeError("Could not find project root containing pyproject.toml and data/")


def model_cache_dir(root: Path | None = None) -> Path:
    settings = get_settings()
    if root is None:
        try:
            selected_root = project_root()
        except RuntimeError:
            selected_root = Path.cwd().resolve()
    else:
        selected_root = Path(root).expanduser().resolve()
    path = settings.model_cache_path.expanduser()
    return path if path.is_absolute() else selected_root / path


@lru_cache(maxsize=8)
def get_settings(env_file: str | Path | None = None) -> Settings:
    """Load project settings from process env and an optional project ``.env``."""

    if env_file is None:
        try:
            selected_env_file = project_root() / ".env"
        except RuntimeError:
            selected_env_file = Path(".env")
    else:
        selected_env_file = Path(env_file).expanduser().resolve()
    return Settings(_env_file=selected_env_file)


__all__ = [
    "Settings",
    "get_settings",
    "project_root",
    "model_cache_dir",
    "EmbeddingConfig",
    "LLMConfig",
    "ChunkingConfig",
    "QdrantConfig",
    "RAGConfig",
    "EXTRACTOR_VERSION",
    "EXPECTED_ARTICLES",
    "REPEAL_RANGES",
]
