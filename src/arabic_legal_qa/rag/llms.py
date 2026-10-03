"""Shared model loaders for RAG components.

Embedding models are expensive to initialize, so this module owns both local
disk caching and process-level reuse. Indexing and retrieval should import the
embedding bundle from here instead of constructing model objects themselves.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from langchain_core.embeddings import Embeddings
from arabic_legal_qa.rag.config import EmbeddingConfig, model_cache_dir, project_root


@dataclass(frozen=True)
class EmbeddingBundle:
    """Resolved embedding model, tokenizer, and immutable config."""

    embedder: object
    tokenizer: object
    config: EmbeddingConfig
    cache_dir: Path


class PrefixedEmbeddings(Embeddings):
    """Add the prefixes expected by multilingual E5 for each task."""

    def __init__(self, base: Embeddings, query_prefix: str, passage_prefix: str):
        self.base = base
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base.embed_documents([self._prefix(text, self.passage_prefix) for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query(self._prefix(text, self.query_prefix))

    @staticmethod
    def _prefix(text: str, prefix: str) -> str:
        return text if not prefix or text.startswith(prefix) else prefix + text


def default_model_cache_dir(project_root: Path | None = None) -> Path:
    """Return the project-local model cache directory."""

    return model_cache_dir(project_root)


def load_embedding_bundle(config: EmbeddingConfig | None = None, cache_dir: Path | str | None = None) -> EmbeddingBundle:
    """Load or reuse the embedding model and tokenizer.

    The first call downloads missing model files into `data/model_cache`.
    Later calls with the same resolved config and cache directory reuse the
    in-memory objects. If files already exist locally, Hugging Face loads them
    from that cache.
    """

    selected = config or EmbeddingConfig()
    selected_cache = Path(cache_dir) if cache_dir is not None else default_model_cache_dir()
    selected_cache.mkdir(parents=True, exist_ok=True)
    revision = selected.model_revision or _resolve_model_revision(selected.model_name)
    resolved = replace(selected, model_revision=revision)
    return _load_embedding_bundle_cached(resolved, selected_cache.resolve())


def _resolve_model_revision(model_name: str) -> str:
    from huggingface_hub import model_info

    return model_info(model_name).sha


@lru_cache(maxsize=4)
def _load_embedding_bundle_cached(config: EmbeddingConfig, cache_dir: Path) -> EmbeddingBundle:
    from langchain_huggingface import HuggingFaceEmbeddings
    from transformers import AutoTokenizer

    base_embedder = HuggingFaceEmbeddings(
        model_name=config.model_name,
        model_kwargs={
            "device": config.device,
            "revision": config.model_revision,
            "trust_remote_code": False,
        },
        encode_kwargs={
            "normalize_embeddings": config.normalize_embeddings,
            "batch_size": config.batch_size,
        },
        cache_folder=str(cache_dir),
        show_progress=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        trust_remote_code=False,
        cache_dir=str(cache_dir),
    )
    if config.embedding_limit > tokenizer.model_max_length:
        raise ValueError("Configured embedding limit exceeds the tokenizer's supported input length")
    embedder = PrefixedEmbeddings(
        base=base_embedder,
        query_prefix=config.query_prefix,
        passage_prefix=config.passage_prefix,
    )
    return EmbeddingBundle(embedder=embedder, tokenizer=tokenizer, config=config, cache_dir=cache_dir)
