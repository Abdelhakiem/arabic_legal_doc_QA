"""Chunking, embedding, and vector database helpers.

This module starts with the same chunking contract used by the reference
notebook: split with the embedding tokenizer, preserve every original
character, and keep article metadata attached to each LangChain document.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from langchain_core.documents import Document

from arabic_legal_qa.rag.llms import EmbeddingBundle, EmbeddingConfig, load_embedding_bundle
from arabic_legal_qa.rag.pdf_loader import Article, stable_hash


@dataclass(frozen=True)
class ChunkingConfig:
    """Token-budget configuration for article-aware chunking."""

    embedding_limit: int = 512
    chunk_tokens: int | None = None
    overlap: int = 0


def load_embedder(config: EmbeddingConfig | None = None, cache_dir: str | None = None) -> EmbeddingBundle:
    """Return the shared embedding bundle managed by `rag.llms`."""

    return load_embedding_bundle(config=config, cache_dir=cache_dir)


def token_count(tokenizer: Any, text: str) -> int:
    """Count tokens with the embedding tokenizer.

    The reference notebook deliberately sizes chunks with the same tokenizer
    used by the embedding model. This avoids creating chunks that later exceed
    the embedding model's input limit.
    """

    # Fast tokenizers can count arbitrarily long input through their backend
    # without Transformers warning that the *unsplit* source exceeds the
    # model limit. The splitter needs this count precisely so it can divide
    # that source into valid chunks.
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        return len(backend.encode(text, add_special_tokens=True).ids)
    return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))


def split_text(text: str, tokenizer: Any, limit: int) -> list[str]:
    """Split text into chunks no larger than `limit` tokenizer tokens.

    The splitter tries increasingly fine boundaries: numbered legal
    paragraphs, lines, sentence punctuation, and finally character boundaries
    found by binary search. It verifies that no text was lost.
    """

    if limit <= 0:
        raise ValueError("Token limit must be positive")
    if token_count(tokenizer, text) <= limit:
        return [text]

    paragraphs = re.split(r"(?<=\n)(?=\s*[()]*[٠-٩0-9]+[).(\s])", text)
    if len(paragraphs) == 1:
        paragraphs = text.splitlines(keepends=True)

    units: list[str] = []
    for paragraph in paragraphs:
        if token_count(tokenizer, paragraph) <= limit:
            units.append(paragraph)
            continue
        for sentence in re.split(r"(?<=[.!؟؛])(?=\s)", paragraph):
            remaining = sentence
            while remaining and token_count(tokenizer, remaining) > limit:
                lo, hi = 1, len(remaining)
                best = 0
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if token_count(tokenizer, remaining[:mid]) <= limit:
                        best, lo = mid, mid + 1
                    else:
                        hi = mid - 1
                if not best:
                    raise ValueError("Token budget cannot accommodate a single character")
                units.append(remaining[:best])
                remaining = remaining[best:]
            if remaining:
                units.append(remaining)

    chunks: list[str] = []
    pending = ""
    for unit in units:
        if pending and token_count(tokenizer, pending + unit) > limit:
            chunks.append(pending)
            pending = ""
        pending += unit
    if pending:
        chunks.append(pending)

    if "".join(chunks) != text:
        raise AssertionError("Chunking lost text")
    if any(token_count(tokenizer, chunk) > limit for chunk in chunks):
        raise AssertionError("Chunking exceeded the token limit")
    return chunks


def _chunk_limit(config: ChunkingConfig) -> int:
    return min(config.chunk_tokens or config.embedding_limit, config.embedding_limit)


def _chunk_id(corpus_hash: str, article_number: int, language: str, position: int,
              text: str, limit: int) -> str:
    identity = stable_hash({
        "corpus": corpus_hash,
        "article": article_number,
        "language": language,
        "position": position,
        "text": text,
        "limit": limit,
    })
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def chunk_documents(documents: Iterable[Document], tokenizer: Any, config: ChunkingConfig,
                    corpus_hash: str) -> list[Document]:
    """Split LangChain documents while preserving article metadata."""

    if config.overlap != 0:
        raise ValueError("This chunker supports zero overlap only")

    limit = _chunk_limit(config)
    chunks: list[Document] = []
    for document in documents:
        metadata = dict(document.metadata)
        article_number = metadata["article_number"]
        language = metadata["language"]
        for position, text in enumerate(split_text(document.page_content, tokenizer, limit)):
            chunks.append(Document(page_content=text, metadata={
                **metadata,
                "chunk_id": _chunk_id(corpus_hash, article_number, language, position, text, limit),
                "chunk_position": position,
                "token_count": token_count(tokenizer, text),
                "corpus_hash": corpus_hash,
            }))
    return chunks


def make_chunks(articles: Iterable[Article], tokenizer: Any, config: ChunkingConfig,
                corpus_hash: str, document_factory: Callable[..., Document] = Document) -> list[Document]:
    """Create language-specific chunks from validated article records.

    This mirrors the reference notebook: Arabic and English are chunked
    separately, each chunk keeps canonical article metadata, and chunk IDs are
    deterministic for reproducible vector-store inserts.
    """

    if config.overlap != 0:
        raise ValueError("This chunker supports zero overlap only")

    limit = _chunk_limit(config)
    documents: list[Document] = []
    for article in articles:
        metadata = article.model_dump(exclude={"text_ar", "text_en"})
        for language, text in (("ar", article.text_ar), ("en", article.text_en)):
            if not text:
                continue
            for position, chunk in enumerate(split_text(text, tokenizer, limit)):
                documents.append(document_factory(page_content=chunk, metadata={
                    **metadata,
                    "language": language,
                    "chunk_id": _chunk_id(corpus_hash, article.article_number, language, position, chunk, limit),
                    "chunk_position": position,
                    "token_count": token_count(tokenizer, chunk),
                    "corpus_hash": corpus_hash,
                }))
    return documents
