"""Application orchestrator for ingestion and inference.

The notebook is an executable experiment; this module is the reusable
pipeline boundary. It keeps the same validated loader, E5-small embedding
contract, article-aware chunking, and local Qdrant hybrid index.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from helpers.config import (
    ChunkingConfig,
    EmbeddingConfig,
    LLMConfig,
    QdrantConfig,
    RAGConfig,
    get_settings,
    project_root,
)
from arabic_legal_qa.rag.helper import stable_hash
from arabic_legal_qa.rag.llms import (
    EmbeddingBundle,
    load_embedding_bundle,
    load_llm,
    resolve_llm_config,
)
from arabic_legal_qa.rag.pdf_loader import EgyptianCivilCodeLoader
from arabic_legal_qa.rag.qdrant_db import (
    build_index,
    hybrid_search,
    open_index,
    prepare_retrieval,
)
from arabic_legal_qa.rag.chunking import chunk_documents

logger = logging.getLogger(__name__)


ANSWER_PROMPT = """You are a careful legal information assistant.
Answer the user's question using only the retrieved Egyptian Civil Code evidence below.
The preferred user language is: {language}. Write the answer in that language.
If the evidence is insufficient, say so clearly and do not guess.
Do not invent article numbers or legal rules. Cite supporting articles inline
using the exact form [Article N]. Treat repealed articles as repealed.

Retrieved evidence:
{context}

User question:
{question}
"""


def _preferred_language(text: str) -> str:
    arabic = len(re.findall(r"[\u0600-\u06ff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return "Arabic" if arabic >= max(1, latin) else "English"


def _message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    return str(content).strip()


@dataclass
class RAG:
    """Own ingestion state and provide retrieval from the persisted index."""

    config: RAGConfig
    embedding_config: EmbeddingConfig = field(default_factory=lambda: get_settings().embedding_config())
    llm_config: LLMConfig = field(default_factory=lambda: get_settings().llm_config())
    embedding_bundle: EmbeddingBundle | None = field(default=None, init=False)
    llm: Any | None = field(default=None, init=False)
    qdrant_client: Any | None = field(default=None, init=False)
    qdrant_config: QdrantConfig | None = field(default=None, init=False)

    def _load_embeddings(self) -> EmbeddingBundle:
        if self.embedding_bundle is None:
            self.embedding_bundle = load_embedding_bundle(
                config=self.embedding_config,
                cache_dir=self.config.model_cache_dir,
            )
        return self.embedding_bundle

    def _qdrant_config(self) -> QdrantConfig:
        if self.qdrant_config is None:
            self.qdrant_config = self.config.qdrant_config()
        return self.qdrant_config

    def _load_llm(self) -> Any:
        if self.llm is None:
            self.llm_config = resolve_llm_config(self.llm_config)
            self.llm = load_llm(self.llm_config)
        return self.llm

    def ingest(self) -> dict[str, Any]:
        """Load, chunk, embed, and replace the stable Qdrant collection."""
        started = time.perf_counter()
        logger.info(
            "RAG ingestion started",
            extra={"event": "rag.ingest.started", "operation": "ingest"},
        )
        try:
            loader = EgyptianCivilCodeLoader(
                pdf_path=self.config.pdf_path,
                output_dir=self.config.processed_dir,
                language="bilingual",
            )
            documents = loader.load()
            bundle = self._load_embeddings()
            corpus_hash = stable_hash([document.page_content for document in documents])
            chunks = chunk_documents(
                documents=documents,
                tokenizer=bundle.tokenizer,
                config=replace(
                    get_settings().chunking_config(),
                    embedding_limit=bundle.config.embedding_limit,
                    chunk_tokens=self.config.chunk_tokens,
                    overlap=self.config.chunk_overlap,
                ),
                corpus_hash=corpus_hash,
            )
            qdrant_config = self._qdrant_config()
            result = build_index(
                documents=chunks,
                embedding_bundle=bundle,
                config=qdrant_config,
                corpus_hash=corpus_hash,
            )
            self.qdrant_client = result["client"]
            summary = {
                "documents": len(documents),
                "chunks": len(chunks),
                "corpus_hash": corpus_hash,
                "embedding_model": bundle.config.model_name,
                "collection": qdrant_config.collection_name,
                "indexed": result["count"],
                "report": loader.report,
            }
            logger.info(
                "RAG ingestion completed",
                extra={
                    "event": "rag.ingest.completed",
                    "operation": "ingest",
                    "stage": "index",
                    "corpus_version": corpus_hash,
                    "model_name": bundle.config.model_name,
                    "document_count": len(documents),
                    "chunk_count": len(chunks),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            return summary
        except Exception:
            logger.exception(
                "RAG ingestion failed",
                extra={
                    "event": "rag.ingest.failed",
                    "operation": "ingest",
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            raise

    def _index_ready(self) -> bool:
        """Return whether a completed local index is available."""

        manifest_path = self.config.qdrant_path / self.config.qdrant_manifest_name
        if not manifest_path.exists():
            return False
        try:
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("complete") is not True:
                return False
            # A pre-ONNX index has vectors from a different inference path.
            # Rebuild it once so the manifest always describes the active
            # backend and the collection is internally consistent.
            if manifest.get("embedding_backend") != "onnxruntime-cpu":
                return False
            client = open_index(self._qdrant_config())
            count = client.count(self._qdrant_config().collection_name, exact=True).count
            if count != manifest.get("chunk_count"):
                client.close()
                return False
            self.qdrant_client = client
            return count > 0
        except (FileNotFoundError, OSError, ValueError, KeyError):
            return False

    def retrieval_client(self) -> Any:
        """Return a ready Qdrant client, ingesting only when necessary."""

        if self.qdrant_client is None and not self._index_ready():
            self.ingest()
        return self.load_index()

    def load_index(self) -> Any:
        """Open the persisted collection without rebuilding it."""

        if self.qdrant_client is None:
            self.qdrant_client = open_index(self._qdrant_config())
        return self.qdrant_client

    def initialize(self) -> int:
        """Prepare the index, embedding model, and chat model once for serving.

        If the index is absent or incomplete, ``retrieval_client`` runs the
        existing ingestion flow before startup is considered successful.
        Model objects and the Qdrant client are retained on this RAG instance.
        """

        client = self.retrieval_client()
        self._load_embeddings()
        prepare_retrieval(self._qdrant_config())
        self._load_llm()
        indexed = client.count(self._qdrant_config().collection_name, exact=True).count
        if indexed <= 0:
            raise RuntimeError("The Qdrant collection is empty after initialization")
        return indexed

    def health(self) -> int:
        """Check the initialized local index and return its document count."""

        if self.qdrant_client is None:
            raise RuntimeError("RAG initialization has not completed")
        indexed = self.qdrant_client.count(
            self._qdrant_config().collection_name,
            exact=True,
        ).count
        if indexed <= 0:
            raise RuntimeError("The Qdrant collection is empty")
        return indexed

    def retrieve(self, question: str, k: int | None = None,
                 prefetch_k: int | None = None) -> list[Document]:
        """Run native dense+sparse hybrid retrieval for one question."""

        if not question.strip():
            raise ValueError("Question must not be empty")
        k = self.config.retrieval_k if k is None else k
        prefetch_k = self.config.prefetch_k if prefetch_k is None else prefetch_k
        return hybrid_search(
            query=question,
            embedding_bundle=self._load_embeddings(),
            client=self.retrieval_client(),
            config=self._qdrant_config(),
            k=k,
            prefetch_k=prefetch_k,
        )

    @staticmethod
    def _references(documents: list[Document]) -> list[dict[str, Any]]:
        references: list[dict[str, Any]] = []
        seen: set[tuple[Any, Any]] = set()
        for document in documents:
            metadata = document.metadata
            key = (metadata.get("article_number"), metadata.get("language"))
            if key in seen:
                continue
            seen.add(key)
            references.append({
                "article_number": metadata.get("article_number"),
                "language": metadata.get("language"),
                "citation": metadata.get("citation"),
                "source_page": metadata.get("source_page"),
                "is_repealed": metadata.get("is_repealed", False),
            })
        return references

    def query(self, question: str, k: int | None = None, prefetch_k: int | None = None,
              min_evidence: int | None = None) -> dict[str, Any]:
        """Retrieve using the user's question verbatim and answer with evidence."""
        k = self.config.retrieval_k if k is None else k
        prefetch_k = self.config.prefetch_k if prefetch_k is None else prefetch_k
        min_evidence = self.config.min_evidence if min_evidence is None else min_evidence
        started = time.perf_counter()
        logger.info(
            "RAG query started",
            extra={"event": "rag.query.started", "operation": "query"},
        )
        try:
            documents = self.retrieve(question, k=k, prefetch_k=prefetch_k)
            references = self._references(documents)
            language = _preferred_language(question)
            logger.info(
                "RAG retrieval completed",
                extra={
                    "event": "rag.retrieval.completed",
                    "operation": "query",
                    "stage": "retrieval",
                    "query_count": 1,
                    "retrieved_count": len(documents),
                    "reference_count": len(references),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            if len(documents) < min_evidence:
                answer = (
                    "لا توجد أدلة كافية في النص المسترجع للإجابة عن هذا السؤال."
                    if language == "Arabic"
                    else "There is insufficient evidence in the retrieved text to answer this question."
                )
                logger.warning(
                    "RAG query abstained: insufficient evidence",
                    extra={
                        "event": "rag.query.abstained",
                        "operation": "query",
                        "stage": "retrieval",
                        "query_count": 1,
                        "retrieved_count": len(documents),
                        "reference_count": len(references),
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    },
                )
                return {"answer": answer, "references": references, "evidence_count": 0}

            context_parts = []
            for index, document in enumerate(documents, start=1):
                metadata = document.metadata
                context_parts.append(
                    f"Evidence {index} | Article {metadata.get('article_number')} | "
                    f"Language {metadata.get('language')} | Repealed {metadata.get('is_repealed', False)}\n"
                    f"{document.page_content}"
                )
            response = self._load_llm().invoke(
                ANSWER_PROMPT.format(
                    language=language,
                    context="\n\n".join(context_parts),
                    question=question,
                )
            )
            result = {
                "answer": _message_text(response),
                "references": references,
                "evidence_count": len(documents),
            }
            logger.info(
                "RAG answer generated",
                extra={
                    "event": "rag.query.completed",
                    "operation": "query",
                    "stage": "generation",
                    "model_name": self.llm_config.model_name,
                    "query_count": 1,
                    "retrieved_count": len(documents),
                    "reference_count": len(references),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            return result
        except Exception:
            logger.exception(
                "RAG query failed",
                extra={
                    "event": "rag.query.failed",
                    "operation": "query",
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            raise

    def close(self) -> None:
        if self.qdrant_client is not None:
            self.qdrant_client.close()
            self.qdrant_client = None
        self.llm = None


def create_rag(root: Path | str | None = None) -> RAG:
    """Create a RAG instance from the centralized project settings."""

    if root is None:
        try:
            resolved_root = project_root()
        except RuntimeError:
            resolved_root = Path.cwd().resolve()
    else:
        resolved_root = Path(root).expanduser().resolve()
    settings = get_settings(resolved_root / ".env")
    return RAG(
        settings.rag_config(resolved_root),
        settings.embedding_config(),
        settings.llm_config(),
    )


def ingest(root: Path | str | None = None) -> dict[str, Any]:
    """Convenience function for one complete ingestion run."""

    orchestrator = create_rag(root)
    try:
        return orchestrator.ingest()
    finally:
        orchestrator.close()


__all__ = ["RAGConfig", "RAG", "create_rag", "ingest"]
