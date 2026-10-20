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

from arabic_legal_qa.rag.chunking import chunk_documents
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
from helpers.config import (
    EmbeddingConfig,
    LLMConfig,
    QdrantConfig,
    RAGConfig,
    get_settings,
    project_root,
)
from helpers.langfuse import observation, update_observation

logger = logging.getLogger(__name__)


ANSWER_PROMPT_EN = """You are a careful legal information assistant.
Answer the user's question in English using only the retrieved Egyptian Civil Code evidence below.
If the evidence is insufficient, say so clearly and do not guess.
Do not invent article numbers or legal rules. Cite supporting articles inline
using the exact form [Article N]. Treat repealed articles as repealed.

Retrieved evidence:
{context}

User question:
{question}
"""
ANSWER_PROMPT_AR = """أنت مساعد قانوني دقيق.
أجب عن سؤال المستخدم باللغة العربية، مستندًا فقط إلى أدلة القانون المدني المصري المسترجعة أدناه.
إذا كانت الأدلة غير كافية، فاذكر ذلك بوضوح ولا تخمّن.
لا تخترع أرقام مواد أو قواعد قانونية. استشهد بالمواد المؤيدة داخل الإجابة
بالصيغة الدقيقة [Article N]. وتعامل مع المواد الملغاة على أنها ملغاة.

الأدلة المسترجعة:
{context}

سؤال المستخدم:
{question}
"""
ANSWER_PROMPT_VERSION = "legal-grounded-bilingual-v2"


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
    embedding_config: EmbeddingConfig = field(
        default_factory=lambda: get_settings().embedding_config()
    )
    llm_config: LLMConfig = field(default_factory=lambda: get_settings().llm_config())
    langfuse_client: Any | None = field(default=None, repr=False)
    langfuse_capture_content: bool = False
    langfuse_required: bool = False
    embedding_bundle: EmbeddingBundle | None = field(default=None, init=False)
    llm: Any | None = field(default=None, init=False)
    qdrant_client: Any | None = field(default=None, init=False)
    qdrant_config: QdrantConfig | None = field(default=None, init=False)
    registry_config_version: str | None = field(default=None, init=False)

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
            chunking_config = replace(
                get_settings().chunking_config(),
                embedding_limit=bundle.config.embedding_limit,
                chunk_tokens=self.config.chunk_tokens,
                overlap=self.config.chunk_overlap,
            )
            result = build_index(
                documents=chunks,
                embedding_bundle=bundle,
                config=qdrant_config,
                corpus_hash=corpus_hash,
                chunking_config=chunking_config,
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
            # Resolve the embedding revision before deciding whether vectors
            # can be reused. The config may omit a revision, while the index
            # manifest records the resolved immutable commit SHA.
            active_embedding = self._load_embeddings().config
            if manifest.get("complete") is not True:
                return False
            # A pre-ONNX index has vectors from a different inference path.
            # Rebuild it once so the manifest always describes the active
            # backend and the collection is internally consistent.
            if manifest.get("embedding_backend") != "onnxruntime-cpu":
                return False
            if manifest.get("embedding_model") != active_embedding.model_name:
                return False
            if manifest.get("embedding_revision") != active_embedding.model_revision:
                return False
            if (
                manifest.get("onnx_model_filename")
                != active_embedding.onnx_model_filename
            ):
                return False
            if manifest.get("chunk_tokens") != self.config.chunk_tokens:
                return False
            if manifest.get("chunk_overlap") != self.config.chunk_overlap:
                return False
            client = open_index(self._qdrant_config())
            count = client.count(
                self._qdrant_config().collection_name, exact=True
            ).count
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

    def retrieve(
        self, question: str, k: int | None = None, prefetch_k: int | None = None
    ) -> list[Document]:
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
            references.append(
                {
                    "article_number": metadata.get("article_number"),
                    "language": metadata.get("language"),
                    "citation": metadata.get("citation"),
                    "source_page": metadata.get("source_page"),
                    "is_repealed": metadata.get("is_repealed", False),
                }
            )
        return references

    def query(
        self,
        question: str,
        k: int | None = None,
        prefetch_k: int | None = None,
        min_evidence: int | None = None,
        *,
        include_trace: bool = False,
        trace_id: str | None = None,
        trace_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrieve using the user's question verbatim and answer with evidence."""
        language = _preferred_language(question)
        trace_options: dict[str, Any] = {}
        if trace_id is not None:
            trace_options["trace_context"] = {"trace_id": trace_id}
        metadata = {
            "question_language": language,
            "prompt_version": ANSWER_PROMPT_VERSION,
            **(trace_metadata or {}),
        }
        with observation(
            self.langfuse_client,
            required=self.langfuse_required,
            as_type="span",
            name="rag.query",
            input={"question": question} if self.langfuse_capture_content else None,
            metadata=metadata,
            **trace_options,
        ) as trace:
            result = self._query_impl(
                question,
                k=k,
                prefetch_k=prefetch_k,
                min_evidence=min_evidence,
                include_trace=include_trace,
                trace_client=self.langfuse_client,
            )
            if include_trace:
                active_trace_id = trace_id
                if active_trace_id is None and self.langfuse_client is not None:
                    try:
                        active_trace_id = self.langfuse_client.get_current_trace_id()
                    except Exception:
                        active_trace_id = None
                result["trace_id"] = active_trace_id
            trace_output: dict[str, Any] = {
                "evidence_count": result["evidence_count"],
                "sources": [
                    ref.get("citation")
                    for ref in result.get("references", [])
                    if ref.get("citation")
                ],
            }
            if self.langfuse_capture_content:
                trace_output["answer"] = result["answer"]
            update_observation(trace, output=trace_output)
            return result

    def _query_impl(
        self,
        question: str,
        k: int | None = None,
        prefetch_k: int | None = None,
        min_evidence: int | None = None,
        *,
        include_trace: bool = False,
        trace_client: Any | None = None,
    ) -> dict[str, Any]:
        """Execute retrieval and generation, optionally recording child spans."""
        k = self.config.retrieval_k if k is None else k
        prefetch_k = self.config.prefetch_k if prefetch_k is None else prefetch_k
        min_evidence = (
            self.config.min_evidence if min_evidence is None else min_evidence
        )
        started = time.perf_counter()
        logger.info(
            "RAG query started",
            extra={"event": "rag.query.started", "operation": "query"},
        )
        try:
            language = _preferred_language(question)
            with observation(
                trace_client,
                required=self.langfuse_required,
                as_type="retriever",
                name="qdrant.hybrid_search",
                input=(
                    {"question": question, "k": k}
                    if self.langfuse_capture_content
                    else {"k": k}
                ),
                metadata={"question_language": language},
            ) as retrieval_span:
                documents = self.retrieve(question, k=k, prefetch_k=prefetch_k)
            references = self._references(documents)
            update_observation(
                retrieval_span,
                output={
                    "retrieved_count": len(documents),
                    "articles": [
                        ref.get("citation") for ref in references if ref.get("citation")
                    ],
                    "languages": sorted(
                        {
                            str(doc.metadata.get("language"))
                            for doc in documents
                            if doc.metadata.get("language")
                        }
                    ),
                },
            )
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
                result = {
                    "answer": answer,
                    "references": references,
                    "evidence_count": 0,
                }
                if include_trace:
                    result["retrieved_contexts"] = [
                        doc.page_content for doc in documents
                    ]
                    result["retrieved_article_numbers"] = [
                        doc.metadata.get("article_number") for doc in documents
                    ]
                return result

            context_parts = []
            for index, document in enumerate(documents, start=1):
                metadata = document.metadata
                context_parts.append(
                    f"Evidence {index} | Article {metadata.get('article_number')} | "
                    f"Language {metadata.get('language')} | Repealed {metadata.get('is_repealed', False)}\n"
                    f"{document.page_content}"
                )
            answer_prompt = (
                ANSWER_PROMPT_AR if language == "Arabic" else ANSWER_PROMPT_EN
            )
            prompt = answer_prompt.format(
                context="\n\n".join(context_parts),
                question=question,
            )
            llm = self._load_llm()
            with observation(
                trace_client,
                required=self.langfuse_required,
                as_type="generation",
                name="answer-generation",
                model=self.llm_config.model_name,
                input=prompt if self.langfuse_capture_content else None,
                metadata={
                    "provider": self.llm_config.provider,
                    "prompt_version": ANSWER_PROMPT_VERSION,
                    "answer_language": language,
                },
            ) as generation:
                response = llm.invoke(prompt)
                answer = _message_text(response)
                update_observation(
                    generation,
                    output=(
                        answer if self.langfuse_capture_content else {"generated": True}
                    ),
                )
            result = {
                "answer": answer,
                "references": references,
                "evidence_count": len(documents),
            }
            if include_trace:
                result["retrieved_contexts"] = [doc.page_content for doc in documents]
                result["retrieved_article_numbers"] = [
                    doc.metadata.get("article_number") for doc in documents
                ]
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
    rag_config = settings.rag_config(resolved_root)
    embedding_config = settings.embedding_config()
    if settings.mlflow_tracking_uri:
        from helpers.mlflow_registry import load_registered_rag_config

        try:
            resolved = load_registered_rag_config(settings)
        except Exception:
            if settings.mlflow_config_required:
                raise
            logger.warning(
                "MLflow registry config unavailable; using centralized defaults",
                exc_info=True,
                extra={"event": "mlflow.registry.config_fallback"},
            )
            resolved = None
        if resolved is None and settings.mlflow_config_required:
            raise RuntimeError(
                "MLflow config is required but no registry alias was resolved"
            )
        if resolved is not None:
            registered, identity = resolved
            rag_config = replace(
                rag_config,
                chunk_tokens=registered.chunk_tokens,
                chunk_overlap=registered.chunk_overlap,
                retrieval_k=registered.retrieval_k,
                prefetch_k=registered.prefetch_k,
                min_evidence=registered.min_evidence,
            )
            embedding_config = replace(
                embedding_config,
                model_name=registered.embedding_model,
                model_revision=registered.embedding_revision,
            )
            logger.info(
                "Applying registered RAG configuration",
                extra={
                    "event": "mlflow.registry.config_applied",
                    "model_version": identity,
                },
            )
            rag = RAG(rag_config, embedding_config, settings.llm_config())
            rag.registry_config_version = identity
            return rag
    return RAG(rag_config, embedding_config, settings.llm_config())


def ingest(root: Path | str | None = None) -> dict[str, Any]:
    """Convenience function for one complete ingestion run."""

    orchestrator = create_rag(root)
    try:
        return orchestrator.ingest()
    finally:
        orchestrator.close()


__all__ = ["RAG", "RAGConfig", "create_rag", "ingest"]
