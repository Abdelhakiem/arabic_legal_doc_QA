"""Application orchestrator for ingestion and inference.

The notebook is an executable experiment; this module is the reusable
pipeline boundary. It keeps the same validated loader, E5-small embedding
contract, article-aware chunking, and local Qdrant hybrid index.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from arabic_legal_qa.rag.config import EmbeddingConfig, project_root
from arabic_legal_qa.rag.helper import stable_hash
from arabic_legal_qa.rag.llms import (
    EmbeddingBundle,
    LLMConfig,
    load_embedding_bundle,
    load_llm,
)
from arabic_legal_qa.rag.pdf_loader import EgyptianCivilCodeLoader
from arabic_legal_qa.rag.qdrant_db import (
    QdrantConfig,
    build_index,
    hybrid_search,
    open_index,
)
from arabic_legal_qa.rag.chunking import ChunkingConfig, chunk_documents


QUERY_EXPANSION_PROMPT = """You expand a legal retrieval query for an Arabic/English civil-code corpus.
The preferred user language is: {language}.
Rewrite the question into up to three complementary search queries.
Keep every query in the preferred user language. Preserve article numbers,
legal terms, and named concepts. Return only one query per line, with no
numbering, explanation, translation, or markdown.

User question:
{question}
"""


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


@dataclass(frozen=True)
class RAGConfig:
    """Filesystem and model contract shared by ingestion and inference."""

    root: Path
    pdf_path: Path | None = None
    processed_dir: Path | None = None
    model_cache_dir: Path | None = None
    qdrant_path: Path | None = None
    collection_name: str = "egyptian_civil_code"
    chunk_tokens: int = 480
    exact_search: bool = True

    def __post_init__(self) -> None:
        root = Path(self.root).resolve()
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "pdf_path", Path(self.pdf_path) if self.pdf_path else root / "data/raw/egyptian_civil_law.pdf")
        object.__setattr__(self, "processed_dir", Path(self.processed_dir) if self.processed_dir else root / "data/processed")
        object.__setattr__(self, "model_cache_dir", Path(self.model_cache_dir) if self.model_cache_dir else root / "data/model_cache")
        object.__setattr__(self, "qdrant_path", Path(self.qdrant_path) if self.qdrant_path else root / "data/vector_store/qdrant")


@dataclass
class RAG:
    """Own ingestion state and provide retrieval from the persisted index."""

    config: RAGConfig
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm_config: LLMConfig = field(default_factory=LLMConfig)
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
            self.qdrant_config = QdrantConfig(
                path=self.config.qdrant_path,
                collection_name=self.config.collection_name,
                exact_search=self.config.exact_search,
            )
        return self.qdrant_config

    def _load_llm(self) -> Any:
        if self.llm is None:
            self.llm = load_llm(self.llm_config)
        return self.llm

    def ingest(self) -> dict[str, Any]:
        """Load, chunk, embed, and replace the stable Qdrant collection."""

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
            config=ChunkingConfig(
                embedding_limit=bundle.config.embedding_limit,
                chunk_tokens=self.config.chunk_tokens,
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
        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "corpus_hash": corpus_hash,
            "embedding_model": bundle.config.model_name,
            "collection": qdrant_config.collection_name,
            "indexed": result["count"],
            "report": loader.report,
        }

    def _index_ready(self) -> bool:
        """Return whether a completed local index is available."""

        manifest_path = self.config.qdrant_path / "manifest.json"
        if not manifest_path.exists():
            return False
        try:
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("complete") is not True:
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

    def retrieve(self, question: str, k: int = 5, prefetch_k: int = 24) -> list[Document]:
        """Run native dense+sparse hybrid retrieval for one question."""

        if not question.strip():
            raise ValueError("Question must not be empty")
        return hybrid_search(
            query=question,
            embedding_bundle=self._load_embeddings(),
            client=self.retrieval_client(),
            config=self._qdrant_config(),
            k=k,
            prefetch_k=prefetch_k,
        )

    def expand_query(self, question: str, max_queries: int = 3) -> list[str]:
        """Generate same-language retrieval variants with the configured LLM."""

        if not question.strip():
            raise ValueError("Question must not be empty")
        language = _preferred_language(question)
        response = self._load_llm().invoke(
            QUERY_EXPANSION_PROMPT.format(language=language, question=question)
        )
        queries: list[str] = []
        for line in _message_text(response).splitlines():
            cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if cleaned and cleaned not in queries:
                queries.append(cleaned)
        return [question, *queries[:max_queries]]

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

    def answer(self, question: str, k: int = 5, prefetch_k: int = 24,
               max_queries: int = 3, min_evidence: int = 1) -> dict[str, Any]:
        """Expand, retrieve, and answer with grounded references."""

        queries = self.expand_query(question, max_queries=max_queries)
        documents: list[Document] = []
        seen: set[str] = set()
        for query in queries:
            for document in self.retrieve(query, k=k, prefetch_k=prefetch_k):
                identity = str(document.metadata.get("chunk_id", document.page_content))
                if identity not in seen:
                    seen.add(identity)
                    documents.append(document)

        references = self._references(documents)
        language = _preferred_language(question)
        if len(documents) < min_evidence:
            answer = (
                "لا توجد أدلة كافية في النص المسترجع للإجابة عن هذا السؤال."
                if language == "Arabic"
                else "There is insufficient evidence in the retrieved text to answer this question."
            )
            return {"answer": answer, "references": references, "queries": queries}

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
        return {
            "answer": _message_text(response),
            "references": references,
            "queries": queries,
            "evidence_count": len(documents),
        }

    def close(self) -> None:
        if self.qdrant_client is not None:
            self.qdrant_client.close()
            self.qdrant_client = None
        self.llm = None


def create_rag(root: Path | str | None = None) -> RAG:
    """Create a RAG instance using the project root and E5 revision."""

    resolved_root = project_root(Path(root) if root else None)
    embedding_config = EmbeddingConfig(
        model_name="intfloat/multilingual-e5-small",
        model_revision=os.getenv("E5_SMALL_REVISION") or None,
    )
    llm_config = LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "ollama"),
        model_name=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    return RAG(RAGConfig(root=resolved_root), embedding_config, llm_config)


def ingest(root: Path | str | None = None) -> dict[str, Any]:
    """Convenience function for one complete ingestion run."""

    orchestrator = create_rag(root)
    try:
        return orchestrator.ingest()
    finally:
        orchestrator.close()


__all__ = ["RAGConfig", "RAG", "create_rag", "ingest"]
