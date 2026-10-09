"""Embedded Qdrant indexing and native dense+sparse retrieval."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document

from arabic_legal_qa.rag.helper import save_json, stable_hash

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QdrantConfig:
    path: Path
    collection_name: str = "egyptian_civil_code"
    vector_size: int = 384
    batch_size: int = 64
    sparse_model: str = "Qdrant/bm25"
    exact_search: bool = True
    manifest_name: str = "manifest.json"


_SPARSE_ENCODERS: dict[tuple[str, str | None], Any] = {}


def _sparse_encoder(model_name: str, cache_dir: Path | None = None):
    from fastembed import SparseTextEmbedding

    key = (model_name, str(cache_dir) if cache_dir else None)
    if key not in _SPARSE_ENCODERS:
        _SPARSE_ENCODERS[key] = SparseTextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    return _SPARSE_ENCODERS[key]


def _sparse_vector(encoded: Any):
    from qdrant_client import models

    return models.SparseVector(
        indices=encoded.indices.tolist(),
        values=encoded.values.tolist(),
    )


def _payload(document: Document) -> dict[str, Any]:
    return {
        "page_content": document.page_content,
        **{key: value for key, value in document.metadata.items()
           if isinstance(value, (str, int, float, bool)) or value is None},
    }


def _manifest(documents: list[Document], embedding_bundle: Any,
              config: QdrantConfig, corpus_hash: str) -> dict[str, Any]:
    model_config = embedding_bundle.config
    return {
        "schema": 1,
        "collection_name": config.collection_name,
        "embedding_model": model_config.model_name,
        "embedding_revision": model_config.model_revision,
        "vector_size": config.vector_size,
        "distance": "cosine",
        "exact_search": config.exact_search,
        "sparse_model": config.sparse_model,
        "corpus_hash": corpus_hash,
        "chunk_count": len(documents),
        "chunk_digest": stable_hash([
            {"id": d.metadata.get("chunk_id"), "text": d.page_content}
            for d in documents
        ]),
    }


def _build_index(documents: Iterable[Document], embedding_bundle: Any,
                 config: QdrantConfig, corpus_hash: str) -> dict[str, Any]:
    """Replace the stable collection and upsert dense+sparse vectors."""

    from qdrant_client import QdrantClient, models

    docs = list(documents)
    if not docs:
        raise ValueError("Cannot build a Qdrant index from zero documents")
    config.path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(config.path))
    if client.collection_exists(config.collection_name):
        client.delete_collection(config.collection_name)
    dense_config = models.VectorParams(
        size=config.vector_size,
        distance=models.Distance.COSINE,
        hnsw_config=models.HnswConfigDiff(m=0) if config.exact_search else None,
    )
    client.create_collection(
        collection_name=config.collection_name,
        vectors_config={"dense": dense_config},
        sparse_vectors_config={"sparse": models.SparseVectorParams(
            index=models.SparseIndexParams(on_disk=False),
        )},
    )

    sparse_encoder = _sparse_encoder(config.sparse_model, config.path / "sparse_cache")
    for offset in range(0, len(docs), config.batch_size):
        batch = docs[offset:offset + config.batch_size]
        dense_vectors = embedding_bundle.embedder.embed_documents(
            [doc.page_content for doc in batch]
        )
        if any(len(vector) != config.vector_size for vector in dense_vectors):
            raise ValueError(f"Expected {config.vector_size}-dimensional dense vectors")
        sparse_vectors = list(sparse_encoder.embed([doc.page_content for doc in batch]))
        points = []
        for document, dense, sparse in zip(batch, dense_vectors, sparse_vectors):
            chunk_id = document.metadata.get("chunk_id")
            if not chunk_id:
                raise ValueError("Every document must contain a deterministic chunk_id")
            points.append(models.PointStruct(
                id=chunk_id,
                vector={"dense": dense, "sparse": _sparse_vector(sparse)},
                payload=_payload(document),
            ))
        client.upsert(collection_name=config.collection_name, points=points, wait=True)

    count = client.count(config.collection_name, exact=True).count
    if count != len(docs):
        raise RuntimeError(f"Qdrant count mismatch: expected {len(docs)}, found {count}")
    manifest = {**_manifest(docs, embedding_bundle, config, corpus_hash), "complete": True}
    save_json(config.path / config.manifest_name, manifest)
    return {"client": client, "manifest": manifest, "count": count}


def build_index(documents: Iterable[Document], embedding_bundle: Any,
                config: QdrantConfig, corpus_hash: str) -> dict[str, Any]:
    """Build the collection and emit a single lifecycle event per run."""

    docs = list(documents)
    started = time.perf_counter()
    logger.info(
        "Qdrant indexing started",
        extra={
            "event": "rag.index.started",
            "operation": "ingest",
            "stage": "vector_index",
            "collection_name": config.collection_name,
            "chunk_count": len(docs),
            "vector_size": config.vector_size,
            "model_name": embedding_bundle.config.model_name,
            "corpus_version": corpus_hash,
        },
    )
    try:
        result = _build_index(docs, embedding_bundle, config, corpus_hash)
    except Exception:
        logger.exception(
            "Qdrant indexing failed",
            extra={
                "event": "rag.index.failed",
                "operation": "ingest",
                "stage": "vector_index",
                "collection_name": config.collection_name,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        raise
    logger.info(
        "Qdrant indexing completed",
        extra={
            "event": "rag.index.completed",
            "operation": "ingest",
            "stage": "vector_index",
            "collection_name": config.collection_name,
            "chunk_count": result["count"],
            "vector_size": config.vector_size,
            "model_name": embedding_bundle.config.model_name,
            "corpus_version": corpus_hash,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )
    return result


def open_index(config: QdrantConfig):
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(config.path))
    if not client.collection_exists(config.collection_name):
        raise FileNotFoundError(f"Qdrant collection is missing: {config.collection_name}")
    return client


def hybrid_search(query: str, embedding_bundle: Any, client: Any,
                  config: QdrantConfig, k: int = 8, prefetch_k: int = 24,
                  query_filter: Any = None) -> list[Document]:
    """Retrieve with Qdrant's native reciprocal-rank dense+sparse fusion."""

    from qdrant_client import models

    dense = embedding_bundle.embedder.embed_query(query)
    sparse = next(_sparse_encoder(config.sparse_model, config.path / "sparse_cache").embed([query]))
    response = client.query_points(
        collection_name=config.collection_name,
        prefetch=[
            models.Prefetch(query=dense, using="dense", limit=prefetch_k,
                            filter=query_filter,
                            params=models.SearchParams(exact=config.exact_search)),
            models.Prefetch(query=_sparse_vector(sparse), using="sparse", limit=prefetch_k,
                            filter=query_filter),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=True,
    )
    documents = []
    for point in response.points:
        payload = dict(point.payload or {})
        documents.append(Document(
            page_content=payload.pop("page_content", ""),
            metadata=payload,
        ))
    return documents
