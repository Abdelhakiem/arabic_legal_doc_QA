"""Build the reproducible DVC-managed Qdrant index from canonical articles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arabic_legal_qa.rag.chunking import chunk_documents
from arabic_legal_qa.rag.helper import stable_hash
from arabic_legal_qa.rag.llms import load_embedding_bundle
from arabic_legal_qa.rag.pdf_loader import (
    article_to_documents,
    load_validated_articles,
)
from arabic_legal_qa.rag.qdrant_db import build_index
from helpers.config import Settings
from helpers.logging_config import configure_logging


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def build_tracked_index(root: Path) -> dict:
    """Index validated ``articles.json``; do not re-parse the source PDF here."""

    root = root.expanduser().resolve()
    # DVC must use committed defaults, not machine-local .env overrides.
    settings = Settings.model_construct()
    pdf_path = _resolve(root, settings.pdf_path)
    processed_dir = _resolve(root, settings.processed_dir)
    articles = load_validated_articles(
        processed_dir, pdf_path=pdf_path, settings=settings
    )

    status = json.loads(
        (processed_dir / settings.validation_status_filename).read_text(
            encoding="utf-8"
        )
    )
    report = json.loads(
        (processed_dir / settings.extraction_report_filename).read_text(
            encoding="utf-8"
        )
    )
    if report.get("source_hash") != status.get("source_hash"):
        raise ValueError(
            "Extraction report and validation receipt refer to different PDFs"
        )
    # JSON object keys are strings; the article metadata helper looks them up by int.
    report["articles"] = {
        int(number): details for number, details in report.get("articles", {}).items()
    }

    corpus_hash = stable_hash([article.model_dump() for article in articles])
    documents = [
        document
        for article in articles
        for document in article_to_documents(
            article,
            report,
            status["source_hash"],
            corpus_hash,
            pdf_path.name,
            "bilingual",
        )
    ]

    embedding_config = settings.embedding_config()
    embedding_bundle = load_embedding_bundle(
        config=embedding_config,
        cache_dir=_resolve(root, settings.model_cache_path),
    )
    chunking_config = settings.chunking_config()
    chunks = chunk_documents(
        documents=documents,
        tokenizer=embedding_bundle.tokenizer,
        config=chunking_config,
        corpus_hash=corpus_hash,
    )
    qdrant_config = settings.qdrant_config(root=root)
    result = build_index(
        documents=chunks,
        embedding_bundle=embedding_bundle,
        config=qdrant_config,
        corpus_hash=corpus_hash,
        chunking_config=chunking_config,
    )
    count = result["count"]
    result["client"].close()
    return {
        "articles": len(articles),
        "documents": len(documents),
        "chunks": len(chunks),
        "indexed": count,
        "corpus_hash": corpus_hash,
        "embedding_model": embedding_bundle.config.model_name,
        "embedding_revision": embedding_bundle.config.model_revision,
        "chunk_tokens": chunking_config.chunk_tokens,
        "chunk_overlap": chunking_config.overlap,
        "index_path": str(qdrant_config.path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the DVC-managed hybrid Qdrant index from validated articles.json"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    configure_logging(Settings.model_construct().log_level)
    result = build_tracked_index(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
