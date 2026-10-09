"""Public RAG package API and component exports."""

__all__ = [
    "RAG",
    "RAGConfig",
    "create_rag",
    "ingest",
    "main",
    "Article",
    "EgyptianCivilCodeLoader",
    "article_to_document",
    "article_to_documents",
    "extract_to_json",
    "load_validated_articles",
]


def __getattr__(name):
    if name in {"RAG", "RAGConfig", "create_rag", "ingest"}:
        from arabic_legal_qa.rag import orchestrator

        return getattr(orchestrator, name)
    if name == "main":
        from arabic_legal_qa.rag.cli import main

        return main
    if name in __all__:
        from arabic_legal_qa.rag import pdf_loader

        return getattr(pdf_loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
