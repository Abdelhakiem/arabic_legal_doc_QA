"""Public RAG package API and component exports."""

__all__ = [
    "RAG",
    "Article",
    "EgyptianCivilCodeLoader",
    "RAGConfig",
    "article_to_document",
    "article_to_documents",
    "create_rag",
    "extract_to_json",
    "ingest",
    "load_validated_articles",
    "main",
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
