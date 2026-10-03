"""RAG components for the Arabic Legal Q&A project."""

__all__ = [
    "Article",
    "EgyptianCivilCodeLoader",
    "article_to_document",
    "article_to_documents",
    "extract_to_json",
    "load_validated_articles",
]


def __getattr__(name):
    if name in __all__:
        from arabic_legal_qa.rag import pdf_loader

        return getattr(pdf_loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
