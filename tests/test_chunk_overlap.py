from langchain_core.documents import Document

from helpers.config import ChunkingConfig
from arabic_legal_qa.rag.chunking import chunk_documents, token_count


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=True):
        return types.SimpleNamespace(ids=list(text))


import types


def test_chunking_applies_overlap_without_exceeding_embedding_budget() -> None:
    tokenizer = CharacterTokenizer()
    source = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_documents(
        [Document(page_content=source, metadata={"article_number": 1, "language": "en"})],
        tokenizer,
        ChunkingConfig(embedding_limit=12, chunk_tokens=10, overlap=2),
        corpus_hash="corpus",
    )

    assert len(chunks) > 1
    assert all(token_count(tokenizer, chunk.page_content) <= 10 for chunk in chunks)
    assert chunks[1].page_content[:2] == chunks[0].page_content[-2:]
    assert all(chunk.metadata["article_number"] == 1 for chunk in chunks)
