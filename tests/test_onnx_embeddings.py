"""Small contract tests for direct ONNX Runtime embedding inference."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from helpers.config import EmbeddingConfig
from arabic_legal_qa.rag.chunking import token_count
from arabic_legal_qa.rag.llms import (
    OnnxEmbeddings,
    PrefixedEmbeddings,
    _load_embedding_bundle_cached,
)


class FakeTokenizer:
    def token_to_id(self, token: str) -> int | None:
        return 0 if token == "<pad>" else None

    def encode(self, text: str, add_special_tokens: bool = True):
        del add_special_tokens
        # One deterministic token per whitespace-separated term.
        return SimpleNamespace(ids=[int(word) for word in text.split()])


class FakeSession:
    def get_inputs(self):
        return [
            SimpleNamespace(name="input_ids", type="tensor(int64)"),
            SimpleNamespace(name="attention_mask", type="tensor(int64)"),
        ]

    def run(self, _outputs, feeds):
        token_ids = feeds["input_ids"].astype(np.float32)
        # Include padding in the output deliberately; the attention mask must
        # exclude it from mean pooling.
        return [np.stack((token_ids, token_ids * 2), axis=-1)]


def test_onnx_embeddings_mean_pool_mask_and_normalize():
    config = EmbeddingConfig(batch_size=2, embedding_limit=8)
    embeddings = OnnxEmbeddings(FakeSession(), FakeTokenizer(), config)

    vectors = embeddings.embed_documents(["1 3", "2"])

    assert len(vectors) == 2
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0)
    assert np.linalg.norm(vectors[1]) == pytest.approx(1.0)
    # The shorter sequence is padded, but its pooled vector still reflects
    # only token 2, not the pad token.
    assert vectors[1] == pytest.approx([1 / np.sqrt(5), 2 / np.sqrt(5)])


def test_onnx_embeddings_enforce_model_input_limit():
    embeddings = OnnxEmbeddings(
        FakeSession(), FakeTokenizer(), EmbeddingConfig(embedding_limit=2)
    )

    with pytest.raises(ValueError, match="exceeds"):
        embeddings.embed_query("1 2 3")


def test_prefixed_embedding_uses_query_and_passage_prefixes():
    class RecordingEmbeddings:
        def embed_documents(self, texts):
            return texts

        def embed_query(self, text):
            return text

    embeddings = PrefixedEmbeddings(RecordingEmbeddings(), "query: ", "passage: ")

    assert embeddings.embed_query("case") == "query: case"
    assert embeddings.embed_documents(["article"])[0] == "passage: article"


def test_token_count_accepts_rust_tokenizers_encoding():
    assert token_count(FakeTokenizer(), "1 2 3") == 3


def test_loader_downloads_onnx_artifacts_from_their_model_subdirectory(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "snapshot" / "onnx"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "model.onnx").touch()
    (artifact_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "config.json").write_text(
        json.dumps({"max_position_embeddings": 8}), encoding="utf-8"
    )

    requested = {}

    def fake_snapshot_download(**kwargs):
        requested.update(kwargs)
        return str(tmp_path / "snapshot")

    class FakeTokenizerLoader:
        @staticmethod
        def from_file(path):
            assert path == str(artifact_dir / "tokenizer.json")
            return FakeTokenizer()

    class FakeOrt:
        @staticmethod
        def InferenceSession(path, providers):
            assert path == str(artifact_dir / "model.onnx")
            assert providers == ["CPUExecutionProvider"]
            return FakeSession()

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setitem(__import__("sys").modules, "tokenizers", SimpleNamespace(Tokenizer=FakeTokenizerLoader))
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", FakeOrt)

    bundle = _load_embedding_bundle_cached(
        EmbeddingConfig(model_revision="test-revision", embedding_limit=8),
        tmp_path / "cache",
    )

    assert bundle.tokenizer.__class__ is FakeTokenizer
    assert requested["allow_patterns"] == [
        "onnx/model.onnx",
        "onnx/tokenizer.json",
        "onnx/config.json",
    ]
