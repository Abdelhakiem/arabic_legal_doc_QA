"""Shared model loaders for RAG components.

Embedding models are expensive to initialize, so this module owns both local
disk caching and process-level reuse. Indexing and retrieval should import the
embedding bundle from here instead of constructing model objects themselves.
"""
from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
import time

import numpy as np
from langchain_core.embeddings import Embeddings
from helpers.config import EmbeddingConfig, LLMConfig, get_settings, model_cache_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingBundle:
    """Resolved embedding model, tokenizer, and immutable config."""

    embedder: object
    tokenizer: object
    config: EmbeddingConfig
    cache_dir: Path


def _adapter_available(module_name: str) -> bool:
    """Return whether the optional LangChain provider adapter is importable."""

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        # ``find_spec`` may raise for a test-injected module with no spec.
        return module_name in __import__("sys").modules


def resolve_llm_config(config: LLMConfig | None = None) -> LLMConfig:
    """Choose Groq when configured, otherwise use local Ollama.

    An explicit ``LLM_PROVIDER`` (or ``config.provider``) can pin a provider;
    the default ``auto`` policy prefers Groq when a token and adapter exist.
    This checks configuration and installed adapters, not remote API health.
    """

    selected = config or get_settings().llm_config()
    provider = (selected.provider or "auto").strip().lower()
    groq_token = selected.groq_token
    groq_model = selected.model_name or selected.groq_model
    ollama_model = selected.model_name or selected.ollama_model

    if provider == "auto":
        if groq_token and _adapter_available("langchain_groq"):
            return replace(
                selected,
                provider="groq",
                model_name=groq_model,
                groq_token=groq_token,
            )
        if groq_token:
            logger.warning(
                "Groq credentials are configured but the LangChain Groq adapter is unavailable; trying Ollama",
                extra={"event": "rag.llm.provider_fallback", "provider": "ollama"},
            )
        if _adapter_available("langchain_ollama"):
            return replace(selected, provider="ollama", model_name=ollama_model)
        raise RuntimeError(
            "No LLM provider is available. Configure GROQ_TOKEN and install "
            "langchain-groq, or install/configure langchain-ollama and an Ollama server."
        )

    if provider == "groq":
        if not groq_token:
            raise RuntimeError("Groq was selected but GROQ_TOKEN is not configured")
        if not _adapter_available("langchain_groq"):
            raise RuntimeError("Groq was selected but langchain-groq is not installed")
        return replace(selected, provider="groq", model_name=groq_model, groq_token=groq_token)

    if provider == "ollama":
        if not _adapter_available("langchain_ollama"):
            raise RuntimeError("Ollama was selected but langchain-ollama is not installed")
        return replace(selected, provider="ollama", model_name=ollama_model)

    if provider in {"google", "gemini", "google_genai"}:
        return replace(
            selected,
            provider="google",
            model_name=selected.model_name or selected.gemini_model,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER {provider!r}; use auto, groq, or ollama")


def load_llm(config: LLMConfig | None = None):
    """Resolve and load a cached LangChain chat model."""

    selected = config or get_settings().llm_config()
    return _load_llm_cached(resolve_llm_config(selected))


@lru_cache(maxsize=8)
def _load_llm_cached(config: LLMConfig):
    started = time.perf_counter()
    provider = config.provider.lower()
    logger.info(
        "Chat model initialization started",
        extra={
            "event": "rag.llm.initialization.started",
            "stage": "llm_initialization",
            "provider": provider,
            "model_name": config.model_name,
        },
    )
    if provider == "groq":
        from langchain_groq import ChatGroq

        model = ChatGroq(
            model=config.model_name,
            api_key=config.groq_token,
            temperature=config.temperature,
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        model = ChatOllama(
            model=config.model_name,
            base_url=config.base_url,
            temperature=config.temperature,
            num_ctx=config.num_ctx,
        )
    elif provider in {"google", "gemini", "google_genai"}:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
        )
    else:
        raise ValueError(f"Unsupported chat model provider: {config.provider}")
    logger.info(
        "Chat model initialized",
        extra={
            "event": "rag.llm.initialization.completed",
            "stage": "llm_initialization",
            "provider": provider,
            "model_name": config.model_name,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )
    return model


class PrefixedEmbeddings(Embeddings):
    """Add the prefixes expected by multilingual E5 for each task."""

    def __init__(self, base: Embeddings, query_prefix: str, passage_prefix: str):
        self.base = base
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base.embed_documents([self._prefix(text, self.passage_prefix) for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query(self._prefix(text, self.query_prefix))

    @staticmethod
    def _prefix(text: str, prefix: str) -> str:
        return text if not prefix or text.startswith(prefix) else prefix + text


class OnnxEmbeddings(Embeddings):
    """Mean-pool token embeddings from an ONNX encoder on CPU.

    The E5 model exports contextual token vectors, so this class applies the
    same attention-mask-aware mean pooling and optional L2 normalization used
    by the model's Sentence Transformers configuration.
    """

    def __init__(self, session: object, tokenizer: object, config: EmbeddingConfig):
        self.session = session
        self.tokenizer = tokenizer
        self.config = config
        self.pad_token_id = tokenizer.token_to_id("<pad>")
        if self.pad_token_id is None:
            raise ValueError("The embedding tokenizer does not define a <pad> token")
        self.input_specs = {item.name: item.type for item in session.get_inputs()}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), self.config.batch_size):
            batch = texts[offset : offset + self.config.batch_size]
            encodings = [self.tokenizer.encode(text, add_special_tokens=True).ids for text in batch]
            if any(not ids for ids in encodings):
                raise ValueError("Embedding input cannot tokenize to an empty sequence")
            if any(len(ids) > self.config.embedding_limit for ids in encodings):
                raise ValueError(
                    f"Embedding input exceeds the configured {self.config.embedding_limit}-token limit"
                )

            width = max(map(len, encodings))
            input_ids = np.full((len(batch), width), self.pad_token_id, dtype=np.int64)
            attention_mask = np.zeros((len(batch), width), dtype=np.int64)
            for row, ids in enumerate(encodings):
                input_ids[row, : len(ids)] = ids
                attention_mask[row, : len(ids)] = 1

            feeds: dict[str, np.ndarray] = {}
            for name, tensor_type in self.input_specs.items():
                if name == "input_ids":
                    value = input_ids
                elif name == "attention_mask":
                    value = attention_mask
                elif name == "token_type_ids":
                    value = np.zeros_like(input_ids)
                else:
                    raise ValueError(f"Unsupported required ONNX model input: {name}")
                dtype = np.int32 if "int32" in tensor_type else np.int64
                feeds[name] = value.astype(dtype, copy=False)

            output = np.asarray(self.session.run(None, feeds)[0])
            if output.ndim != 3 or output.shape[:2] != input_ids.shape:
                raise ValueError(
                    "Expected ONNX token embeddings shaped [batch, sequence, hidden], "
                    f"received {output.shape}"
                )
            mask = attention_mask.astype(output.dtype)[..., None]
            pooled = (output * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
            if self.config.normalize_embeddings:
                norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                pooled = pooled / np.maximum(norms, 1e-12)
            vectors.extend(pooled.astype(np.float32).tolist())
        return vectors


def default_model_cache_dir(project_root: Path | None = None) -> Path:
    """Return the project-local model cache directory."""

    return model_cache_dir(project_root)


def load_embedding_bundle(config: EmbeddingConfig | None = None, cache_dir: Path | str | None = None) -> EmbeddingBundle:
    """Load or reuse the embedding model and tokenizer.

    The first call downloads the ONNX encoder and tokenizer into
    `data/model_cache`; PyTorch and Transformers are not used. Later calls with
    the same resolved config and cache directory reuse the in-memory objects.
    """

    selected = config or get_settings().embedding_config()
    selected_cache = Path(cache_dir) if cache_dir is not None else default_model_cache_dir()
    selected_cache.mkdir(parents=True, exist_ok=True)
    revision = selected.model_revision or _resolve_model_revision(selected.model_name)
    resolved = replace(selected, model_revision=revision)
    return _load_embedding_bundle_cached(resolved, selected_cache.resolve())


def _resolve_model_revision(model_name: str) -> str:
    from huggingface_hub import model_info

    return model_info(model_name).sha


@lru_cache(maxsize=4)
def _load_embedding_bundle_cached(config: EmbeddingConfig, cache_dir: Path) -> EmbeddingBundle:
    started = time.perf_counter()
    logger.info(
        "Embedding model initialization started",
        extra={
            "event": "rag.embedding.initialization.started",
            "stage": "embedding_initialization",
            "model_name": config.model_name,
        },
    )
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    onnx_relative_path = Path(config.onnx_model_filename)
    if onnx_relative_path.is_absolute() or ".." in onnx_relative_path.parts:
        raise ValueError("ONNX model filename must be a relative path inside the model repository")
    artifact_subdir = onnx_relative_path.parent
    model_dir = Path(snapshot_download(
        repo_id=config.model_name,
        revision=config.model_revision,
        cache_dir=str(cache_dir),
        allow_patterns=[
            config.onnx_model_filename,
            (artifact_subdir / "tokenizer.json").as_posix(),
            (artifact_subdir / "config.json").as_posix(),
        ],
    ))
    model_path = model_dir / config.onnx_model_filename
    tokenizer_path = model_dir / artifact_subdir / "tokenizer.json"
    model_config_path = model_dir / artifact_subdir / "config.json"
    for required_path in (model_path, tokenizer_path, model_config_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required ONNX embedding artifact is missing: {required_path}")

    import json

    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    model_limit = model_config.get("max_position_embeddings")
    if model_limit is not None and config.embedding_limit > model_limit:
        raise ValueError("Configured embedding limit exceeds the tokenizer's supported input length")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    base_embedder = OnnxEmbeddings(session=session, tokenizer=tokenizer, config=config)
    embedder = PrefixedEmbeddings(
        base=base_embedder,
        query_prefix=config.query_prefix,
        passage_prefix=config.passage_prefix,
    )
    bundle = EmbeddingBundle(embedder=embedder, tokenizer=tokenizer, config=config, cache_dir=cache_dir)
    logger.info(
        "Embedding model initialized",
        extra={
            "event": "rag.embedding.initialization.completed",
            "stage": "embedding_initialization",
            "model_name": config.model_name,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )
    return bundle
