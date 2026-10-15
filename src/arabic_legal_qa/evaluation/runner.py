"""Evaluate RAG chunking/retrieval candidates, log to MLflow, and promote winner."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from typing import Any

from helpers.config import (
    DEFAULT_MLFLOW_TRACKING_URI,
    RegisteredRAGConfig,
    get_settings,
    project_root,
)
from arabic_legal_qa.rag.orchestrator import ANSWER_PROMPT_VERSION, RAG
from arabic_legal_qa.rag.helper import stable_hash
from arabic_legal_qa.evaluation.registry import log_rag_config_model, register_logged_rag_config
from helpers.mlflow_registry import load_registered_rag_config
from arabic_legal_qa.rag.qdrant_db import prepare_retrieval

logger = logging.getLogger(__name__)
DEFAULT_CANDIDATES: tuple[tuple[int, int, int], ...] = (
    (256, 0, 5),
    (384, 0, 5),
    (480, 0, 5),
    (384, 32, 5),
    (480, 32, 5),
    (384, 32, 8),
)


def _load_questions(path: Path, split: str) -> list[dict[str, Any]]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    selected = [row for row in questions if row.get("split") == split]
    if not selected:
        raise ValueError(f"No evaluation questions found for split {split!r} in {path}")
    return selected


def _metric_number(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _evaluate_ragas(
    rows: list[dict[str, Any]], rag: RAG, *, judge_metrics: str = "core",
    judge_workers: int = 1,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Score core context precision and faithfulness, with optional extra judges.

    The no-reference metric set works with the current question set. When every
    example has a reviewed reference answer, reference-based RAGAS metrics are
    enabled automatically; article IDs are never treated as answer references.
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextRecall,
        ContextUtilization,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextPrecisionWithoutReference,
    )

    dataset = EvaluationDataset.from_list(rows)
    have_references = bool(rows) and all(row.get("reference") for row in rows)
    metric_set = [
        LLMContextPrecisionWithReference() if have_references
        else LLMContextPrecisionWithoutReference(),
        Faithfulness(),
    ]
    if judge_metrics == "all":
        # Groq's chat completion endpoint only accepts n=1; strictness=1
        # avoids the unsupported multi-generation request.
        metric_set.extend([ContextUtilization(), AnswerRelevancy(strictness=1)])
    if have_references:
        metric_set.extend([ContextRecall(), AnswerCorrectness()])
    ragas_result = evaluate(
        dataset=dataset,
        metrics=metric_set,
        llm=LangchainLLMWrapper(rag._load_llm()),
        embeddings=LangchainEmbeddingsWrapper(rag._load_embeddings().embedder),
        run_config=RunConfig(max_workers=judge_workers),
    )
    frame = ragas_result.to_pandas()
    records = json.loads(frame.to_json(orient="records", force_ascii=False))
    aggregate: dict[str, float] = {}
    for name in (
        "llm_context_precision_without_reference",
        "llm_context_precision_with_reference",
        "faithfulness",
        "context_utilization",
        "answer_relevancy",
        "context_recall",
        "answer_correctness",
    ):
        if name in frame:
            values = [float(value) for value in frame[name].dropna().tolist()]
            if values:
                aggregate[name] = mean(values)
    return aggregate, records


def _article_retrieval_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    hits: list[float] = []
    recalls: list[float] = []
    for row in rows:
        expected = {int(value) for value in row.get("expected_articles", [])}
        if not expected:
            continue
        retrieved = {int(value) for value in row.get("retrieved_article_numbers", []) if value is not None}
        hits.append(float(bool(expected & retrieved)))
        recalls.append(len(expected & retrieved) / len(expected))
    return {
        "article_hit_rate_at_k": mean(hits) if hits else 0.0,
        "article_recall_at_k": mean(recalls) if recalls else 0.0,
    }


def _quick_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return up to four questions balanced across Arabic and English."""
    selected: list[dict[str, Any]] = []
    for language in ("ar", "en"):
        selected.extend([row for row in questions if row.get("language") == language][:2])
    if len(selected) < min(4, len(questions)):
        selected_ids = {row.get("id") for row in selected}
        selected.extend(row for row in questions if row.get("id") not in selected_ids)
    return selected[:4]


def _cache_read(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _cache_write(path: Path, value: Any) -> None:
    """Atomically persist a completed stage for safe resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _cache_path(cache_root: Path, stage: str, key_payload: Any) -> Path:
    return cache_root / stage / f"{stable_hash(key_payload)}.json"


def _retrieval_only_rows(
    rag: RAG,
    questions: list[dict[str, Any]],
    *,
    retrieval_k: int,
    prefetch_k: int,
    cache_root: Path,
    corpus_hash: str,
    candidate: RegisteredRAGConfig,
    use_cache: bool,
) -> list[dict[str, Any]]:
    rows = []
    for question in questions:
        cache_file = _cache_path(cache_root, "retrieval", {
            "question": question["question"], "question_id": question.get("id"),
            "corpus_hash": corpus_hash, "candidate": asdict(candidate),
            "retrieval_k": retrieval_k, "prefetch_k": prefetch_k,
        })
        row = _cache_read(cache_file) if use_cache else None
        if row is None:
            documents = rag.retrieve(question["question"], k=retrieval_k, prefetch_k=prefetch_k)
            row = {
                "user_input": question["question"],
                "retrieved_contexts": [doc.page_content for doc in documents],
                "retrieved_article_numbers": [doc.metadata.get("article_number") for doc in documents],
                "question_id": question.get("id"), "language": question.get("language"),
                "expected_articles": question.get("expected_articles", []),
            }
            if use_cache:
                _cache_write(cache_file, row)
        rows.append(row)
    return rows


def _answer_rows(
    rag: RAG,
    questions: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    *,
    cache_root: Path,
    candidate: RegisteredRAGConfig,
    llm_identity: str,
    use_cache: bool,
) -> list[dict[str, Any]]:
    rows = []
    for question, retrieved in zip(questions, retrieval_rows, strict=True):
        cache_file = _cache_path(cache_root, "answers", {
            "question": question["question"], "contexts": retrieved["retrieved_contexts"],
            "candidate": asdict(candidate), "llm": llm_identity,
            "prompt_version": ANSWER_PROMPT_VERSION,
        })
        cached = _cache_read(cache_file) if use_cache else None
        if cached is None:
            output = rag.query(question["question"], k=candidate.retrieval_k,
                               prefetch_k=candidate.prefetch_k, include_trace=True)
            cached = {"response": output["answer"]}
            if use_cache:
                _cache_write(cache_file, cached)
        rows.append({
            "user_input": question["question"],
            "retrieved_contexts": retrieved["retrieved_contexts"],
            "response": cached["response"],
            **({"reference": question["reference_answer"]}
               if question.get("reference_answer") else {}),
            "language": question.get("language"),
            "expected_articles": question.get("expected_articles", []),
            "retrieved_article_numbers": retrieved["retrieved_article_numbers"],
            "question_id": question.get("id"),
        })
    return rows


def _score_retrieval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        expected = {int(value) for value in row.get("expected_articles", [])}
        retrieved = {int(value) for value in row.get("retrieved_article_numbers", []) if value is not None}
        scored.append({
            "question_id": row.get("question_id"), "language": row.get("language"),
            "expected_articles": sorted(expected),
            "retrieved_article_numbers": row.get("retrieved_article_numbers", []),
            "article_hit_at_k": float(bool(expected & retrieved)),
            "article_recall_at_k": len(expected & retrieved) / len(expected) if expected else None,
        })
    return scored


def _rank_retrieval(result: dict[str, Any]) -> tuple[float, float]:
    metrics = result["metrics"]
    return (_metric_number(metrics, "article_recall_at_k"),
            _metric_number(metrics, "article_hit_rate_at_k"))


def _language_metrics(
    rows: list[dict[str, Any]],
    evaluated_rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Log RAGAS and article retrieval scores separately for Arabic/English."""
    names = (
        "llm_context_precision_without_reference",
        "llm_context_precision_with_reference",
        "context_utilization",
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "answer_correctness",
        "article_hit_at_k",
        "article_recall_at_k",
    )
    grouped: dict[str, dict[str, list[float]]] = {}
    for source, scored in zip(rows, evaluated_rows, strict=True):
        language = source.get("language") or "unknown"
        bucket = grouped.setdefault(language, {})
        for name in names:
            value = scored.get(name)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                bucket.setdefault(name, []).append(number)
    return {
        f"{name}_{language}": mean(values)
        for language, metrics in grouped.items()
        for name, values in metrics.items()
        if values
    }


def evaluate_candidates(
    *,
    root: Path | None = None,
    split: str = "dev",
    promote: bool = True,
    candidate_pairs: tuple[tuple[int, int, int], ...] = DEFAULT_CANDIDATES,
    validate_champion: bool = False,
    mode: str = "full",
    finalists: int = 2,
    judge_metrics: str = "core",
    judge_workers: int = 1,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Screen retrieval for all candidates, then generate/judge only finalists.

    The held-out split remains validation-only. Retrieval/answer/judge stages
    are cached independently beneath ``data/evaluation/cache`` for resumability.
    """
    import mlflow

    if mode not in {"quick", "full"}:
        raise ValueError("mode must be 'quick' or 'full'")
    if finalists < 1:
        raise ValueError("finalists must be at least 1")
    if judge_metrics not in {"core", "all"}:
        raise ValueError("judge_metrics must be 'core' or 'all'")
    if judge_workers < 1:
        raise ValueError("judge_workers must be at least 1")
    if not candidate_pairs:
        raise ValueError("At least one candidate configuration is required")
    selected_root = (root or project_root()).expanduser().resolve()
    settings = get_settings(selected_root / ".env")
    tracking_uri = settings.mlflow_tracking_uri or DEFAULT_MLFLOW_TRACKING_URI
    if validate_champion:
        if split != "heldout":
            raise ValueError("Champion validation must use the heldout split")
        resolved = load_registered_rag_config(settings)
        if resolved is None:
            raise RuntimeError("No MLflow registry configuration is available to validate")
        selected_config, registry_identity = resolved
        candidate_pairs = ((
            selected_config.chunk_tokens,
            selected_config.chunk_overlap,
            selected_config.retrieval_k,
        ),)
        promote = False
    elif split != "dev":
        raise ValueError("Use --validate-champion to evaluate held-out data without tuning")
    else:
        registry_identity = None
    question_path = selected_root / "data/evaluation/foundation_questions.json"
    questions = _load_questions(question_path, split)
    if mode == "quick" and split == "dev":
        questions = _quick_questions(questions)
    cache_root = selected_root / "data/evaluation/cache"
    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment(settings.mlflow_experiment_name)
    except Exception as exc:
        raise RuntimeError(
            f"MLflow tracking request to {tracking_uri} failed while selecting "
            f"experiment {settings.mlflow_experiment_name!r}: {exc}. "
            "If the response is HTTP 403, check the MLflow server's allowed-host "
            "configuration and the Host in MLFLOW_TRACKING_URI. Start/recreate "
            "the service with `docker compose up -d --force-recreate mlflow`, "
            "then verify `curl -i http://127.0.0.1:5000/health`."
        ) from exc
    dataset_hash = stable_hash(questions)
    results: list[dict[str, Any]] = []
    rag_by_index: dict[tuple[int, int], RAG] = {}
    index_dirs: dict[tuple[int, int], tempfile.TemporaryDirectory[str]] = {}
    manifests: dict[tuple[int, int], dict[str, Any]] = {}
    try:
        # Build one isolated Qdrant collection per (chunk size, overlap), not
        # per candidate: candidates differing only by k share this index.
        for chunk_tokens, chunk_overlap, retrieval_k in candidate_pairs:
            candidate = RegisteredRAGConfig(
                chunk_tokens=chunk_tokens,
                chunk_overlap=chunk_overlap,
                embedding_model=(selected_config.embedding_model if validate_champion else settings.embedding_model),
                embedding_revision=(selected_config.embedding_revision if validate_champion else settings.embedding_revision),
                retrieval_k=(selected_config.retrieval_k if validate_champion else retrieval_k),
                prefetch_k=(selected_config.prefetch_k if validate_champion
                            else max(settings.prefetch_k, retrieval_k * 4)),
                min_evidence=(selected_config.min_evidence if validate_champion else settings.min_evidence),
            )
            index_key = (chunk_tokens, chunk_overlap)
            rag = rag_by_index.get(index_key)
            if rag is None:
                rag_config = replace(
                    settings.rag_config(selected_root),
                    chunk_tokens=chunk_tokens, chunk_overlap=chunk_overlap,
                    retrieval_k=candidate.retrieval_k, prefetch_k=candidate.prefetch_k,
                    min_evidence=candidate.min_evidence,
                )
                index_dir = tempfile.TemporaryDirectory(prefix="arabic-legal-rag-eval-")
                index_dirs[index_key] = index_dir
                rag_config = replace(rag_config, qdrant_path=Path(index_dir.name))
                embedding_config = replace(
                    settings.embedding_config(), model_name=candidate.embedding_model,
                    model_revision=candidate.embedding_revision,
                )
                rag = RAG(rag_config, embedding_config, settings.llm_config())
                # retrieval_client triggers ingestion when needed; unlike
                # initialize(), this does not load the chat model.
                rag.retrieval_client()
                rag._load_embeddings()
                prepare_retrieval(rag._qdrant_config())
                rag_by_index[index_key] = rag
                manifest_path = rag.config.qdrant_path / rag.config.qdrant_manifest_name
                manifests[index_key] = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = manifests[index_key]
            corpus_hash = manifest["corpus_hash"]
            embedding_revision = rag._load_embeddings().config.model_revision or "default"
            candidate = replace(candidate, embedding_revision=embedding_revision)
            run_name = (
                f"heldout-champion-{registry_identity.rsplit('#', 1)[-1]}"
                if validate_champion else f"chunk-{chunk_tokens}-overlap-{chunk_overlap}-k-{retrieval_k}"
            )
            retrieval_rows = _retrieval_only_rows(
                rag, questions, retrieval_k=candidate.retrieval_k,
                prefetch_k=candidate.prefetch_k, cache_root=cache_root,
                corpus_hash=corpus_hash, candidate=candidate, use_cache=use_cache,
            )
            scored_retrieval = _score_retrieval_rows(retrieval_rows)
            retrieval_metrics = _article_retrieval_metrics(retrieval_rows)
            retrieval_metrics.update(_language_metrics(retrieval_rows, scored_retrieval))
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_params({
                    "chunk_tokens": chunk_tokens, "chunk_overlap": chunk_overlap,
                    "embedding_model": candidate.embedding_model,
                    "embedding_revision": embedding_revision,
                    "retrieval_k": candidate.retrieval_k,
                    "prefetch_k": candidate.prefetch_k,
                    "prompt_version": ANSWER_PROMPT_VERSION,
                    "corpus_hash": corpus_hash, "evaluation_set_version": dataset_hash,
                    "split": split, "question_count": len(questions), "mode": mode,
                })
                mlflow.set_tags({
                    "framework": "ragas", "stage": "retrieval_screening",
                    "retrieval": "qdrant_dense_sparse_rrf",
                    "generator": rag.llm_config.model_name or rag.llm_config.provider,
                    "evaluation_set_version": dataset_hash, "corpus_hash": corpus_hash,
                    "prompt_version": ANSWER_PROMPT_VERSION,
                    "registry_source": registry_identity or "candidate_sweep",
                })
                mlflow.log_metrics(retrieval_metrics)
                mlflow.log_dict({
                    "candidate": asdict(candidate), "corpus_hash": corpus_hash,
                    "evaluation_set_version": dataset_hash,
                    "qdrant_manifest": manifest, "metrics": retrieval_metrics,
                    "per_question": scored_retrieval,
                }, "retrieval_screening.json")
                mlflow.log_dict(manifest, "qdrant_manifest.json")
                mlflow.log_dict(asdict(candidate), "rag_config.json")
                results.append({
                    "run_id": run.info.run_id, "config": candidate,
                    "metrics": retrieval_metrics, "model_uri": "",
                    "_index_key": index_key, "_retrieval_rows": retrieval_rows,
                })

        # Local deterministic scores choose finalists; only these spend Groq
        # calls for answer generation and LLM-as-judge metrics.
        finalists_results = sorted(results, key=_rank_retrieval, reverse=True)[:min(finalists, len(results))]
        for result in finalists_results:
            candidate = result["config"]
            rag = rag_by_index[result["_index_key"]]
            retrieval_rows = result["_retrieval_rows"]
            llm_identity = rag.llm_config.model_name or rag.llm_config.provider
            answer_rows = _answer_rows(
                rag, questions, retrieval_rows, cache_root=cache_root,
                candidate=candidate, llm_identity=llm_identity, use_cache=use_cache,
            )
            cache_key = _cache_path(cache_root, "ragas", {
                "question_set": dataset_hash, "candidate": asdict(candidate),
                "answer_rows": answer_rows, "llm": llm_identity,
                "prompt_version": ANSWER_PROMPT_VERSION,
                "judge_metrics": judge_metrics, "judge_workers": judge_workers,
            })
            cached_scores = _cache_read(cache_key) if use_cache else None
            if cached_scores is None:
                ragas_metrics, evaluated_rows = _evaluate_ragas(
                    answer_rows, rag, judge_metrics=judge_metrics,
                    judge_workers=judge_workers,
                )
                for question, retrieved, scored in zip(
                    questions, retrieval_rows, evaluated_rows, strict=True,
                ):
                    expected = {int(value) for value in question.get("expected_articles", [])}
                    got = {int(value) for value in retrieved["retrieved_article_numbers"] if value is not None}
                    scored.update({
                        "question_id": question.get("id"), "language": question.get("language"),
                        "expected_articles": sorted(expected),
                        "retrieved_article_numbers": retrieved["retrieved_article_numbers"],
                        "article_hit_at_k": float(bool(expected & got)),
                        "article_recall_at_k": len(expected & got) / len(expected) if expected else None,
                    })
                cached_scores = {"ragas_metrics": ragas_metrics, "per_question": evaluated_rows}
                if use_cache:
                    _cache_write(cache_key, cached_scores)
            evaluated_rows = cached_scores["per_question"]
            metrics = {
                **result["metrics"], **cached_scores["ragas_metrics"],
                **_language_metrics(answer_rows, evaluated_rows),
            }
            with mlflow.start_run(run_id=result["run_id"]):
                mlflow.set_tag("stage", "finalist_ragas_scored")
                mlflow.set_tag("judge_metrics", judge_metrics)
                mlflow.log_param("judge_workers", judge_workers)
                mlflow.log_metrics(metrics)
                mlflow.log_dict({
                    "candidate": asdict(candidate), "prompt_version": ANSWER_PROMPT_VERSION,
                    "evaluation_set_version": dataset_hash,
                    "metrics": metrics, "per_question": evaluated_rows,
                }, "generation_and_ragas.json")
                mlflow.log_dict({"judge_metrics": judge_metrics, "rows": answer_rows}, "answers_and_contexts.json")
                result["model_uri"] = log_rag_config_model(candidate) if not validate_champion else ""
            result["metrics"] = metrics
            result["finalist"] = True

        # Candidates not promoted to finalist retain their retrieval-only
        # scores. Select/promote only among generation-evaluated candidates.
        results_for_selection = finalists_results
    finally:
        for rag in rag_by_index.values():
            rag.close()
        for index_dir in index_dirs.values():
            index_dir.cleanup()

    def rank(result: dict[str, Any]) -> tuple[float, float, float]:
        metrics = result["metrics"]
        return (
            _metric_number(metrics, "faithfulness"),
            max(
                _metric_number(metrics, "llm_context_precision_without_reference"),
                _metric_number(metrics, "llm_context_precision_with_reference"),
            ),
            _metric_number(metrics, "answer_relevancy"),
            _metric_number(metrics, "article_recall_at_k"),
        )

    winner = max(results_for_selection, key=rank)
    faithful = _metric_number(winner["metrics"], "faithfulness")
    if validate_champion:
        logger.info(
            "Completed held-out evaluation of registered configuration; no promotion performed",
            extra={"event": "mlflow.registry.heldout_validation", "registry_source": registry_identity},
        )
    elif promote and faithful >= settings.mlflow_min_faithfulness:
        version = register_logged_rag_config(
            winner["config"],
            model_uri=winner["model_uri"],
            model_name=settings.mlflow_config_model,
            tracking_uri=tracking_uri,
            run_id=winner["run_id"],
            alias=settings.mlflow_config_alias,
        )
        logger.info(
            "Promoted passing RAG configuration",
            extra={
                "event": "mlflow.registry.config_promoted",
                "model_name": settings.mlflow_config_model,
                "model_version": version,
                "model_alias": settings.mlflow_config_alias,
                "faithfulness": faithful,
            },
        )
    else:
        logger.warning(
            "No config was promoted; winner did not meet the faithfulness gate or promotion was disabled",
            extra={"event": "mlflow.registry.promotion_skipped", "faithfulness": faithful},
        )
    return [{key: value for key, value in row.items() if not key.startswith("_")} for row in results]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--validate-champion", action="store_true",
                        help="Evaluate the current champion on held-out questions without promoting")
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--mode", choices=("quick", "full"), default="full",
                        help="quick uses up to four balanced dev questions; full uses the complete split")
    parser.add_argument("--finalists", type=int, default=2,
                        help="number of retrieval leaders to evaluate with generation/RAGAS")
    parser.add_argument("--judge-metrics", choices=("core", "all"), default="core",
                        help="core: faithfulness + context precision; all adds context utilization and answer relevance")
    parser.add_argument("--judge-workers", type=int, default=1,
                        help="maximum concurrent RAGAS judge tasks; lower values reduce API bursts, not total calls")
    parser.add_argument("--no-cache", action="store_true",
                        help="disable persisted retrieval, answer, and RAGAS stage caches")
    args = parser.parse_args()
    results = evaluate_candidates(
        root=args.root,
        split="heldout" if args.validate_champion else "dev",
        promote=not args.no_promote,
        validate_champion=args.validate_champion,
        mode=args.mode,
        finalists=args.finalists,
        judge_metrics=args.judge_metrics,
        judge_workers=args.judge_workers,
        use_cache=not args.no_cache,
    )
    print(json.dumps([
        {"run_id": row["run_id"], "config": asdict(row["config"]), "metrics": row["metrics"]}
        for row in results
    ], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
