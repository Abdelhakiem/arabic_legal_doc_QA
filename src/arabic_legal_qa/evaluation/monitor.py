"""Scheduled 50-question RAGAS monitoring with MLflow and Langfuse scores."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from arabic_legal_qa.evaluation.runner import _evaluate_ragas
from arabic_legal_qa.rag.helper import stable_hash
from arabic_legal_qa.rag.orchestrator import ANSWER_PROMPT_VERSION, RAG, create_rag
from arabic_legal_qa.rag.qdrant_db import prepare_retrieval
from helpers.config import DEFAULT_MLFLOW_TRACKING_URI, get_settings, project_root
from helpers.langfuse import create_langfuse_client, shutdown_langfuse
from helpers.logging_config import configure_logging
from helpers.metrics import push_ragas_metrics

logger = logging.getLogger(__name__)

# Metric names returned by the four reference-based RAGAS metrics.
RAGAS_METRICS = {
    "context_precision": "llm_context_precision_with_reference",
    "faithfulness": "faithfulness",
    "context_recall": "context_recall",
    "answer_correctness": "answer_correctness",
}
_STOP = threading.Event()


def _load_questions(path: Path) -> list[dict[str, Any]]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or len(questions) < 50:
        raise ValueError(f"RAGAS monitoring requires at least 50 questions: {path}")
    ids: set[str] = set()
    for row in questions:
        if not isinstance(row, dict):
            raise ValueError("Every monitoring question must be a JSON object")
        question_id = row.get("id")
        if not isinstance(question_id, str) or not question_id or question_id in ids:
            raise ValueError("Monitoring question ids must be non-empty and unique")
        ids.add(question_id)
        if row.get("language") not in {"ar", "en"}:
            raise ValueError(f"Question {question_id} must declare ar or en language")
        if not str(row.get("question", "")).strip():
            raise ValueError(f"Question {question_id} is empty")
        if not str(row.get("reference_answer", "")).strip():
            raise ValueError(f"Question {question_id} has no reference_answer")
    return questions


def _finite_metric(value: Any, name: str, question_id: str | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"RAGAS did not return {name} for {question_id}") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"RAGAS returned a non-finite {name} for {question_id}")
    return number


def _aggregate_scores(
    scored_rows: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Require and aggregate all four RAGAS measures overall and by language."""
    overall: dict[str, float] = {}
    by_language: dict[str, dict[str, float]] = {}
    for public_name, ragas_name in RAGAS_METRICS.items():
        values = [
            _finite_metric(row.get(ragas_name), public_name, row.get("question_id"))
            for row in scored_rows
        ]
        overall[public_name] = mean(values)
        for language in ("ar", "en"):
            language_values = [
                _finite_metric(row.get(ragas_name), public_name, row.get("question_id"))
                for row in scored_rows
                if row.get("language") == language
            ]
            if language_values:
                by_language.setdefault(public_name, {})[language] = mean(
                    language_values
                )
    return overall, by_language


def _find_previous_run(
    tracking_uri: str,
    experiment_id: str,
    *,
    dataset_hash: str,
    corpus_hash: str,
    config_hash: str,
) -> Any | None:
    """Find the latest successful run on the exact same dataset and corpus."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=100,
    )
    for run in runs:
        if run.info.status != "FINISHED":
            continue
        params = run.data.params
        if (
            params.get("evaluation_set_hash") == dataset_hash
            and params.get("corpus_hash") == corpus_hash
            and params.get("monitor_config_hash") == config_hash
            and all(name in run.data.metrics for name in RAGAS_METRICS)
        ):
            return run
    return None


def _load_manifest(rag: RAG) -> dict[str, Any]:
    path = rag.config.qdrant_path / rag.config.qdrant_manifest_name
    if not path.is_file():
        raise FileNotFoundError(f"Qdrant manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or not manifest.get("corpus_hash"):
        raise RuntimeError(f"Qdrant index is incomplete or has no corpus hash: {path}")
    return manifest


def _question_trace_id(run_id: str, question_id: str) -> str:
    """Use a fresh valid 32-hex Langfuse trace id for each run/question pair."""
    return uuid.uuid5(uuid.UUID(run_id), question_id).hex


def _attach_langfuse(rag: RAG, client: Any, *, capture_content: bool) -> None:
    """Require all monitor RAG spans to use its authenticated Langfuse client."""
    rag.langfuse_client = client
    rag.langfuse_capture_content = capture_content
    rag.langfuse_required = True


def _log_langfuse_score(
    client: Any,
    *,
    trace_id: str,
    value: float,
    question_id: str,
    run_id: str,
    dataset_hash: str,
) -> None:
    client.create_score(
        name="ragas_faithfulness",
        value=value,
        trace_id=trace_id,
        data_type="NUMERIC",
        comment="Reference-set RAGAS faithfulness score.",
        metadata={
            "question_id": question_id,
            "mlflow_run_id": run_id,
            "evaluation_set_hash": dataset_hash,
        },
    )


def run_monitor(
    root: Path | None = None,
    *,
    rag_factory: Callable[[Path], RAG] | None = None,
    scorer: Callable[..., tuple[dict[str, float], list[dict[str, Any]]]] | None = None,
) -> dict[str, Any]:
    """Run the full monitoring set once; never rebuild or promote the index."""
    import mlflow

    selected_root = (root or project_root()).expanduser().resolve()
    settings = get_settings(selected_root / ".env")
    configure_logging(settings.log_level)
    tracking_uri = settings.mlflow_tracking_uri or DEFAULT_MLFLOW_TRACKING_URI
    questions_path = selected_root / "data/evaluation/monitoring_questions.json"
    rag: RAG | None = None
    langfuse: Any | None = None
    try:
        questions = _load_questions(questions_path)
        dataset_hash = stable_hash(questions)
        if not settings.langfuse_enabled:
            raise RuntimeError(
                "RAGAS monitoring requires Langfuse tracing; set LANGFUSE_ENABLED=true "
                "and configure this project's public/secret API keys"
            )
        langfuse = create_langfuse_client(
            settings.model_copy(update={"langfuse_required": True})
        )
        if langfuse is None:
            raise RuntimeError(
                "Could not initialize Langfuse; no monitor traces were sent"
            )
        rag = (rag_factory or create_rag)(selected_root)
        _attach_langfuse(
            rag,
            langfuse,
            capture_content=settings.langfuse_capture_content,
        )
        if not rag._index_ready():
            raise RuntimeError(
                "The current Qdrant index is absent or incompatible. Monitoring is "
                "read-only; build/reindex it explicitly before running RAGAS."
            )
        rag.load_index()
        rag._load_embeddings()
        prepare_retrieval(rag._qdrant_config())
        rag._load_llm()
        manifest = _load_manifest(rag)
        corpus_hash = str(manifest["corpus_hash"])
        actual_model = rag.llm_config.model_name or rag.llm_config.provider
        embedding = rag._load_embeddings().config
        monitor_config_hash = stable_hash(
            {
                "embedding_model": embedding.model_name,
                "embedding_revision": embedding.model_revision,
                "chunk_tokens": rag.config.chunk_tokens,
                "chunk_overlap": rag.config.chunk_overlap,
                "retrieval_k": rag.config.retrieval_k,
                "prefetch_k": rag.config.prefetch_k,
                "generator_model": actual_model,
                "prompt_version": ANSWER_PROMPT_VERSION,
            }
        )

        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.set_experiment(settings.ragas_monitor_experiment_name)
        previous = _find_previous_run(
            tracking_uri,
            experiment.experiment_id,
            dataset_hash=dataset_hash,
            corpus_hash=corpus_hash,
            config_hash=monitor_config_hash,
        )
        evaluation_run_id: str | None = None
        timestamp = datetime.now(UTC).isoformat()
        run_name = f"ragas-monitor-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

        with mlflow.start_run(run_name=run_name) as run:
            evaluation_run_id = run.info.run_id
            mlflow.log_params(
                {
                    "question_count": len(questions),
                    "evaluation_set_hash": dataset_hash,
                    "corpus_hash": corpus_hash,
                    "monitor_config_hash": monitor_config_hash,
                    "embedding_model": embedding.model_name,
                    "embedding_revision": embedding.model_revision or "default",
                    "chunk_tokens": rag.config.chunk_tokens,
                    "chunk_overlap": rag.config.chunk_overlap,
                    "retrieval_k": rag.config.retrieval_k,
                    "prefetch_k": rag.config.prefetch_k,
                    "generator_model": actual_model,
                    "prompt_version": ANSWER_PROMPT_VERSION,
                    "faithfulness_threshold": settings.ragas_faithfulness_threshold,
                    "judge_workers": 1,
                }
            )
            mlflow.set_tags(
                {
                    "monitor_type": "scheduled_ragas",
                    "evaluation_set": questions_path.name,
                    "evaluation_set_hash": dataset_hash,
                    "corpus_hash": corpus_hash,
                    "qdrant_collection": rag.config.collection_name,
                    "registry_config_version": rag.registry_config_version
                    or "defaults",
                    "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
                    "prompt_version": ANSWER_PROMPT_VERSION,
                }
            )

            answer_rows: list[dict[str, Any]] = []
            trace_ids: dict[str, str] = {}
            for question in questions:
                question_id = question["id"]
                trace_id = _question_trace_id(evaluation_run_id, question_id)
                trace_ids[question_id] = trace_id
                result = rag.query(
                    question["question"],
                    include_trace=True,
                    trace_id=trace_id,
                    trace_metadata={
                        "source": "ragas_monitor",
                        "question_id": question_id,
                        "evaluation_run_id": evaluation_run_id,
                        "evaluation_set_hash": dataset_hash,
                    },
                )
                answer_rows.append(
                    {
                        "user_input": question["question"],
                        "retrieved_contexts": result.get("retrieved_contexts", []),
                        "response": result["answer"],
                        "reference": question["reference_answer"],
                        "question_id": question_id,
                        "language": question["language"],
                        "expected_articles": question.get("expected_articles", []),
                        "retrieved_article_numbers": result.get(
                            "retrieved_article_numbers", []
                        ),
                    }
                )

            # Ensure trace events arrive before their linked score events.
            langfuse.flush()
            ragas_summary, scored_rows = (scorer or _evaluate_ragas)(
                answer_rows,
                rag,
                judge_metrics="core",
                judge_workers=1,
            )
            if len(scored_rows) != len(questions):
                raise RuntimeError(
                    f"RAGAS returned {len(scored_rows)} rows for {len(questions)} questions"
                )

            # RAGAS outputs retain row order. Enrich with stable ids and attach
            # each faithfulness score to its originating Langfuse trace.
            for question, score_row in zip(questions, scored_rows, strict=True):
                score_row["question_id"] = question["id"]
                score_row["language"] = question["language"]
                score_row["trace_id"] = trace_ids[question["id"]]
            overall, by_language = _aggregate_scores(scored_rows)
            for question, score_row in zip(questions, scored_rows, strict=True):
                faithfulness = _finite_metric(
                    score_row.get(RAGAS_METRICS["faithfulness"]),
                    "faithfulness",
                    question["id"],
                )
                _log_langfuse_score(
                    langfuse,
                    trace_id=trace_ids[question["id"]],
                    value=faithfulness,
                    question_id=question["id"],
                    run_id=evaluation_run_id,
                    dataset_hash=dataset_hash,
                )

            retrieval_hits = []
            retrieval_recalls = []
            for question, score_row in zip(questions, scored_rows, strict=True):
                expected = {
                    int(value) for value in question.get("expected_articles", [])
                }
                retrieved = {
                    int(value)
                    for value in score_row.get("retrieved_article_numbers", [])
                    if value is not None
                }
                if expected:
                    retrieval_hits.append(float(bool(expected & retrieved)))
                    retrieval_recalls.append(len(expected & retrieved) / len(expected))
                score_row["article_hit_at_k"] = float(bool(expected & retrieved))
                score_row["article_recall_at_k"] = (
                    len(expected & retrieved) / len(expected) if expected else None
                )

            mlflow_metrics: dict[str, float] = dict(overall)
            for metric, language_values in by_language.items():
                for language, value in language_values.items():
                    mlflow_metrics[f"{metric}_{language}"] = value
            if retrieval_hits:
                mlflow_metrics["article_hit_rate_at_k"] = mean(retrieval_hits)
                mlflow_metrics["article_recall_at_k"] = mean(retrieval_recalls)

            deltas: dict[str, float] = {}
            if previous is not None:
                for metric in RAGAS_METRICS:
                    deltas[metric] = overall[metric] - float(
                        previous.data.metrics[metric]
                    )
                    mlflow_metrics[f"{metric}_delta_vs_previous"] = deltas[metric]
                mlflow.set_tag("previous_comparable_run_id", previous.info.run_id)

            faithfulness_alert = (
                overall["faithfulness"] < settings.ragas_faithfulness_threshold
            )
            drift_alert = (
                deltas.get("faithfulness", 0.0) <= -settings.ragas_drift_threshold
            )
            alerts = {
                "faithfulness_below_threshold": faithfulness_alert,
                "faithfulness_drift": drift_alert,
            }
            mlflow_metrics["faithfulness_alert"] = float(faithfulness_alert)
            mlflow_metrics["faithfulness_drift_alert"] = float(drift_alert)
            mlflow.log_metrics(mlflow_metrics)
            mlflow.set_tags(
                {
                    "quality_alert": "true" if faithfulness_alert else "false",
                    "drift_alert": "true" if drift_alert else "false",
                }
            )
            mlflow.log_dict(
                {
                    "evaluation_set_hash": dataset_hash,
                    "corpus_hash": corpus_hash,
                    "run_id": evaluation_run_id,
                    "created_at": timestamp,
                    "metrics": mlflow_metrics,
                    "ragas_reported_aggregate": ragas_summary,
                    "per_question": [
                        {
                            "question_id": row["question_id"],
                            "language": row["language"],
                            "trace_id": row["trace_id"],
                            "expected_articles": row.get("expected_articles", []),
                            "retrieved_article_numbers": row.get(
                                "retrieved_article_numbers", []
                            ),
                            "article_hit_at_k": row.get("article_hit_at_k"),
                            "article_recall_at_k": row.get("article_recall_at_k"),
                            **{
                                public_name: row.get(ragas_name)
                                for public_name, ragas_name in RAGAS_METRICS.items()
                            },
                        }
                        for row in scored_rows
                    ],
                },
                "monitoring_results.json",
            )
            mlflow.log_dict(manifest, "qdrant_manifest.json")
            mlflow.log_dict(overall, "ragas_aggregate_scores.json")

            if faithfulness_alert:
                logger.error(
                    "RAGAS faithfulness is below the configured threshold",
                    extra={
                        "event": "ragas.monitor.alert",
                        "alert": "faithfulness_below_threshold",
                        "faithfulness": overall["faithfulness"],
                        "threshold": settings.ragas_faithfulness_threshold,
                        "mlflow_run_id": evaluation_run_id,
                    },
                )
            if drift_alert:
                logger.error(
                    "RAGAS faithfulness drift exceeded the configured limit",
                    extra={
                        "event": "ragas.monitor.alert",
                        "alert": "faithfulness_drift",
                        "delta": deltas["faithfulness"],
                        "threshold": settings.ragas_drift_threshold,
                        "mlflow_run_id": evaluation_run_id,
                    },
                )

            push_scores = {
                metric: {"all": value, **by_language.get(metric, {})}
                for metric, value in overall.items()
            }
            push_ragas_metrics(
                settings.prometheus_pushgateway_url,
                scores=push_scores,
                deltas=deltas,
                faithfulness_threshold=settings.ragas_faithfulness_threshold,
                drift_threshold=settings.ragas_drift_threshold,
                success=True,
                alerts=alerts,
            )
            langfuse.flush()

        return {
            "run_id": evaluation_run_id,
            "question_count": len(questions),
            "evaluation_set_hash": dataset_hash,
            "corpus_hash": corpus_hash,
            "metrics": mlflow_metrics,
            "alerts": alerts,
        }
    except Exception:
        try:
            push_ragas_metrics(
                settings.prometheus_pushgateway_url,
                faithfulness_threshold=settings.ragas_faithfulness_threshold,
                drift_threshold=settings.ragas_drift_threshold,
                success=False,
            )
        except Exception:
            logger.exception("Could not publish failed monitor status to Pushgateway")
        logger.exception(
            "RAGAS monitor run failed",
            extra={"event": "ragas.monitor.failed", "operation": "monitor"},
        )
        raise
    finally:
        if rag is not None:
            rag.close()
        shutdown_langfuse(langfuse)


def run_periodically(root: Path | None, interval_hours: float) -> None:
    if interval_hours <= 0:
        raise ValueError("interval_hours must be greater than zero")
    _STOP.clear()
    signal.signal(signal.SIGTERM, lambda *_args: _STOP.set())
    signal.signal(signal.SIGINT, lambda *_args: _STOP.set())
    while not _STOP.is_set():
        try:
            run_monitor(root)
        except Exception:
            # Keep the service alive so the next scheduled run can recover.
            logger.exception("Scheduled RAGAS monitor attempt failed")
        _STOP.wait(interval_hours * 60 * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=None,
        help="repeat monitoring after each interval; omit to run once and exit",
    )
    args = parser.parse_args()
    if args.interval_hours is None:
        print(json.dumps(run_monitor(args.root), ensure_ascii=False, indent=2))
    else:
        run_periodically(args.root, args.interval_hours)


if __name__ == "__main__":
    main()
