"""Command-line entry point for RAG ingestion and question answering."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from helpers.config import get_settings
from helpers.logging_config import configure_logging
from arabic_legal_qa.rag.orchestrator import create_rag


def main() -> None:
    parser = argparse.ArgumentParser(prog="arabic-legal-qa")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project data root (defaults to the current directory)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest", help="Build or replace the local Qdrant index")
    query_parser = commands.add_parser("query", help="Ask a question against the Civil Code")
    query_parser.add_argument("question", help="Question in Arabic or English")
    query_parser.add_argument("--k", type=int, default=None, help="Chunks retrieved per query")
    query_parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    configure_logging(get_settings().log_level)
    rag = create_rag(root=args.root)
    try:
        result = rag.ingest() if args.command == "ingest" else rag.query(
            args.question,
            k=args.k,
            max_queries=args.max_queries,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    finally:
        rag.close()
