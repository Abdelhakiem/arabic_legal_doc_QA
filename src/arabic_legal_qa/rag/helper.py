"""Small, dependency-free helpers shared by RAG pipeline modules."""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path


def stable_hash(value: object) -> str:
    """Hash JSON deterministically, including non-ASCII legal text."""

    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path: str | Path, value: object) -> None:
    """Atomically write JSON and avoid partial artifacts on interruption."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def normalize_search(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", text)
    return text.translate(str.maketrans("أإآٱى٠١٢٣٤٥٦٧٨٩", "ااااي0123456789"))


def glyph_text(chars: list[dict]) -> str:
    """Reconstruct RTL PDF glyphs while preserving LTR numeric runs."""

    from pdfplumber.utils import cluster_objects

    lines = []
    for line in cluster_objects(chars, "top", 3):
        glyphs = sorted(line, key=lambda c: c["x0"], reverse=True)
        parts, i = [], 0
        while i < len(glyphs):
            char = glyphs[i]
            if re.fullmatch(r"[\d./:\-]+", char["text"]):
                run = [char]
                i += 1
                while i < len(glyphs) and re.fullmatch(r"[\d./:\-]+", glyphs[i]["text"]):
                    run.append(glyphs[i]); i += 1
                parts.append("".join(c["text"] for c in reversed(run)))
            else:
                parts.append(char["text"]); i += 1
        text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", "".join(parts))).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)
