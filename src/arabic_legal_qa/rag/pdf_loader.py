"""Source-specific, position-aware extractor for the bilingual Egyptian Civil Code PDF.

Candidates and diagnostics are always written. Canonical JSON is written only after the
complete 1–1149 article validation gate passes. Source corrections are explicit and
reported so the canonical corpus remains auditable.
"""
from __future__ import annotations
import argparse
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Iterator, Literal
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, model_validator
from helpers.config import Settings, get_settings
from arabic_legal_qa.rag.helper import file_hash, glyph_text, normalize_search, save_json, stable_hash

logger = logging.getLogger(__name__)

ARTICLE_1022_AR_CORRECTION = (
    "ما لم يوجد اتفاق على خلاف ذلك، يتحمل مالك العقار المرتفق تكاليف الأعمال اللازمة "
    "لاستعمال حق الارتفاق والمحافظة عليه.\n"
    "فإذا كان مالك العقار المرتفق به هو المكلف بإجراء هذه الأعمال على نفقته، كان له "
    "دائماً أن يتخلص من هذا التكليف بأن يتخلى عن العقار المرتفق به كله أو بعضه "
    "لمالك العقار المرتفق.\n"
    "وإذا كانت الأعمال نافعة أيضاً لمالك العقار المرتفق به، كانت نفقة الصيانة على "
    "الطرفين كل بنسبة ما يعود عليه من الفائدة."
)
SOURCE_CORRECTIONS = {
    1022: {
        "field": "text_ar",
        "reason": "Arabic Article 1022 is missing from page 147 and its Arabic paragraphs are attached to Article 1021.",
        "method": "Manual Arabic translation from the supplied English Article 1022 text.",
        "text": ARTICLE_1022_AR_CORRECTION,
    }
}


class Article(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    article_number: int
    book: str | None = None
    chapter: str | None = None
    section: str | None = None
    topic: str | None = None
    text_ar: str = Field(min_length=1)
    text_en: str | None = None
    is_repealed: bool
    source_page: int
    citation: str

    @model_validator(mode="after")
    def check_content(self):
        if not self.text_ar.strip() or not re.search(r"[\u0621-\u064a]", self.text_ar):
            raise ValueError("Arabic article text is required")
        if self.citation != f"Egyptian Civil Code, Article {self.article_number}":
            raise ValueError("Noncanonical citation")
        return self


ArticleLanguage = Literal["ar", "en", "bilingual"]


def extract_rows(pdf_path, settings: Settings | None = None):
    import pdfplumber
    settings = settings or get_settings()
    rows, issues = [], []
    table_settings = {
        "vertical_strategy": "explicit",
        "explicit_vertical_lines": [settings.pdf_table_left_x, settings.pdf_table_middle_x,
                                    settings.pdf_table_right_x],
        "snap_tolerance": settings.pdf_table_snap_tolerance,
        "intersection_tolerance": settings.pdf_table_intersection_tolerance,
    }
    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) != settings.pdf_page_count:
            raise ValueError(
                f"This extractor is calibrated for the configured {settings.pdf_page_count}-page edition"
            )
        for page_number, page in enumerate(pdf.pages, 1):
            tables = page.find_tables(table_settings)
            if len(tables) != 1:
                issues.append(f"Page {page_number}: expected exactly one bilingual table")
                continue
            for row in tables[0].rows:
                if len(row.cells) != 2 or any(c is None for c in row.cells):
                    issues.append(f"Page {page_number}: uncertain table boundary {row.bbox}")
                    continue
                left, right = (page.within_bbox(box) for box in row.cells)
                en = (left.extract_text() or "").strip()
                ar = glyph_text(right.chars)
                visible = [c for c in left.chars if c["text"].strip()]
                bold = bool(visible) and all("bold" in c["fontname"].lower() for c in visible)
                rows.append({"page": page_number, "en": en, "ar": ar, "bold": bold,
                             "boxes": row.cells})
            if page_number % settings.pdf_progress_interval_pages == 0:
                logger.debug(
                    "PDF extraction progress",
                    extra={
                        "event": "rag.pdf.extraction.progress",
                        "operation": "ingest",
                        "stage": "pdf_extraction",
                        "page_count": page_number,
                    },
                )
    return rows, issues


def heading_number(text, language):
    pattern = r"^Article\s*(\d+)\b" if language == "en" else r"^مادة\s*[()]*\s*(\d+)"
    match = re.match(pattern, text, re.I)
    return int(match[1]) if match else None


def parse_articles(rows, settings: Settings | None = None):
    settings = settings or get_settings()
    records, diagnostics, errors, warnings = {}, {}, [], []
    hierarchy = dict.fromkeys(("book", "chapter", "section", "topic"))
    current = None
    pending_heading = None
    body_started = False

    def start(number, row, ar, en, repealed=False):
        nonlocal current
        if number in records:
            errors.append(f"Duplicate article {number} at page {row['page']}")
            return
        records[number] = {"article_number": number, **hierarchy, "text_ar": ar, "text_en": en or None,
                           "is_repealed": repealed, "source_page": row["page"],
                           "citation": f"Egyptian Civil Code, Article {number}"}
        diagnostics[number] = {"pages": [row["page"]], "regions": [{"page": row["page"], "boxes": row["boxes"]}]}
        current = number

    for row in rows:
        en, ar = row["en"], row["ar"]
        if not body_started:
            if row["page"] == 1 and en.startswith("SECTION I"):
                body_started = True
            else:
                continue  # Opening enactment has its own numbering and is outside this corpus.
        if (row["page"] == settings.pdf_article_452_repair_page
                and en.startswith("rticle 452") and heading_number(ar, "ar") == 452):
            en = "A" + en
            warnings.append("Page 59: recognized printed 'rticle 452' using the matching Arabic heading")
        # Notices represent whole ranges, not missing article bodies.
        notice = re.search(r"Articles?\s*(\d+)\s*[-–]\s*(\d+).*repealed", en, re.I | re.S)
        if notice:
            lo, hi = int(notice[1]), int(notice[2])
            if (lo, hi) not in settings.repeal_ranges or not re.search(r"ألغيت|ملغاة", ar):
                errors.append(f"Unverified repeal range at page {row['page']}")
                continue
            for number in range(lo, hi + 1):
                start(number, row, ar, en, repealed=True)
                diagnostics[number]["repeal_range"] = [lo, hi]
            current = None
            continue
        number, arabic_number = heading_number(en, "en"), heading_number(ar, "ar")
        if number is not None:
            if arabic_number != number:
                message = f"Bilingual heading mismatch at page {row['page']}: EN {number}, AR {arabic_number}"
                if number in SOURCE_CORRECTIONS:
                    warnings.append(message + "; handled by explicit source correction")
                else:
                    errors.append(message)
            start(number, row, re.sub(r"^مادة[^\n]*(?:\n|$)", "", ar).strip(),
                  re.sub(r"^Article\s*\d+[^\n]*(?:\n|$)", "", en, flags=re.I).strip())
            pending_heading = None
            continue
        if arabic_number is not None:
            errors.append(f"Arabic-only article {arabic_number} at page {row['page']}")
            continue
        if not en and not ar:
            continue
        if row["bold"]:
            key = next((k for k, pattern in (("book", r"^BOOK\b"), ("chapter", r"^CHAPTER\b"),
                                              ("section", r"^SECTION\b")) if re.match(pattern, en, re.I)), None)
            if re.match(r"^(FIRST|SECOND) PART", en, re.I):
                current = None
                continue
            if key:
                hierarchy[key] = " ".join(en.split())
                keys = list(hierarchy)
                for child in keys[keys.index(key) + 1:]:
                    hierarchy[child] = None
                pending_heading = key if len(en.splitlines()) == 1 and len(en.split()) <= 2 else None
            elif pending_heading:
                hierarchy[pending_heading] += " " + " ".join(en.split())
                pending_heading = None
            else:
                hierarchy["topic"] = " ".join(en.split()) or ar
            current = None
            continue
        if current is None:
            errors.append(f"Unassigned continuation at page {row['page']}: {en[:60]!r}")
            continue
        record = records[current]
        for key, value in (("text_ar", ar), ("text_en", en)):
            if value:
                record[key] = ((record[key] or "") + "\n" + value).strip()
        if row["page"] not in diagnostics[current]["pages"]:
            diagnostics[current]["pages"].append(row["page"])
        diagnostics[current]["regions"].append({"page": row["page"], "boxes": row["boxes"]})

    if 1022 in records and not records[1022]["text_ar"].strip():
        records[1022]["text_ar"] = SOURCE_CORRECTIONS[1022]["text"]
        diagnostics[1022]["source_correction"] = SOURCE_CORRECTIONS[1022]
        warnings.append("Page 147: applied explicit Arabic source correction for Article 1022")
        if 1021 in records:
            records[1021]["text_ar"] = re.sub(
                r"\n\(?٢\(?[\s\S]*$",
                "",
                records[1021]["text_ar"],
            ).strip()
            diagnostics[1021]["source_correction"] = {
                "field": "text_ar",
                "reason": "Removed Arabic Article 1022 paragraphs that were attached to Article 1021 by the defective source layout.",
                "method": "Kept only the first Arabic paragraph, matching Article 1021 English text.",
            }
    articles = []
    for number in sorted(records):
        try:
            articles.append(Article.model_validate(records[number]))
        except ValueError as exc:
            errors.append(f"Article {number}: invalid required article fields ({type(exc).__name__})")
    errors.extend(validate_articles(articles, settings))
    for article in articles:
        if not article.text_en:
            warnings.append(f"No English text: {article.article_number}")
    return articles, {"errors": errors, "warnings": warnings, "articles": diagnostics,
                      "article_count": len(articles), "extractor_version": settings.extractor_version}


def validate_articles(articles, settings: Settings | None = None) -> list[str]:
    """Check article continuity, Arabic text, sane length, and repeal flags."""
    settings = settings or get_settings()
    errors: list[str] = []
    article_numbers = [int(_record_value(article, "article_number")) for article in articles]
    found = set(article_numbers)
    expected = settings.expected_articles
    if len(article_numbers) != len(found) or found != expected:
        errors.append(
            f"Article numbers are not contiguous: missing {sorted(expected - found)}; "
            f"unexpected {sorted(found - expected)}; duplicates "
            f"{sorted(number for number in found if article_numbers.count(number) > 1)}"
        )

    for article in articles:
        number = int(_record_value(article, "article_number"))
        if number not in expected:
            errors.append(f"Article number {number} is outside the configured corpus range")
        text_ar = _record_value(article, "text_ar")
        if not isinstance(text_ar, str) or not text_ar.strip():
            errors.append(f"Article {number} has empty Arabic text")
        elif len(text_ar) > settings.article_text_max_length:
            errors.append(
                f"Article {number} Arabic text exceeds {settings.article_text_max_length} characters"
            )
        text_en = _record_value(article, "text_en")
        if isinstance(text_en, str) and len(text_en) > settings.article_text_max_length:
            errors.append(
                f"Article {number} English text exceeds {settings.article_text_max_length} characters"
            )
        is_repealed = bool(_record_value(article, "is_repealed"))
        should_be_repealed = any(lo <= number <= hi for lo, hi in settings.repeal_ranges)
        if is_repealed != should_be_repealed:
            expected_state = "repealed" if should_be_repealed else "active"
            errors.append(f"Article {number} should be flagged as {expected_state}")
        page = _record_value(article, "source_page")
        if page is not None and not 1 <= int(page) <= settings.pdf_page_count:
            errors.append(f"Article {number} source page is outside the configured PDF page range")
    return errors


def _record_value(article, field: str):
    return article.get(field) if isinstance(article, dict) else getattr(article, field, None)


def require_valid_corpus(articles, report, settings: Settings | None = None):
    settings = settings or get_settings()
    errors = report.get("errors", []) or validate_articles(articles, settings)
    if errors:
        raise ValueError("Corpus validation failed; inspect extraction_report.json: " + "; ".join(errors[:8]))


def article_content(article: Article, language: Literal["ar", "en"]):
    """Return display text for a language-specific LangChain document."""
    if language == "ar":
        return article.text_ar
    if not article.text_en:
        raise ValueError(f"Article {article.article_number} has no English text")
    return article.text_en


def article_metadata(article: Article, report, source_name: str, language: Literal["ar", "en"],
                     chunk_id: str, chunk_position: int = 0):
    diagnostics = report.get("articles", {}).get(article.article_number, {})
    metadata = article.model_dump(exclude={"text_ar", "text_en"})
    metadata["book"] = metadata.get("book") or source_name
    return {
        **metadata,
        "article_number": article.article_number,
        "citation": article.citation,
        "source_page": article.source_page,
        "language": language,
        "chunk_id": chunk_id,
        "chunk_position": chunk_position,
        "has_source_correction": "source_correction" in diagnostics,
    }


def article_to_document(article: Article, report, source_hash, corpus_hash,
                        source_name: str, language: Literal["ar", "en"], chunk_position: int = 0):
    content = article_content(article, language)
    identity = stable_hash({
        "corpus": corpus_hash,
        "article": article.article_number,
        "language": language,
        "position": chunk_position,
        "text": content,
    })
    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
    return Document(
        page_content=content,
        metadata=article_metadata(article, report, source_name, language, chunk_id, chunk_position),
    )


def article_to_documents(article: Article, report, source_hash, corpus_hash,
                         source_name: str, language: ArticleLanguage = "bilingual"):
    languages = ("ar", "en") if language == "bilingual" else (language,)
    for selected_language in languages:
        if selected_language == "en" and not article.text_en:
            continue
        yield article_to_document(article, report, source_hash, corpus_hash, source_name, selected_language)


class EgyptianCivilCodeLoader(BaseLoader):
    """LangChain loader for validated Egyptian Civil Code article documents.

    The loader still runs the repository extraction gate. It writes review artifacts,
    validates the complete article set, and only yields LangChain `Document`s from a
    valid canonical corpus.
    """

    def __init__(self, pdf_path, output_dir, language: ArticleLanguage = "bilingual",
                 settings: Settings | None = None):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.language = language
        self.settings = settings or get_settings()
        self.articles: list[Article] | None = None
        self.report: dict | None = None
        self.source_hash: str | None = None
        self.corpus_hash: str | None = None

    def load_articles(self):
        articles, report = extract_to_json(self.pdf_path, self.output_dir, self.settings)
        require_valid_corpus(articles, report, self.settings)
        self.articles = articles
        self.report = report
        self.source_hash = report["source_hash"]
        self.corpus_hash = stable_hash([article.model_dump() for article in articles])
        return articles

    def lazy_load(self) -> Iterator[Document]:
        articles = self.load_articles()
        assert self.report is not None
        assert self.source_hash is not None
        assert self.corpus_hash is not None
        for article in articles:
            yield from article_to_documents(
                article,
                self.report,
                self.source_hash,
                self.corpus_hash,
                self.pdf_path.name,
                self.language,
            )


def extract_to_json(pdf_path, output_dir, settings: Settings | None = None):
    """Write reviewable candidates/report, and canonical articles only on full validation."""
    pdf_path, output_dir = Path(pdf_path), Path(output_dir)
    settings = settings or get_settings()
    started = time.perf_counter()
    logger.info(
        "PDF corpus extraction started",
        extra={"event": "rag.pdf.extraction.started", "operation": "ingest", "stage": "pdf_extraction"},
    )
    try:
        source_hash = file_hash(pdf_path)
        cache_path = output_dir / settings.extraction_rows_filename
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        cache_hit = (
            cache.get("source_hash") == source_hash
            and cache.get("extractor_version") == settings.extractor_version
        )
        if cache_hit:
            rows, layout_errors = cache["rows"], cache["layout_errors"]
        else:
            rows, layout_errors = extract_rows(pdf_path, settings)
        articles, report = parse_articles(rows, settings)
        report["source_hash"] = source_hash
        report["errors"] = layout_errors + report["errors"]
        report["valid"] = not report["errors"]
        save_json(output_dir / settings.extraction_rows_filename, {"source_hash": source_hash,
                  "extractor_version": settings.extractor_version, "rows": rows,
                  "layout_errors": layout_errors})
        save_json(output_dir / settings.article_candidates_filename, [a.model_dump() for a in articles])
        save_json(output_dir / settings.extraction_report_filename, report)
        if report["valid"]:
            require_valid_corpus(articles, report, settings)
            save_json(output_dir / settings.canonical_articles_filename, [a.model_dump() for a in articles])
        save_json(output_dir / settings.validation_status_filename, {"valid": report["valid"],
                  "source_hash": source_hash,
                  "corpus_hash": stable_hash([a.model_dump() for a in articles]) if report["valid"] else None,
                  "errors": report["errors"]})
        log_method = logger.info if report["valid"] else logger.error
        log_method(
            "PDF corpus extraction completed" if report["valid"] else "PDF corpus validation failed",
            extra={
                "event": "rag.pdf.extraction.completed" if report["valid"] else "rag.pdf.extraction.invalid",
                "operation": "ingest",
                "stage": "validation",
                "corpus_version": stable_hash([a.model_dump() for a in articles]) if report["valid"] else None,
                "article_count": len(articles),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "cache_hit": cache_hit,
                "error_count": len(report["errors"]),
                "warning_count": len(report["warnings"]),
            },
        )
        return articles, report
    except Exception:
        logger.exception(
            "PDF corpus extraction failed",
            extra={
                "event": "rag.pdf.extraction.failed",
                "operation": "ingest",
                "stage": "pdf_extraction",
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        raise


def load_validated_articles(output_dir, pdf_path=None, settings: Settings | None = None):
    """Read only a canonical corpus whose validation receipt matches its content."""
    output_dir = Path(output_dir)
    settings = settings or get_settings()
    status = json.loads((output_dir / settings.validation_status_filename).read_text(encoding="utf-8"))
    if not status.get("valid"):
        raise ValueError("Corpus validation is incomplete; canonical articles are unavailable")
    if pdf_path is not None and file_hash(pdf_path) != status.get("source_hash"):
        raise ValueError("Canonical corpus was generated from a different PDF")
    articles = [Article.model_validate(row) for row in json.loads(
        (output_dir / settings.canonical_articles_filename).read_text(encoding="utf-8"))]
    if stable_hash([a.model_dump() for a in articles]) != status.get("corpus_hash"):
        raise ValueError("Canonical corpus content differs from the validation receipt")
    require_valid_corpus(articles, {"errors": []}, settings)
    return articles


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract validated Egyptian Civil Code article JSON")
    parser.add_argument("pdf_path", type=Path, help="Path to the bilingual source PDF")
    parser.add_argument("output_dir", type=Path, help="Folder for candidate JSON and reports")
    args = parser.parse_args(argv)
    articles, report = extract_to_json(args.pdf_path, args.output_dir)
    print(json.dumps({"valid": report["valid"], "article_count": len(articles),
                      "errors": report["errors"]}, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
