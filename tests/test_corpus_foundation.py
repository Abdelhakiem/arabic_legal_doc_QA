"""Meaningful checks for the source gate and notebook's deterministic boundaries."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from arabic_legal_qa.corpus import extract as corpus


class SourceExtractionTests(unittest.TestCase):
    def test_pdf_digit_order_and_source_correction(self):
        cache = ROOT / "data/processed/extraction_rows.json"
        pdf = ROOT / "data/raw/egyptian_civil_law.pdf"
        if cache.exists():
            rows = json.loads(cache.read_text(encoding="utf-8"))["rows"]
        elif pdf.exists():
            rows, layout_errors = corpus.extract_rows(pdf)
            self.assertFalse(layout_errors)
        else:
            self.skipTest("The DVC-managed source PDF is unavailable")
        article_147 = next(row for row in rows if row["en"].startswith("Article 147"))
        self.assertTrue(article_147["ar"].startswith("مادة ١٤٧"))
        articles, report = corpus.parse_articles(rows)
        self.assertEqual(len(articles), 1149)
        self.assertFalse(report["errors"])
        self.assertIn(147, {a.article_number for a in articles})
        self.assertIn(1022, {a.article_number for a in articles})
        self.assertIn(54, {a.article_number for a in articles if a.is_repealed})
        self.assertIn(389, {a.article_number for a in articles if a.is_repealed})
        article_1022 = next(a for a in articles if a.article_number == 1022)
        self.assertIn("يتحمل مالك العقار المرتفق تكاليف الأعمال", article_1022.text_ar)
        self.assertIn("source_correction", report["articles"][1022])
        corpus.require_valid_corpus(articles, report)

    def test_publish_only_after_complete_validation(self):
        def row(en, ar):
            return {"page": 1, "en": en, "ar": ar, "bold": False,
                    "boxes": [[30, 0, 297, 100], [297, 0, 564, 100]]}

        header = row("SECTION I\nTest", "الفصل الأول")
        header["bold"] = True
        invalid = [header, row("Article 1\nFirst", "مادة ١\nالنص الأول"),
                   row("Article 2\nSecond", "")]
        corrected = [header, row("Article 1\nFirst", "مادة ١\nالنص الأول"),
                     row("Article 2\nSecond", "مادة ٢\nالنص الثاني")]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pdf"
            output = Path(directory) / "processed"
            source.write_bytes(b"source-v1")
            with patch.object(corpus, "EXPECTED_ARTICLES", {1, 2}), patch.object(
                corpus, "extract_rows", side_effect=[(invalid, []), (corrected, [])]
            ):
                articles, report = corpus.extract_to_json(source, output)
                self.assertFalse(report["valid"])
                self.assertFalse((output / "articles.json").exists())
                self.assertTrue((output / "article_candidates.json").exists())
                with self.assertRaises(ValueError):
                    corpus.load_validated_articles(output, source)
                source.write_bytes(b"source-v2")
                articles, report = corpus.extract_to_json(source, output)
                self.assertTrue(report["valid"])
                self.assertEqual(len(articles), 2)
                self.assertEqual(len(json.loads((output / "articles.json").read_text())), 2)
                self.assertEqual(len(corpus.load_validated_articles(output, source)), 2)
                source.write_bytes(b"source-v3")
                with self.assertRaises(ValueError):
                    corpus.load_validated_articles(output, source)


class NotebookBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads((ROOT / "notebooks/basic_rag.ipynb").read_text(encoding="utf-8"))
        cls.notebook = notebook
        cls.ns = {}
        cells = {cell["id"]: cell for cell in notebook["cells"]}
        for cell_id in ("setup", "parser-import", "run-extraction", "preview-articles",
                        "load-canonical-when-valid"):
            exec("".join(cells[cell_id]["source"]), cls.ns)

    def test_notebook_is_parsing_first(self):
        markdown = "\n".join("".join(cell.get("source", [])) for cell in self.notebook["cells"]
                             if cell["cell_type"] == "markdown")
        self.assertIn("Parse PDF into JSON", markdown)
        self.assertIn("canonical `articles.json`", markdown)
        self.assertIn("Chunking, vector indexing, LLM generation, and evaluation will be added", markdown)

    def test_notebook_uses_reusable_parser_gate(self):
        self.assertIs(self.ns["corpus"], corpus)
        report = self.ns["extraction_report"]
        articles = self.ns["articles"]
        self.assertTrue(report["valid"])
        self.assertEqual(len(articles), 1149)
        self.assertFalse(report["errors"])
        self.assertEqual(len(self.ns["canonical_articles"]), 1149)
        self.assertIn("source_correction", report["articles"][1022])
        preview = self.ns["article_preview"](next(a for a in articles if a.article_number == 147))
        self.assertEqual(preview["citation"], "Egyptian Civil Code, Article 147")
        self.assertIn("العقد", preview["text_ar_preview"])


if __name__ == "__main__":
    unittest.main()
