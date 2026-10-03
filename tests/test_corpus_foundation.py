"""Meaningful checks for the source gate and notebook's deterministic boundaries."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from arabic_legal_qa.rag import pdf_loader
from arabic_legal_qa.rag import llms
from arabic_legal_qa.rag import chunking as vec_db


class CharacterTokenizer:
    def encode(self, text, **_kwargs):
        return list(text)


class SourceExtractionTests(unittest.TestCase):
    def test_pdf_digit_order_and_source_correction(self):
        cache = ROOT / "data/processed/extraction_rows.json"
        pdf = ROOT / "data/raw/egyptian_civil_law.pdf"
        if cache.exists():
            rows = json.loads(cache.read_text(encoding="utf-8"))["rows"]
        elif pdf.exists():
            rows, layout_errors = pdf_loader.extract_rows(pdf)
            self.assertFalse(layout_errors)
        else:
            self.skipTest("The DVC-managed source PDF is unavailable")
        article_147 = next(row for row in rows if row["en"].startswith("Article 147"))
        self.assertTrue(article_147["ar"].startswith("مادة ١٤٧"))
        articles, report = pdf_loader.parse_articles(rows)
        self.assertEqual(len(articles), 1149)
        self.assertFalse(report["errors"])
        self.assertIn(147, {a.article_number for a in articles})
        self.assertIn(1022, {a.article_number for a in articles})
        self.assertIn(54, {a.article_number for a in articles if a.is_repealed})
        self.assertIn(389, {a.article_number for a in articles if a.is_repealed})
        article_1022 = next(a for a in articles if a.article_number == 1022)
        self.assertIn("يتحمل مالك العقار المرتفق تكاليف الأعمال", article_1022.text_ar)
        self.assertIn("source_correction", report["articles"][1022])
        pdf_loader.require_valid_corpus(articles, report)

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
            with patch.object(pdf_loader, "EXPECTED_ARTICLES", {1, 2}), patch.object(
                pdf_loader, "extract_rows", side_effect=[(invalid, []), (corrected, [])]
            ):
                articles, report = pdf_loader.extract_to_json(source, output)
                self.assertFalse(report["valid"])
                self.assertFalse((output / "articles.json").exists())
                self.assertTrue((output / "article_candidates.json").exists())
                with self.assertRaises(ValueError):
                    pdf_loader.load_validated_articles(output, source)
                source.write_bytes(b"source-v2")
                articles, report = pdf_loader.extract_to_json(source, output)
                self.assertTrue(report["valid"])
                self.assertEqual(len(articles), 2)
                self.assertEqual(len(json.loads((output / "articles.json").read_text())), 2)
                self.assertEqual(len(pdf_loader.load_validated_articles(output, source)), 2)
                source.write_bytes(b"source-v3")
                with self.assertRaises(ValueError):
                    pdf_loader.load_validated_articles(output, source)

    def test_langchain_loader_preserves_article_metadata(self):
        pdf = ROOT / "data/raw/egyptian_civil_law.pdf"
        if not pdf.exists():
            self.skipTest("The DVC-managed source PDF is unavailable")
        loader = pdf_loader.EgyptianCivilCodeLoader(pdf, ROOT / "data/processed")
        documents = loader.load()
        self.assertEqual(len(documents), 2298)
        article_147_ar = next(doc for doc in documents
                              if doc.metadata["article_number"] == 147 and doc.metadata["language"] == "ar")
        article_147_en = next(doc for doc in documents
                              if doc.metadata["article_number"] == 147 and doc.metadata["language"] == "en")
        self.assertIn("العقد", article_147_ar.page_content)
        self.assertIn("contract", article_147_en.page_content.lower())
        self.assertEqual(article_147_ar.metadata["citation"], "Egyptian Civil Code, Article 147")
        self.assertEqual(article_147_ar.metadata["chunk_position"], 0)
        self.assertIn("chunk_id", article_147_ar.metadata)
        self.assertNotIn("corpus_hash", article_147_ar.metadata)
        self.assertNotIn("source_hash", article_147_ar.metadata)
        self.assertNotIn("extractor_version", article_147_ar.metadata)
        self.assertNotEqual(article_147_ar.metadata["chunk_id"], article_147_en.metadata["chunk_id"])
        article_1_ar = next(doc for doc in documents
                            if doc.metadata["article_number"] == 1 and doc.metadata["language"] == "ar")
        self.assertEqual(article_1_ar.metadata["book"], pdf.name)
        article_1022 = next(doc for doc in documents
                            if doc.metadata["article_number"] == 1022 and doc.metadata["language"] == "ar")
        self.assertTrue(article_1022.metadata["has_source_correction"])
        self.assertIn("يتحمل مالك العقار المرتفق تكاليف الأعمال", article_1022.page_content)


class NotebookBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads((ROOT / "notebooks/basic_rag.ipynb").read_text(encoding="utf-8"))
        cls.notebook = notebook
        cls.ns = {}
        load_sources = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and ("EgyptianCivilCodeLoader" in "".join(cell["source"])
                 or "loader = EgyptianCivilCodeLoader" in "".join(cell["source"]))
        ]
        cls.load_cell_source = "\n".join(load_sources)
        exec(cls.load_cell_source, cls.ns)

    def test_notebook_loads_from_rag_loader(self):
        markdown = "\n".join("".join(cell.get("source", [])) for cell in self.notebook["cells"]
                             if cell["cell_type"] == "markdown")
        self.assertIn("Load the document", markdown)
        self.assertIn("Split the document into chunks", markdown)
        self.assertIn("arabic_legal_qa.rag.pdf_loader", self.load_cell_source)

    def test_notebook_loader_outputs_documents(self):
        loader = self.ns["loader"]
        documents = self.ns["documents"]
        self.assertIsInstance(loader, pdf_loader.EgyptianCivilCodeLoader)
        self.assertEqual(len(documents), 2298)
        self.assertTrue(loader.report["valid"])
        article_147_ar = next(doc for doc in documents
                              if doc.metadata["article_number"] == 147 and doc.metadata["language"] == "ar")
        self.assertEqual(article_147_ar.metadata["citation"], "Egyptian Civil Code, Article 147")
        self.assertIn("العقد", article_147_ar.page_content)


class ChunkingTests(unittest.TestCase):
    def test_split_text_preserves_text_and_token_limit(self):
        tokenizer = CharacterTokenizer()
        text = "١) أول بند طويل.\n٢) ثاني بند أطول قليلا.\n٣) ثالث بند."
        chunks = vec_db.split_text(text, tokenizer, limit=18)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(vec_db.token_count(tokenizer, chunk) <= 18 for chunk in chunks))

    def test_make_chunks_preserves_article_metadata_and_stable_ids(self):
        tokenizer = CharacterTokenizer()
        article = pdf_loader.Article(
            article_number=147,
            book="BOOK I",
            chapter="CHAPTER I",
            section="Section I",
            topic="Contracts",
            text_ar="العقد شريعة المتعاقدين.\nولا يجوز نقضه إلا باتفاق الطرفين.",
            text_en="The contract is the law of the parties.",
            is_repealed=False,
            source_page=16,
            citation="Egyptian Civil Code, Article 147",
        )
        config = vec_db.ChunkingConfig(embedding_limit=32, chunk_tokens=16)
        first = vec_db.make_chunks([article], tokenizer, config, "corpus")
        second = vec_db.make_chunks([article], tokenizer, config, "corpus")
        self.assertEqual([doc.metadata["chunk_id"] for doc in first],
                         [doc.metadata["chunk_id"] for doc in second])
        self.assertTrue(all(doc.metadata["article_number"] == 147 for doc in first))
        self.assertTrue(all(doc.metadata["citation"] == article.citation for doc in first))
        self.assertEqual(
            "".join(doc.page_content for doc in first if doc.metadata["language"] == "ar"),
            article.text_ar,
        )

    def test_chunk_documents_preserves_loader_metadata(self):
        tokenizer = CharacterTokenizer()
        document = pdf_loader.article_to_document(
            pdf_loader.Article(
                article_number=1,
                text_ar="أول نص طويل جدا يحتاج إلى تقسيم.",
                text_en="First text.",
                is_repealed=False,
                source_page=1,
                citation="Egyptian Civil Code, Article 1",
            ),
            report={"articles": {}, "extractor_version": "test"},
            source_hash="source",
            corpus_hash="corpus",
            source_name="egyptian_civil_law.pdf",
            language="ar",
        )
        chunks = vec_db.chunk_documents([document], tokenizer, vec_db.ChunkingConfig(chunk_tokens=10), "corpus")
        self.assertEqual("".join(chunk.page_content for chunk in chunks), document.page_content)
        self.assertTrue(all(chunk.metadata["book"] == "egyptian_civil_law.pdf" for chunk in chunks))
        self.assertTrue(all(chunk.metadata["corpus_hash"] == "corpus" for chunk in chunks))


class EmbeddingLoaderTests(unittest.TestCase):
    def test_embedding_bundle_resolves_revision_and_reuses_process_cache(self):
        calls = []

        @lru_cache(maxsize=4)
        def fake_load(config, cache_dir):
            calls.append((config, cache_dir))
            return llms.EmbeddingBundle(
                embedder=object(),
                tokenizer=object(),
                config=config,
                cache_dir=cache_dir,
            )

        llms._load_embedding_bundle_cached.cache_clear()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            llms, "_resolve_model_revision", return_value="resolved-sha"
        ), patch.object(llms, "_load_embedding_bundle_cached", fake_load):
            config = llms.EmbeddingConfig(model_name="test/model", embedding_limit=128)
            first = llms.load_embedding_bundle(config, directory)
            second = llms.load_embedding_bundle(config, directory)
            self.assertEqual(first.config.model_revision, "resolved-sha")
            self.assertEqual(second.config.model_revision, "resolved-sha")
            self.assertIs(first, second)
            self.assertEqual(len(calls), 1)
            self.assertTrue(Path(directory).exists())
        llms._load_embedding_bundle_cached.cache_clear()

    def test_vec_db_delegates_embedding_loader_to_llms(self):
        with patch.object(vec_db, "load_embedding_bundle", return_value="bundle") as mocked:
            self.assertEqual(vec_db.load_embedder(), "bundle")
            mocked.assert_called_once_with(config=None, cache_dir=None)


if __name__ == "__main__":
    unittest.main()
