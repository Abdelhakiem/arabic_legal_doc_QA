# Arabic Legal Document Q&A

A bilingual, retrieval-augmented Q&A service over the Egyptian Civil Code (Law 131 of 1948). The service answers Arabic and English questions using only validated Civil Code articles, returns human-verifiable article citations, and abstains when the corpus does not provide sufficient support.

This repository is in the foundation phase. The design and delivery checklist are maintained in [AGENT.md](AGENT.md) and [TODO.md](TODO.md); completed capabilities must be verified before this README describes them as available.

## Foundation extraction status

The PDF extraction and LangChain loader code is in `src/arabic_legal_qa/rag/pdf_loader.py`. Run it from the repository root after installing the project dependencies:

```bash
uv sync --all-groups
uv run python -m arabic_legal_qa.rag.pdf_loader data/raw/egyptian_civil_law.pdf data/processed
```

The command writes `extraction_rows.json`, `article_candidates.json`, `extraction_report.json`, and `validation_status.json` under `data/processed/`. It writes canonical `articles.json` only when the complete corpus passes validation. The supplied PDF has a page 147 source defect around Article 1022; the loader applies an explicit Arabic correction, records that correction in `extraction_report.json`, and validates 1,149 articles. `load_validated_articles()` also checks the PDF and corpus hashes before returning canonical records.

The active experimental pipeline is in `notebooks/basic_rag.ipynb`. It starts by loading one LangChain `Document` per validated article through `EgyptianCivilCodeLoader`; chunking, embeddings, vector storage, generation, and evaluation are implemented in later notebook sections.

The default embedding model is `intfloat/multilingual-e5-small`: a lightweight multilingual model suitable for Arabic and English retrieval. It emits 384-dimensional vectors and accepts up to 512 tokens. The shared loader applies `passage: ` to indexed documents and `query: ` to user queries, as required by E5. Changing the embedding model or dimension requires rebuilding the Qdrant collection and its manifest.

## Architecture

```text
                         Versioned, DVC-tracked inputs
                                      │
                                      ▼
                     Bilingual Egyptian Civil Code PDF
                                      │
                                      ▼
                   Position-aware extraction and parsing
                                      │
                                      ▼
             Canonical, validated article JSON (source of truth)
              │                       │                         │
              │                       │                         └──► corpus reports / spot checks
              │                       ▼
              │             article-aware bilingual chunks
              │                       │
              │                       ▼
              │            embeddings + persistent vector index
              │                       │
              ▼                       ▼
      DVC provenance               retrieve (+ optional rerank)
                                              │
                                              ▼
                              evidence sufficiency gate
                                   │                  │
                              insufficient        sufficient
                                   │                  │
                                   ▼                  ▼
                              abstention     grounded generator
                                                     │
                                                     ▼
                         deterministic citation / repeal validation
                                                     │
                                                     ▼
                         FastAPI `/ask` → answer, sources, grounded
                                                     │
                         ┌───────────────────────────┴───────────────────────────┐
                         ▼                                                       ▼
              MLflow + offline evaluation                         Langfuse + service metrics
```

### Source-of-truth rule

The raw PDF is an input artifact, not a retrieval source. `data/processed/articles.json` (or a versioned equivalent) is the canonical corpus. The vector index, chunks, reports, and evaluation outputs are all reproducible derivatives. Neither a model response nor a vector-store payload may create a citation.

### Query path

1. Validate the question and infer the preferred answer language.
2. Retrieve article chunks with their complete legal metadata.
3. Apply an evidence-sufficiency policy. Weak or out-of-corpus retrieval returns an explicit abstention.
4. Prompt the generator with only retrieved article text and instructions to avoid unsupported claims.
5. Create and deduplicate citations from retrieved `article_number` metadata; disclose the repeal status of any cited article.
6. Return an informational, source-bounded answer—not legal advice.

`POST /ask` returns:

```json
{
  "answer": "...",
  "sources": ["Egyptian Civil Code, Article 147"],
  "grounded": true
}
```

`GET /health` returns the application, corpus, and index state without requiring a model call.

## Component contracts

| Component | Input | Output | Invariant |
| --- | --- | --- | --- |
| `rag.pdf_loader` | DVC-tracked PDF | validated LangChain article documents | uncertain splits are flagged, corrections are reported, and invalid corpora are not indexed |
| `rag.pdf_loader` | article candidates | canonical article records | integer article number, Arabic text, source page, repeal flag, canonical citation |
| `rag.chunking` | canonical articles | article-aware chunks | all chunks retain article identity and legal metadata |
| `rag.index` | deterministic chunks + model config | local, versioned index | index can be rebuilt from source artifacts |
| `rag.retrieval` | question + index | ranked evidence | article metadata remains intact through filters and ranking |
| `rag.generation` | approved evidence only | candidate answer | does not invent sources or rely on background knowledge |
| `rag.citations` | retrieved metadata | unique display citations | citations are deterministic and human-verifiable |
| `api` | HTTP request | validated response | blanks are rejected; weak evidence abstains |

## Planned repository layout

```text
src/arabic_legal_qa/
  api.py                 # FastAPI endpoints and dependency wiring
  config.py              # environment-based settings
  schemas.py             # HTTP schemas shared by the API
  rag/                   # load PDF, chunk, embed, index, retrieve, generate, cite
  evaluation/            # gold set, metrics, RAGAS, MLflow runners
  monitoring/            # traces, metrics, drift and retention controls
scripts/                 # thin CLIs; no business logic
tests/                   # unit, integration, regression tests
data/                    # DVC-managed raw, processed, index, evaluation data
reports/                 # small, reproducible quality evidence
```

## Key decisions

- **Article-first retrieval:** an article is the normal chunk boundary. Long articles split only at meaningful paragraphs and every split retains the parent article metadata.
- **Bilingual evidence:** Arabic and English article texts are indexed with explicit language metadata and a shared article identity. Equivalent bilingual questions are regression-tested against the same expected article.
- **Provider boundaries:** embedding, vector index, generator, reranker, tracing, and experiment tracking are accessed through small interfaces. Tests use local fakes; production adapters remain replaceable.
- **Safety before fluency:** retrieval thresholds, source validation, and repeal disclosure execute before an answer is returned. Reranking, streaming, quantization, and high-throughput serving are later enhancements, not correctness mechanisms.
- **Reproducibility:** pin and record corpus, extractor, chunking, model, prompt, and index versions with every experiment and request trace.

## Intended reviewer workflow

Once the corresponding foundation work is complete, a reviewer should need only these three commands:

```bash
uv sync --all-groups
```

```bash
dvc pull && dvc repro
```

```bash
uv run uvicorn arabic_legal_qa.api:app --host 0.0.0.0 --port 8000
```

The current implementation status and evidence requirements are tracked in [TODO.md](TODO.md). Do not add credentials to Git; copy `.env.example` to `.env` only after that file is introduced.

## Quality gates

- Corpus records must preserve Arabic text, article identity, source page, repeal status, and canonical citation.
- The gold set must include at least 50 questions spanning Arabic/English equivalents, answerable cases, repeal cases, abstentions, and domain-shift questions.
- The selected configuration needs RAGAS faithfulness of at least `0.75`; monitoring alerts below `0.80`.
- CI must reject corpus-validation failures, retrieval regressions, failed deterministic tests, and failed evaluation gates.

## Scope and limitation

This is an informational retrieval system limited to its versioned Egyptian Civil Code corpus. It is not legal advice, does not establish the current law beyond the included source, and must abstain rather than extrapolate to facts, laws, or jurisdictions not contained in that corpus.
