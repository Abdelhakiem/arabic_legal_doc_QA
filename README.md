# Arabic Legal Document Q&A

A bilingual, retrieval-augmented Q&A service over the Egyptian Civil Code (Law 131 of 1948). The service answers Arabic and English questions using only validated Civil Code articles, returns human-verifiable article citations, and abstains when the corpus does not provide sufficient support.

The repository contains a validated corpus pipeline, a local hybrid Qdrant index, a reusable RAG orchestrator, and a FastAPI service. The design and remaining delivery checklist are maintained in [AGENT.md](AGENT.md) and [TODO.md](TODO.md).

## Foundation extraction status

The PDF extraction and LangChain loader code is in `src/arabic_legal_qa/rag/pdf_loader.py`. Run it from the repository root after installing the project dependencies:

```bash
uv sync --all-groups
uv run python -m arabic_legal_qa.rag.pdf_loader data/raw/egyptian_civil_law.pdf data/processed
```

The command writes `extraction_rows.json`, `article_candidates.json`, `extraction_report.json`, and `validation_status.json` under `data/processed/`. It writes canonical `articles.json` only when the complete corpus passes validation. The supplied PDF has a page 147 source defect around Article 1022; the loader applies an explicit Arabic correction, records that correction in `extraction_report.json`, and validates 1,149 articles. `load_validated_articles()` also checks the PDF and corpus hashes before returning canonical records.

The active experimental pipeline is in `notebooks/basic_rag.ipynb`. It starts by loading one LangChain `Document` per validated article through `EgyptianCivilCodeLoader`; chunking, embeddings, vector storage, generation, and evaluation are implemented in later notebook sections.

The default embedding model is `intfloat/multilingual-e5-small`: a lightweight multilingual model suitable for Arabic and English retrieval. It emits 384-dimensional vectors and accepts up to 512 tokens. The shared loader applies `passage: ` to indexed documents and `query: ` to user queries, as required by E5. Changing the embedding model or dimension requires rebuilding the Qdrant collection and its manifest.

## Installable RAG package

The public Python API is exported from `arabic_legal_qa.rag`; the `RAG` class provides `ingest()`, `retrieve()`, and `query()`. The installed `arabic-legal-qa` command uses this RAG package as its entry point:

```bash
uv sync --all-groups
uv run arabic-legal-qa --root . ingest
uv run arabic-legal-qa --root . query "ما هي آثار العقد؟"
```

The package wheel includes the orchestrator and CLI. Runtime dependencies for Qdrant and FastEmbed are installed with the package, rather than only with notebook dependencies.

## Architecture

```text
                              Source artifacts
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
       corpus provenance           retrieve (+ optional rerank)
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
                               FastAPI `/ask`
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

`POST /ask` returns an answer and canonical article citations (not internal chunk IDs):

```json
{
  "answer": "...",
  "sources": ["Egyptian Civil Code, Article 147"]
}
```

`GET /health` returns `{"status":"healthy","documents_indexed":N}` when ready. If startup or a health check fails, it returns HTTP 503 with an `errors` list. `/ask` rejects empty or whitespace-only questions with HTTP 422.

## Component contracts

| Component | Input | Output | Invariant |
| --- | --- | --- | --- |
| `rag.pdf_loader` | source PDF | validated LangChain article documents | uncertain splits are flagged, corrections are reported, and invalid corpora are not indexed |
| `rag.pdf_loader` | article candidates | canonical article records | integer article number, Arabic text, source page, repeal flag, canonical citation |
| `rag.chunking` | canonical articles | article-aware chunks | all chunks retain article identity and legal metadata |
| `rag.index` | deterministic chunks + model config | local, versioned index | index can be rebuilt from source artifacts |
| `rag.retrieval` | question + index | ranked evidence | article metadata remains intact through filters and ranking |
| `rag.generation` | approved evidence only | candidate answer | does not invent sources or rely on background knowledge |
| `rag.citations` | retrieved metadata | unique display citations | citations are deterministic and human-verifiable |
| `api` | HTTP request | validated response | blanks are rejected; weak evidence abstains |

## Relevant project layout

```text
src/
  api/app.py              # FastAPI app and process lifecycle
  api/schema.py           # HTTP request/response schemas
  helpers/config.py       # centralized project settings
  arabic_legal_qa/rag/    # ingestion, Qdrant, retrieval, LLM, citations
Dockerfile                # single application image
compose.yaml              # one `app` service; supporting services can be added later
data/                     # local source, processed corpus, and Qdrant artifacts
```

## Key decisions

- **Article-first retrieval:** an article is the normal chunk boundary. Long articles split only at meaningful paragraphs and every split retains the parent article metadata.
- **Bilingual evidence:** Arabic and English article texts are indexed with explicit language metadata and a shared article identity. Equivalent bilingual questions are regression-tested against the same expected article.
- **Provider boundaries:** embedding, vector index, generator, reranker, tracing, and experiment tracking are accessed through small interfaces. Tests use local fakes; production adapters remain replaceable.
- **Safety before fluency:** retrieval thresholds, source validation, and repeal disclosure execute before an answer is returned. Reranking, streaming, quantization, and high-throughput serving are later enhancements, not correctness mechanisms.
- **Reproducibility:** pin and record corpus, extractor, chunking, model, prompt, and index versions with every experiment and request trace.

## Run with Docker Compose

The deployment uses one Docker image and one Compose service named `app`. That image contains the FastAPI app, the RAG package, the validated articles, and the embedded local Qdrant index—there is no separate RAG or Qdrant service. Ollama is intentionally not included; Compose selects Groq, so set `GROQ_TOKEN` in `.env`. The Hugging Face embedding model is downloaded on first startup and persisted in a named volume, so first startup needs internet access and may take a few minutes.

Before building, make sure the local artifact bundle exists: `data/raw/egyptian_civil_law.pdf`, `data/processed/articles.json`, and the complete `data/vector_store/qdrant/` directory including `manifest.json`. These generated/data files are git-ignored and are not currently fetched automatically by Compose; a fresh clone needs that bundle supplied separately. The image snapshots the corpus and index at build time, so rebuild the image after changing them.

Set `GROQ_TOKEN` in `.env` after copying the example file. Then the three commands are:

```bash
cp -n .env.example .env
```

```bash
docker compose up --build -d --wait
```

```bash
curl -fsS http://localhost:8000/health && curl -fsS http://localhost:8000/ask -H 'Content-Type: application/json' -d '{"question":"ما آثار العقد؟"}'
```

The Compose health check waits for the API and RAG resources to become ready. Keep `.env` private; do not commit credentials. This uses a local image build, so Docker Compose v2 and the artifact bundle above are prerequisites.

For local development without Docker, install dependencies with `uv sync --all-groups`, then run the CLI or `uv run uvicorn api.app:app --host 0.0.0.0 --port 8000` from the repository root. The command-line entry point is `arabic-legal-qa`. If you want to use Ollama locally, install its optional adapter with `uv sync --all-groups --extra ollama` and configure `LLM_PROVIDER=ollama`.

## Quality gates

- Corpus records must preserve Arabic text, article identity, source page, repeal status, and canonical citation.
- The gold set must include at least 50 questions spanning Arabic/English equivalents, answerable cases, repeal cases, abstentions, and domain-shift questions.
- The selected configuration needs RAGAS faithfulness of at least `0.75`; monitoring alerts below `0.80`.
- CI must reject corpus-validation failures, retrieval regressions, failed deterministic tests, and failed evaluation gates.

## Scope and limitation

This is an informational retrieval system limited to its versioned Egyptian Civil Code corpus. It is not legal advice, does not establish the current law beyond the included source, and must abstain rather than extrapolate to facts, laws, or jurisdictions not contained in that corpus.
