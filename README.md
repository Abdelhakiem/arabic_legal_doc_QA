# Arabic Legal Document Q&A

A bilingual, retrieval-augmented Q&A service over the Egyptian Civil Code (Law 131 of 1948). The service answers Arabic and English questions using only validated Civil Code articles, returns human-verifiable article citations, and abstains when the corpus does not provide sufficient support.

The repository contains a validated corpus pipeline, a local hybrid Qdrant index, a reusable RAG orchestrator, a FastAPI service, and an MLflow-backed evaluation/config registry. The design and remaining delivery checklist are maintained in [AGENT.md](AGENT.md) and [TODO.md](TODO.md).

## Foundation extraction status

The PDF extraction and LangChain loader code is in `src/arabic_legal_qa/rag/pdf_loader.py`. Run it from the repository root after installing the project dependencies:

```bash
uv sync --all-groups
uv run python -m arabic_legal_qa.rag.pdf_loader data/raw/egyptian_civil_law.pdf data/processed
```

The command writes `extraction_rows.json`, `article_candidates.json`, `extraction_report.json`, and `validation_status.json` under `data/processed/`. It writes canonical `articles.json` only when the complete corpus passes validation. The supplied PDF has a page 147 source defect around Article 1022; the loader applies an explicit Arabic correction, records that correction in `extraction_report.json`, and validates 1,149 articles. `load_validated_articles()` also checks the PDF and corpus hashes before returning canonical records.

## DVC data versioning and reproducibility

DVC tracks the source PDF and `data/evaluation/foundation_questions.json` with `.dvc` pointer files in Git. The five processed JSON outputs are produced by the `build-corpus` stage and recorded in `dvc.lock`. The `build-index` stage reads and validates the tracked canonical `articles.json`, chunks and embeds those records with the committed model revision, and builds the local Qdrant collection under `data/vector_store/qdrant/`. DVC tracks that index directory as a reproducible pipeline output. The index stage does not re-parse the PDF; the corpus stage owns PDF extraction.

Each developer configures the machine-local DVC remote once. For the Google Drive for desktop mirror used by this project, point the local remote at that machine's synced `dvc-store` directory (the path is intentionally kept in ignored `.dvc/config.local`):

```bash
uv sync --group dvc
uv run --group dvc dvc remote add --local --default gdrive-sync \
  "/path/to/Google Drive/My Drive/arabic_legal_doc_qa/dvc-store"
```

On a fresh Git checkout, fetch the artifacts and run the dependency-aware pipeline:

```bash
uv run --group dvc dvc pull
uv run --group dvc dvc repro
```

`dvc pull` restores the PDF, evaluation set, processed corpus, and saved index from the configured remote. A normal `dvc repro` skips stages whose tracked inputs and outputs already match `dvc.lock`. To explicitly prove a rebuild from the PDF through extraction and indexing, force the complete pipeline:

```bash
uv run --group dvc dvc repro --force
```

To rebuild only the index from the already-pulled validated articles, run `uv run --group dvc dvc repro --force build-index`. DVC uses deterministic chunk IDs and records the corpus hash, embedding revision, chunk settings, and chunk digest in the Qdrant manifest. The embedding model and FastEmbed sparse model are cached locally under `data/model_cache`; they are re-downloaded on a new machine if absent. Allow Google Drive for desktop to finish syncing before pulling from another machine. Commit the `.dvc` pointers, `dvc.yaml`, `dvc.lock`, pipeline code, and dependency lockfile to Git; do not commit `.dvc/config.local` or the data files themselves.

The active experimental pipeline is in `notebooks/basic_rag.ipynb`. It starts by loading one LangChain `Document` per validated article through `EgyptianCivilCodeLoader`; chunking, embeddings, vector storage, generation, and evaluation are implemented in later notebook sections.

The default embedding model is `intfloat/multilingual-e5-small`, executed with ONNX Runtime on CPU and the lightweight Rust `tokenizers` package. The loader downloads only the ONNX encoder, `tokenizer.json`, and model config—not PyTorch weights—and performs masked mean pooling plus L2 normalization to produce 384-dimensional vectors. It applies `passage: ` to indexed documents and `query: ` to user queries, as required by E5. Changing the embedding backend/model or dimension requires rebuilding the Qdrant collection and its manifest; indexes without the ONNX backend marker are automatically re-ingested.

## Installable RAG package

The public Python API is exported from `arabic_legal_qa.rag`; the `RAG` class provides `ingest()`, `retrieve()`, and `query()`. The installed `arabic-legal-qa` command uses this RAG package as its entry point:

```bash
uv sync --all-groups
uv run arabic-legal-qa --root . ingest
uv run arabic-legal-qa --root . query "ما هي آثار العقد؟"
```

The package wheel includes the orchestrator and CLI. Runtime dependencies for Qdrant and FastEmbed are installed with the package, rather than only with notebook dependencies.

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
compose.yaml              # `app` plus persistent MLflow tracking/registry service
data/                     # local source, processed corpus, and Qdrant artifacts
```

## Key decisions

- **Article-first retrieval:** an article is the normal chunk boundary. Long articles split only at meaningful paragraphs and every split retains the parent article metadata.
- **Bilingual evidence:** Arabic and English article texts are indexed with explicit language metadata and a shared article identity. Equivalent bilingual questions are regression-tested against the same expected article.
- **Provider boundaries:** embedding, vector index, generator, reranker, tracing, and experiment tracking are accessed through small interfaces. Tests use local fakes; production adapters remain replaceable.
- **Safety before fluency:** retrieval thresholds, source validation, and repeal disclosure execute before an answer is returned. Reranking, streaming, quantization, and high-throughput serving are later enhancements, not correctness mechanisms.
- **Reproducibility:** pin and record corpus, extractor, chunking, model, prompt, and index versions with every experiment and request trace.

## Run with Docker Compose

Compose runs two services from the same image: `app` (FastAPI/RAG) and `mlflow` (tracking UI, REST API, and Model Registry). Qdrant remains embedded in the app container; there is no separate Qdrant service. MLflow run metadata and artifacts persist in the `mlflow-data` named volume using SQLite/local artifacts, suitable for a single-host deployment. Ollama is intentionally not included; Compose selects Groq, so set `GROQ_TOKEN` in `.env`. The Hugging Face embedding model is downloaded on first startup and persisted in a named volume, so first startup needs internet access and may take a few minutes.

Before building, restore the artifact bundle with DVC (`dvc pull` after configuring the machine-local remote above). The image snapshots the corpus and index at build time, so rebuild the image after changing them.

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

The Compose health checks wait for MLflow and the API/RAG resources to become ready. MLflow is available at `http://localhost:5000`; its data survives container recreation in `mlflow-data`. Its Host allowlist is limited to the local client and Compose service hostnames; the published port is also bound to loopback. Do not publish it on a network interface without configuring a restrictive allowlist and authentication. The app looks for model `arabic-legal-rag-config` with alias `champion` at startup. Before an alias is promoted it warns and uses centralized defaults; set `MLFLOW_CONFIG_REQUIRED=true` to make startup fail instead. After promoting a new config, restart `app` to load it; the service intentionally does not mutate a live index/config mid-request. Keep `.env` private; do not commit credentials. This uses a local image build, so Docker Compose v2 and the artifact bundle above are prerequisites.

### MLflow experiments and RAGAS

Install the evaluation extra locally and run the six candidates against `dev`. The evaluator uses two stages: it scores expected-article hit rate/recall for every candidate without calling Groq, then runs answer generation and RAGAS only for the two retrieval leaders. The quick mode uses four balanced Arabic/English questions; full mode uses the complete dev split. Candidates that differ only in `k` share the same temporary Qdrant index.

| Run | Chunk tokens | Overlap | Retrieval `k` |
| --- | ---: | ---: | ---: |
| 1 | 256 | 0 | 5 |
| 2 | 384 | 0 | 5 |
| 3 | 480 | 0 | 5 |
| 4 | 384 | 32 | 5 |
| 5 | 480 | 32 | 5 |
| 6 | 384 | 32 | 8 |

```bash
uv sync --extra evaluation --group test
uv run --extra evaluation arabic-legal-qa-evaluate --mode quick
```

Ensure Compose is running (`docker compose up -d mlflow app`) and `GROQ_TOKEN` is set. The evaluator defaults to `http://localhost:5000` if `MLFLOW_TRACKING_URI` is unset. In the MLflow UI, open `rag-config-experiments`: all six runs have retrieval metrics; finalist runs additionally have `faithfulness` and context precision (the default core judge metrics). RAGAS concurrency defaults to one worker to avoid request bursts; this limits concurrency, not total calls. Use `--judge-metrics all` on finalists to add ContextUtilization and AnswerRelevancy, or `--finalists 1` to reduce Groq usage further. Each run logs config, corpus/evaluation hashes, Qdrant manifest, and per-question results. The winner is registered only from generation-scored finalists and only if faithfulness meets `MLFLOW_MIN_FAITHFULNESS` (default `0.75`); use `--no-promote` to prevent alias changes.

For a slower full-dev comparison, or to change the finalist count:

```bash
uv run --extra evaluation arabic-legal-qa-evaluate --mode full --finalists 2
```

Retrieval results, generated answers, and RAGAS scores are cached separately under `data/evaluation/cache`, keyed by the relevant corpus/config/question/prompt/model inputs. Re-running the same command reuses completed stages; use `--no-cache` to force recomputation. Delete the cache directory only when you intentionally want to discard those saved evaluation outputs.

After promotion, restart the app so it loads the alias. Then validate that frozen winner on held-out questions (this command cannot promote or tune):

```bash
uv run --extra evaluation arabic-legal-qa-evaluate --validate-champion
```

The compact foundation set has 10 questions (6 dev, 4 held-out), balanced across Arabic and English. It includes multi-article retrieval cases, repeal-status checks, and out-of-scope questions. It has expected article IDs but no reviewed reference answers, so retrieval screening uses deterministic expected-article hit/recall and finalist RAGAS runs use no-reference context precision and faithfulness by default. `--judge-metrics all` also adds context utilization and answer relevancy. If every question in a split has a human-reviewed `reference_answer`, the runner automatically adds reference-based context precision, context recall, and answer correctness. Do not fill those answers with model-generated text and call them gold labels; this small set is for initial comparisons, not statistically reliable claims.

For local development without Docker, install dependencies with `uv sync --all-groups`, then run the CLI or `uv run uvicorn api.app:app --host 0.0.0.0 --port 8000` from the repository root. The command-line entry point is `arabic-legal-qa`. If you want to use Ollama locally, install its optional adapter with `uv sync --all-groups --extra ollama` and configure `LLM_PROVIDER=ollama`.

## Quality gates

- Corpus records must preserve Arabic text, article identity, source page, repeal status, and canonical citation.
- The gold set must include at least 50 questions spanning Arabic/English equivalents, answerable cases, repeal cases, abstentions, and domain-shift questions.
- The selected configuration needs RAGAS faithfulness of at least `0.75`; monitoring alerts below `0.80`.
- CI must reject corpus-validation failures, retrieval regressions, failed deterministic tests, and failed evaluation gates.

## Scope and limitation

This is an informational retrieval system limited to its versioned Egyptian Civil Code corpus. It is not legal advice, does not establish the current law beyond the included source, and must abstain rather than extrapolate to facts, laws, or jurisdictions not contained in that corpus.
