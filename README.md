# RAG System with Orchestrated Ingestion Pipeline

A retrieval-augmented generation system over SEC 10-K filings, built to compare
retrieval strategies (dense vs. hybrid) on a measured eval set, with ingestion
run as an orchestrated, schedulable pipeline rather than a one-off script.

**Status: Phase 3 complete** (chunking + retrieval comparison, measured on a
24-question eval set). Phases 4–5 (failure analysis, serving) are in
progress — see [Build phases](#build-phases) below.

## Architecture (target — see [Build phases](#build-phases) for what's live)

```
SEC EDGAR (10-K filings)
     |
     v
Orchestrated ingestion pipeline (Dagster, containerized)     [Phase 2, live]
  - extract -> chunk (fixed-size; semantic added Phase 3) -> embed -> load
  - partitioned per company (6 static partitions) so each filing is
    independently materializable/retriable
  - dbt-style data quality check at every stage
     |
     v
Qdrant (Docker)
     |
     v
Retrieval layer: dense baseline vs. hybrid (dense+BM25, RRF) [Phase 3, live]
     |
     v
Generation: llama3.2:3b via Ollama (local, no hosted API)
     |
     v
Evaluation harness: hand-built precision@5/hit@5/MRR +
LLM-judged faithfulness/relevance, 24-question eval set      [Phase 3, live]
     |
     v
FastAPI serving, structured logging                          [Phase 5]
```

## Repo layout

```
src/ragpipeline/
  config.py              # .env-backed settings, shared paths
  ingestion/
    fetch.py              # SEC EDGAR downloader (fetch_one/fetch_all)
    extract.py             # HTML -> cleaned text, strips hidden XBRL, tags [SECTION] headers (extract_one/extract_all)
    chunk.py                # fixed-size token chunker (strategy #1), deterministic chunk IDs
    embed.py                  # Ollama embedding wrapper (nomic-embed-text)
    load.py                    # Qdrant collection create + upsert, dimension checks
  retrieval/
    dense.py               # dense-only retrieval baseline, strategy-filtered
    hybrid.py              # dense + BM25 sparse, fused via Reciprocal Rank Fusion
  generation/
    ollama_client.py       # grounded RAG prompt -> llama3.2:3b
orchestration/             # Phase 2: Dagster ingestion DAG
  assets.py                # extract/chunk/embed/load, partitioned per ticker
  checks.py                # dbt-style data quality assertions per stage
  definitions.py           # Definitions object, job, daily schedule
scripts/
  run_naive_pipeline.py    # Phase 1 end-to-end script (--ingest, --query)
  ingest_chunking_strategies.py  # Phase 3: load both chunking strategies side by side
eval/
  questions.json           # 24 hand-authored questions with expected source ticker
  harness.py                # runs all 4 chunk x retrieval configs, scores + judges each
  results.json              # full per-question results from the last harness run
  make_chart.py              # renders results.json -> comparison_chart.png
data/
  raw/                    # downloaded 10-K HTML (gitignored)
  processed/              # cleaned text per filing (gitignored)
Dockerfile                # image for the dagster service
docker-compose.yml        # qdrant + dagster (api service added in Phase 5)
```

## Setup (Phase 1)

Requires Docker and [Ollama](https://ollama.com) running locally.

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b

python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

cp .env.example .env   # set SEC_EDGAR_USER_AGENT to "<name> <email>" — required by SEC

docker compose up -d qdrant
```

Run the naive pipeline:

```bash
# fetch -> extract -> chunk -> embed -> index (fresh collection each run)
./.venv/bin/python scripts/run_naive_pipeline.py --ingest

# ask a question against the indexed corpus
./.venv/bin/python scripts/run_naive_pipeline.py --query "What are Apple's main risk factors related to supply chain?"
```

## Phase 1 results

- 6 filings ingested (AAPL, MSFT, JPM, XOM, PFE, TSLA), ~1,800 chunks total
  at 500 tokens/chunk with 50-token overlap.
- Dense retrieval correctly surfaces the right company/section for
  in-corpus questions (e.g. a JPMorgan interest-rate question returns
  JPM Item 1A chunks at cosine similarity ~0.78–0.79).
- The system prompt alone (no formal refusal mechanism yet — that's
  Phase 4) is already enough to make `llama3.2:3b` decline out-of-corpus
  questions (tested: asking about Netflix, which isn't ingested, correctly
  returns "I couldn't find any information on Netflix..." instead of
  hallucinating).

## Setup (Phase 2 — orchestrated ingestion)

Everything from Phase 1 setup, plus:

```bash
./.venv/bin/pip install -r requirements.txt   # now includes dagster, dagster-webserver
```

Run the DAG locally (outside Docker) for one company:

```bash
export DAGSTER_HOME=$(pwd)/.dagster_home && mkdir -p "$DAGSTER_HOME"
./.venv/bin/dagster asset materialize --select "*" -m orchestration.definitions --partition AAPL
```

...or bring up the containerized version and use the UI:

```bash
docker compose up -d qdrant dagster
open http://localhost:3000
```

In the UI: **Overview → Assets → Materialize all**, then pick a partition (or
"All partitions") to run the whole `extracted_filing → chunks →
embedded_chunks → loaded_points` chain for each of the 6 tickers, with each
stage's data quality check visible next to it.

## Phase 2 results

- **Partitioned per ticker**: 4 stages × 6 companies, each independently retriable.
- **Data quality checks per stage** (dbt-style): min extracted-text length,
  no blank chunks, embedding dim/count match, Qdrant row-count reconciliation.
- **Clean-state run verified**: dropped the collection, materialized all 24
  stage/partition combos, all checks passed, 1,812 total points — matching
  Phase 1 exactly.
- **Idempotent retries verified**: re-running a partition leaves the point
  count unchanged. Required switching chunk IDs from random `uuid4` to
  deterministic `uuid5` (`ticker/source_file/strategy/chunk_index`) so
  re-loads upsert instead of duplicate — the row-count check is what caught
  the original duplication (270 vs. expected 135 for AAPL).
- **Containerized**: `dagster` service + `Dockerfile`, reaches Qdrant over
  the compose network and the host's Ollama via `host.docker.internal`
  (Ollama stays local, not containerized). Verified assets load correctly
  in-container via the GraphQL API.
- **Caveat**: partitioned asset checks are a preview feature as of Dagster
  1.13 and may change in a future patch release.

## Setup (Phase 3 — chunking + retrieval comparison)

```bash
./.venv/bin/pip install -r requirements.txt   # now includes rank_bm25

# load both chunking strategies into the same collection (tagged by `strategy` payload field)
./.venv/bin/python scripts/ingest_chunking_strategies.py --recreate

# run all 4 configs (fixed_size/semantic x dense/hybrid) against the 24-question eval set
PYTHONPATH=. ./.venv/bin/python -m eval.harness

# optional: regenerate the chart below from eval/results.json
./.venv/bin/pip install matplotlib
PYTHONPATH=. ./.venv/bin/python -m eval.make_chart
```

## Phase 3 results

24 hand-authored questions (one per company, several per company covering
risk factors, financials, and strategy), each with a known expected source
ticker. Retrieval metrics are computed directly (no LLM needed);
faithfulness/relevance are scored by `llama3.2:3b` acting as an LLM judge —
see [Why not RAGAS](#why-not-ragas) below.

| chunking   | retrieval | precision@5 | hit@5 | MRR  | faithfulness | relevance |
|------------|-----------|:-----------:|:-----:|:----:|:------------:|:---------:|
| fixed_size | dense     | **0.93**    | 1.00  | **1.00** | **4.83** | 4.50 |
| fixed_size | hybrid    | 0.86        | 1.00  | 0.98 | 4.83         | 4.38 |
| semantic   | dense     | 0.91        | 1.00  | 0.98 | 4.67         | 4.46 |
| semantic   | hybrid    | 0.83        | 1.00  | 0.95 | 4.79         | **4.50** |

![Chunking x retrieval comparison chart](eval/comparison_chart.png)

**Winner: fixed_size + dense.** Highest precision@5 and MRR, tied for
highest faithfulness. This is the configuration Phase 5 serving will wrap.

**Findings:**

- **hit@5 = 1.00 across every configuration.** With only 6 companies in the
  corpus and each question tied to one of them, the correct company's
  chunks are essentially always retrievable somewhere in the top 5 — this
  metric doesn't discriminate between configs here. It would matter more on
  a larger, more topically overlapping corpus.
- **Hybrid retrieval hurt precision, on both chunking strategies** (0.93→0.86
  fixed_size, 0.91→0.83 semantic). Concretely: q05 asks about MSFT's
  cybersecurity risk factors, but hybrid's BM25 component pulled in TSLA and
  AAPL chunks (retrieved tickers `[MSFT, TSLA, MSFT, TSLA, AAPL]`,
  precision@5 = 0.4) because those companies' risk-factor sections reuse
  similar generic phrasing ("could materially and adversely affect our
  business") that BM25 weights on lexical overlap alone. This is exactly
  the boilerplate-language noise the corpus was chosen for (see
  [Why these choices](#why-these-choices)) — dense embeddings capture the
  topical difference that BM25's bag-of-words model misses. RRF fusion
  still pulls in the BM25-favored-but-wrong-company chunks often enough to
  measurably hurt precision, without improving recall (hit@5 was already
  1.0 via dense alone).
- **fixed_size slightly beat semantic on precision/MRR**, on both retrieval
  methods. The recursive/semantic chunker keeps chunks aligned to sentence
  and paragraph boundaries, which should help downstream generation
  coherence, but it doesn't clearly outperform the naive sliding window on
  this eval set's precision — plausible cause: 10-K risk-factor sections
  are already short-paragraph-per-risk, so a 500-token fixed window rarely
  crosses a topic boundary either.
- **Generation quality (faithfulness/relevance) barely moves across
  configs** (4.67–4.83 faithfulness, 4.38–4.50 relevance) — with hit@5
  always 1.0, the correct context is present in all 4 configs, so
  generation quality mostly reflects the 3B judge/generator model's own
  ceiling rather than the retrieval config.

### Why not RAGAS

The project spec calls for RAGAS configured against a local Ollama judge, or
a hand-built fallback if that proves unreliable. RAGAS's LLM-wrapper
override path is built around LangChain's chat model interface and expects
fairly reliable structured-output parsing per metric — a lot of moving
parts to get right against a 3B local model with no fallback if it
misbehaves. A ~40-line hand-built judge prompt (see `eval/harness.py`)
asking for a `{"faithfulness": int, "relevance": int}` JSON blob is far
easier to make robust against a small local model, and is the fallback the
spec itself names as acceptable.

## Build phases

- [x] **Phase 1** — Corpus + naive baseline.
- [x] **Phase 2** — Orchestrated ingestion DAG (Dagster, partitioned per ticker) with dbt-style data quality checks at every stage; containerized.
- [x] **Phase 3** — Semantic/recursive chunking + hybrid (dense+BM25 RRF) retrieval, 24-question eval set, measured comparison table + chart.
- [ ] **Phase 4** — Failure analysis (no-answer, ambiguous, multi-document questions) and a confidence-based refusal mechanism.
- [ ] **Phase 5** — FastAPI serving with structured logging, full `docker-compose up` from a clean clone, final README pass.
