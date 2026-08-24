# RAG System with Orchestrated Ingestion Pipeline

A retrieval-augmented generation system over SEC 10-K filings, built to compare
retrieval strategies (dense vs. hybrid) on a measured eval set, with ingestion
run as an orchestrated, schedulable pipeline rather than a one-off script.

**Status: Phase 2 complete** (orchestrated ingestion DAG with data quality
checks). Phases 3–5 (retrieval/chunking comparison, failure analysis,
serving) are in progress — see [Build phases](#build-phases) below.

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
Retrieval layer: dense baseline vs. hybrid (dense + BM25)    [Phase 3]
     |
     v
Generation: llama3.2:3b via Ollama (local, no hosted API)
     |
     v
Evaluation harness: hand-built context precision/recall,
faithfulness, answer relevance on a 20-30 question eval set  [Phase 3]
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
    dense.py               # dense-only retrieval baseline
  generation/
    ollama_client.py       # grounded RAG prompt -> llama3.2:3b
orchestration/             # Phase 2: Dagster ingestion DAG
  assets.py                # extract/chunk/embed/load, partitioned per ticker
  checks.py                # dbt-style data quality assertions per stage
  definitions.py           # Definitions object, job, daily schedule
scripts/
  run_naive_pipeline.py    # Phase 1 end-to-end script (--ingest, --query)
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

## Build phases

- [x] **Phase 1** — Corpus + naive baseline.
- [x] **Phase 2** — Orchestrated ingestion DAG (Dagster, partitioned per ticker) with dbt-style data quality checks at every stage; containerized.
- [ ] **Phase 3** — Add semantic/recursive chunking and hybrid (dense+BM25) retrieval, build a 20–30 question eval set, produce a measured comparison table.
- [ ] **Phase 4** — Failure analysis (no-answer, ambiguous, multi-document questions) and a confidence-based refusal mechanism.
- [ ] **Phase 5** — FastAPI serving with structured logging, full `docker-compose up` from a clean clone, final README pass.
