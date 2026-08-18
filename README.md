# RAG System with Orchestrated Ingestion Pipeline

A retrieval-augmented generation system over SEC 10-K filings, built to compare
retrieval strategies (dense vs. hybrid) on a measured eval set, with ingestion
run as an orchestrated, schedulable pipeline rather than a one-off script.

**Status: Phase 1 complete** (naive end-to-end baseline). Phases 2–5
(orchestration, retrieval/chunking comparison, failure analysis, serving) are
in progress — see [Build phases](#build-phases) below.

## Why these choices

**Corpus — SEC 10-K filings** (AAPL, MSFT, JPM, XOM, PFE, TSLA), pulled live
from [SEC EDGAR](https://www.sec.gov/edgar) (public domain, no auth). 10-Ks
were chosen over cleaner corpora (e.g. library docs) because they have real
structural noise: dense legal prose, financial tables, near-duplicate
boilerplate risk-factor language repeated across filers, inconsistent
heading markup, and — specific to inline-XBRL filings — large blocks of
hidden machine-readable data interleaved with the human-readable text (see
`src/ragpipeline/ingestion/extract.py`, which explicitly strips
`<ix:header>`/`<ix:hidden>`/`display:none` elements). That noise is exactly
what should make fixed-size vs. semantic chunking diverge in Phase 3 — a
clean, well-structured corpus wouldn't stress-test chunking strategy the
same way. Financial-document QA is also a realistic enterprise RAG use case.

**Orchestrator — Dagster.** Airflow is more commonly listed in job postings,
but it requires a metadata database, scheduler, and webserver to run
locally — heavy for a solo project. Dagster's asset-based model runs
locally with a single process and is a defensible, modern choice for
a portfolio project built and run by one person.

**Embeddings — `nomic-embed-text` via Ollama.** Since generation already
requires a local Ollama server, using Ollama for embeddings too means one
model server for both stages instead of running a second
sentence-transformers process. It also has an 8k token context, which
gives headroom for variable-length chunks once semantic chunking (Phase 3)
is added. `bge-small-en-v1.5` was considered as a faster CPU-only
alternative and is noted here as the documented tradeoff.

**Generation — `llama3.2:3b` via Ollama.** Small enough to run comfortably
on a laptop CPU while still following instructions well enough to stay
grounded in retrieved context (see [Phase 1 results](#phase-1-results)).

## Architecture (target — see [Build phases](#build-phases) for what's live)

```
SEC EDGAR (10-K filings)
     |
     v
Orchestrated ingestion pipeline (Dagster, containerized)     [Phase 2]
  - extract -> chunk (fixed-size AND semantic) -> embed -> load
  - data quality checks at each stage
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
    fetch.py              # SEC EDGAR downloader
    extract.py             # HTML -> cleaned text, strips hidden XBRL, tags [SECTION] headers
    chunk.py                # fixed-size token chunker (strategy #1)
    embed.py                  # Ollama embedding wrapper (nomic-embed-text)
    load.py                    # Qdrant collection create + upsert, dimension checks
  retrieval/
    dense.py               # dense-only retrieval baseline
  generation/
    ollama_client.py       # grounded RAG prompt -> llama3.2:3b
scripts/
  run_naive_pipeline.py    # Phase 1 end-to-end script (--ingest, --query)
data/
  raw/                    # downloaded 10-K HTML (gitignored)
  processed/              # cleaned text per filing (gitignored)
docker-compose.yml        # Qdrant (api + dagster services added in later phases)
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

## Build phases

- [x] **Phase 1** — Corpus + naive baseline (this README's current state).
- [ ] **Phase 2** — Rebuild ingestion as a Dagster DAG (extract/chunk/embed/load/quality-check as independently retriable tasks), add dbt-style data quality assertions.
- [ ] **Phase 3** — Add semantic/recursive chunking and hybrid (dense+BM25) retrieval, build a 20–30 question eval set, produce a measured comparison table.
- [ ] **Phase 4** — Failure analysis (no-answer, ambiguous, multi-document questions) and a confidence-based refusal mechanism.
- [ ] **Phase 5** — FastAPI serving with structured logging, full `docker-compose up` from a clean clone, final README pass.
