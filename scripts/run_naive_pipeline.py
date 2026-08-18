"""Phase 1: naive, un-orchestrated end-to-end RAG pipeline.

    python scripts/run_naive_pipeline.py --ingest
    python scripts/run_naive_pipeline.py --query "What are Tesla's main risk factors?"

This proves the concept (fetch -> extract -> chunk -> embed -> index ->
dense retrieve -> generate) before Phase 2 rebuilds ingestion as a Dagster DAG.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragpipeline import config
from ragpipeline.generation.ollama_client import generate_answer
from ragpipeline.ingestion.chunk import chunk_all
from ragpipeline.ingestion.extract import extract_all
from ragpipeline.ingestion.fetch import fetch_all
from ragpipeline.ingestion.load import load_chunks
from ragpipeline.retrieval.dense import dense_search


def run_ingest(recreate: bool = True) -> None:
    t0 = time.time()

    print("=== fetch ===")
    fetch_all()

    print("\n=== extract ===")
    extract_all()

    print("\n=== chunk ===")
    chunks = chunk_all()

    print("\n=== embed + load ===")
    n = load_chunks(chunks, recreate=recreate)

    print(f"\nIngest complete: {n} chunks indexed into '{config.QDRANT_COLLECTION}' in {time.time() - t0:.1f}s")


def run_query(question: str, top_k: int = 5) -> None:
    t0 = time.time()
    chunks = dense_search(question, top_k=top_k)

    print(f"Retrieved {len(chunks)} chunks in {time.time() - t0:.2f}s:")
    for c in chunks:
        print(f"  [{c.score:.3f}] {c.ticker} / {c.section.strip()}")

    print("\nGenerating answer...\n")
    t1 = time.time()
    answer = generate_answer(question, chunks)
    print(answer)
    print(f"\n(generation took {time.time() - t1:.1f}s, total {time.time() - t0:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingest", action="store_true", help="Run fetch->extract->chunk->embed->load")
    parser.add_argument("--query", type=str, help="Ask a question against the indexed corpus")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not args.ingest and not args.query:
        parser.error("pass --ingest and/or --query \"...\"")

    if args.ingest:
        run_ingest()

    if args.query:
        if args.ingest:
            print()
        run_query(args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()
