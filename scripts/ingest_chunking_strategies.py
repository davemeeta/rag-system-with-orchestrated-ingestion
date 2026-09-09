"""Phase 3: ingest both chunking strategies (fixed_size, semantic) into the
same Qdrant collection, tagged by the `strategy` payload field, so retrieval
can filter to either one for comparison.

    python scripts/ingest_chunking_strategies.py --recreate
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragpipeline.ingestion.chunk import chunk_all
from ragpipeline.ingestion.load import load_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recreate", action="store_true", help="drop and recreate the collection first")
    args = parser.parse_args()

    recreate = args.recreate
    for strategy in ("fixed_size", "semantic"):
        chunks = chunk_all(strategy=strategy)
        n = load_chunks(chunks, recreate=recreate)
        print(f"[ingest] loaded {n} '{strategy}' chunks\n")
        recreate = False  # only wipe the collection before the first strategy


if __name__ == "__main__":
    main()
