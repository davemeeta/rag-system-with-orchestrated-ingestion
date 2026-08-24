"""Ingestion DAG: extract -> transform (chunk) -> embed -> load, partitioned
per company so each filing is independently materializable/retriable.

Reuses the same functions the Phase 1 naive script calls
(ragpipeline.ingestion.*) — the DAG doesn't reimplement ingestion logic, it
just orchestrates it per-partition and adds data quality checks (checks.py).
"""
import sys
from dataclasses import asdict
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dagster import AssetExecutionContext, StaticPartitionsDefinition, asset  # noqa: E402
from qdrant_client.models import PointStruct  # noqa: E402

from ragpipeline import config  # noqa: E402
from ragpipeline.ingestion.chunk import chunk_document  # noqa: E402
from ragpipeline.ingestion.embed import EMBED_DIM, embed_texts  # noqa: E402
from ragpipeline.ingestion.extract import extract_one  # noqa: E402
from ragpipeline.ingestion.fetch import CIK_MAP, fetch_one  # noqa: E402
from ragpipeline.ingestion.load import ensure_collection, get_client  # noqa: E402

TICKERS = list(CIK_MAP.keys())
ticker_partitions = StaticPartitionsDefinition(TICKERS)

_PARTITION_KWARGS = dict(partitions_def=ticker_partitions)


@asset(**_PARTITION_KWARGS, group_name="extract")
def extracted_filing(context: AssetExecutionContext) -> dict:
    ticker = context.partition_key
    html_path = fetch_one(ticker, config.RAW_DIR)
    text_path = extract_one(html_path, config.PROCESSED_DIR)
    text = text_path.read_text()

    context.add_output_metadata({"ticker": ticker, "source_file": text_path.name, "chars": len(text)})
    return {"ticker": ticker, "source_file": text_path.name, "text": text}


@asset(**_PARTITION_KWARGS, group_name="transform")
def chunks(context: AssetExecutionContext, extracted_filing: dict) -> list[dict]:
    doc_chunks = chunk_document(
        extracted_filing["text"],
        ticker=extracted_filing["ticker"],
        source_file=extracted_filing["source_file"],
    )
    result = [asdict(c) for c in doc_chunks]

    context.add_output_metadata({"num_chunks": len(result)})
    return result


@asset(**_PARTITION_KWARGS, group_name="embed")
def embedded_chunks(context: AssetExecutionContext, chunks: list[dict]) -> list[dict]:
    vectors = embed_texts([c["text"] for c in chunks])
    for c, v in zip(chunks, vectors):
        c["vector"] = v

    context.add_output_metadata({"num_vectors": len(vectors), "dim": len(vectors[0]) if vectors else 0})
    return chunks


@asset(**_PARTITION_KWARGS, group_name="load")
def loaded_points(context: AssetExecutionContext, embedded_chunks: list[dict]) -> int:
    client = get_client()
    ensure_collection(client)  # create-if-missing; never wipes other partitions

    points = [
        PointStruct(
            id=c["id"],
            vector=c["vector"],
            payload={
                "ticker": c["ticker"],
                "source_file": c["source_file"],
                "section": c["section"],
                "chunk_index": c["chunk_index"],
                "text": c["text"],
                "strategy": c["strategy"],
            },
        )
        for c in embedded_chunks
    ]
    client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)

    context.add_output_metadata({"points_upserted": len(points)})
    return len(points)
