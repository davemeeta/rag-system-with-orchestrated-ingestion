"""dbt-test-style data quality assertions, one per ingestion stage.

Each check runs against the value the corresponding asset just materialized
for a given ticker partition, and reports pass/fail with metadata — the same
idea as a dbt schema test, just expressed as a Dagster asset check instead of
a SQL assertion.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dagster import AssetCheckExecutionContext, AssetCheckResult, AssetIn, asset_check  # noqa: E402
from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: E402

from ragpipeline import config  # noqa: E402
from ragpipeline.ingestion.embed import EMBED_DIM  # noqa: E402
from ragpipeline.ingestion.load import get_client  # noqa: E402

from .assets import chunks, embedded_chunks, extracted_filing, loaded_points, ticker_partitions  # noqa: E402

MIN_EXTRACTED_CHARS = 1000


@asset_check(
    asset=extracted_filing,
    description="Extracted text is non-trivial (row count / not-null style check)",
    partitions_def=ticker_partitions,
)
def extract_min_length(context: AssetCheckExecutionContext, extracted_filing: dict) -> AssetCheckResult:
    n = len(extracted_filing["text"].strip())
    return AssetCheckResult(
        passed=n >= MIN_EXTRACTED_CHARS,
        metadata={"chars": n, "min_required": MIN_EXTRACTED_CHARS},
    )


@asset_check(
    asset=chunks,
    description="No null/empty chunks, and at least one chunk was produced",
    partitions_def=ticker_partitions,
)
def chunks_not_empty(context: AssetCheckExecutionContext, chunks: list[dict]) -> AssetCheckResult:
    blanks = [c for c in chunks if not c["text"].strip()]
    return AssetCheckResult(
        passed=len(chunks) > 0 and len(blanks) == 0,
        metadata={"num_chunks": len(chunks), "num_blank_chunks": len(blanks)},
    )


@asset_check(
    asset=embedded_chunks,
    description="Every embedding has the expected dimension; count matches chunk count",
    additional_ins={"chunks": AssetIn("chunks")},
    partitions_def=ticker_partitions,
)
def embedding_dimension_and_count(
    context: AssetCheckExecutionContext, chunks: list[dict], embedded_chunks: list[dict]
) -> AssetCheckResult:
    bad_dim = [c for c in embedded_chunks if len(c.get("vector", [])) != EMBED_DIM]
    return AssetCheckResult(
        passed=len(embedded_chunks) == len(chunks) and len(bad_dim) == 0,
        metadata={
            "num_vectors": len(embedded_chunks),
            "num_chunks": len(chunks),
            "expected_dim": EMBED_DIM,
            "num_wrong_dim": len(bad_dim),
        },
    )


@asset_check(
    asset=loaded_points,
    description="Row-count reconciliation: points actually present in Qdrant for this ticker match the chunk count",
    additional_ins={"chunks": AssetIn("chunks")},
    partitions_def=ticker_partitions,
)
def load_row_count_reconciliation(
    context: AssetCheckExecutionContext, loaded_points: int, chunks: list[dict]
) -> AssetCheckResult:
    ticker = context.partition_key
    client = get_client()
    actual = client.count(
        collection_name=config.QDRANT_COLLECTION,
        count_filter=Filter(must=[FieldCondition(key="ticker", match=MatchValue(value=ticker))]),
        exact=True,
    ).count

    return AssetCheckResult(
        passed=actual == len(chunks),
        metadata={"expected_points": len(chunks), "actual_points_in_qdrant": actual, "reported_upserted": loaded_points},
    )
