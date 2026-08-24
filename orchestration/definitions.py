"""Dagster entry point: `dagster dev -m orchestration.definitions`.

Wires the per-ticker extract/transform/embed/load assets and their data
quality checks into one Definitions object, plus a job + daily schedule so
the ingestion pipeline is schedulable, not just runnable ad hoc.
"""
from dagster import AssetSelection, Definitions, ScheduleDefinition, define_asset_job

from . import assets, checks

ingest_all_filings_job = define_asset_job(
    name="ingest_all_filings",
    selection=AssetSelection.all(),
    partitions_def=assets.ticker_partitions,
)

# Not expected to actually fire in a demo environment, but demonstrates the
# pipeline is schedulable per the project requirements — re-ingest daily to
# pick up amended filings.
daily_ingest_schedule = ScheduleDefinition(
    name="daily_ingest_schedule",
    job=ingest_all_filings_job,
    cron_schedule="0 6 * * *",
)

defs = Definitions(
    assets=[assets.extracted_filing, assets.chunks, assets.embedded_chunks, assets.loaded_points],
    asset_checks=[
        checks.extract_min_length,
        checks.chunks_not_empty,
        checks.embedding_dimension_and_count,
        checks.load_row_count_reconciliation,
    ],
    jobs=[ingest_all_filings_job],
    schedules=[daily_ingest_schedule],
)
