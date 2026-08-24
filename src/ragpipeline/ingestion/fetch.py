"""Download the latest 10-K filing for a handful of companies from SEC EDGAR.

SEC EDGAR is free and requires no API key, but does require a descriptive
User-Agent (name + contact email) on every request, and asks callers to stay
under ~10 requests/second. See https://www.sec.gov/os/accessing-edgar-data.
"""
import time
from pathlib import Path

import requests

from ragpipeline import config

# Ticker -> CIK (SEC's Central Index Key), zero-padded to 10 digits.
# Picked to span sectors (tech, finance, energy, pharma, auto) for topical
# and structural variety in the corpus.
CIK_MAP = {
    "AAPL": "0000320193",  # Apple
    "MSFT": "0000789019",  # Microsoft
    "JPM": "0000019617",  # JPMorgan Chase
    "XOM": "0000034088",  # ExxonMobil
    "PFE": "0000078003",  # Pfizer
    "TSLA": "0001318605",  # Tesla
}

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{doc}"


def _headers() -> dict:
    if not config.SEC_EDGAR_USER_AGENT:
        raise RuntimeError(
            "SEC_EDGAR_USER_AGENT is not set. SEC requires a descriptive "
            "User-Agent with a contact email on every request — set it in .env."
        )
    return {"User-Agent": config.SEC_EDGAR_USER_AGENT}


def _latest_10k(cik: str) -> dict:
    resp = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=_headers(), timeout=30)
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            return {
                "accession": recent["accessionNumber"][i],
                "primary_document": recent["primaryDocument"][i],
                "filing_date": recent["filingDate"][i],
            }
    raise ValueError(f"No 10-K filing found for CIK {cik}")


def fetch_one(ticker: str, out_dir: Path = config.RAW_DIR) -> Path:
    """Fetch a single company's latest 10-K. Used directly by the Dagster
    per-ticker partitioned asset, and looped over by fetch_all below."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cik = CIK_MAP[ticker]
    filing = _latest_10k(cik)
    accession_nodash = filing["accession"].replace("-", "")
    doc_url = DOC_URL.format(cik_int=int(cik), accession_nodash=accession_nodash, doc=filing["primary_document"])

    resp = requests.get(doc_url, headers=_headers(), timeout=60)
    resp.raise_for_status()

    out_path = out_dir / f"{ticker}_10K_{filing['filing_date']}.htm"
    out_path.write_bytes(resp.content)
    print(f"[fetch] {ticker}: 10-K filed {filing['filing_date']} -> {out_path.name} ({len(resp.content):,} bytes)")
    return out_path


def fetch_all(tickers: list[str] | None = None, out_dir: Path = config.RAW_DIR) -> list[Path]:
    tickers = tickers or list(CIK_MAP.keys())
    saved = []
    for ticker in tickers:
        saved.append(fetch_one(ticker, out_dir))
        time.sleep(0.3)  # stay well under SEC's rate limit
    return saved


if __name__ == "__main__":
    fetch_all()
