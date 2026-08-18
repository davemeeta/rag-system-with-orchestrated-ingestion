"""Turn raw 10-K HTML into cleaned plain text, with best-effort section tagging.

10-K markup is inconsistent across filers (inline XBRL, arbitrary <span>/<div>
nesting, headers sometimes split across elements), so section detection here
is best-effort regex over the flattened text, not a structural HTML parse.
That noise is expected and is part of why this corpus was chosen.
"""
import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from ragpipeline import config

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Matches lines like "Item 1A. Risk Factors" / "ITEM 7 - MANAGEMENT'S DISCUSSION..."
ITEM_HEADER_RE = re.compile(r"^\s*item\s+\d+[a-c]?\.?\s*[-–—]?\s*\S", re.IGNORECASE)

_HIDDEN_STYLE_RE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden")


def _is_hidden(tag) -> bool:
    style = tag.get("style")
    return bool(style and _HIDDEN_STYLE_RE.search(style))


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # Inline-XBRL filings carry a large block of machine-readable "facts"
    # (financial figures, dates, member tags) that isn't meant to be read as
    # prose. It lives in <ix:header>/<ix:hidden> and display:none elements —
    # strip those, or their text bleeds into the corpus as noise.
    for tag in soup(["script", "style", "head", "title", "ix:header", "ix:hidden"]):
        tag.decompose()
    for tag in soup.find_all(style=True):
        if _is_hidden(tag):
            tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def tag_sections(text: str) -> str:
    """Prefix likely 'Item N...' section headers with a [SECTION] marker so
    the chunker can attach a best-guess section label to each chunk."""
    out_lines = []
    for line in text.splitlines():
        if len(line) < 100 and ITEM_HEADER_RE.match(line):
            out_lines.append(f"[SECTION] {line.strip()}")
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def extract_all(raw_dir: Path = config.RAW_DIR, out_dir: Path = config.PROCESSED_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for html_path in sorted(raw_dir.glob("*.htm*")):
        raw_html = html_path.read_text(errors="ignore")
        text = tag_sections(html_to_text(raw_html))
        out_path = out_dir / (html_path.stem + ".txt")
        out_path.write_text(text)
        saved.append(out_path)
        print(f"[extract] {html_path.name} -> {out_path.name} ({len(text):,} chars)")
    return saved


if __name__ == "__main__":
    extract_all()
