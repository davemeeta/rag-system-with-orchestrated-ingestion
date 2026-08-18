"""Fixed-size token chunker (Phase 1 baseline strategy).

Splits each document into per-section runs (using the [SECTION] markers left
by extract.py), then slides a fixed-size token window with overlap over each
section's text. A second, semantic/recursive chunking strategy is added in
Phase 3 for comparison against this one.
"""
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from ragpipeline import config

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

_ENCODING = tiktoken.get_encoding("cl100k_base")
_SECTION_MARKER_RE = re.compile(r"^\[SECTION\]\s*(.*)$")


@dataclass
class Chunk:
    id: str
    ticker: str
    source_file: str
    section: str
    chunk_index: int
    text: str
    strategy: str = "fixed_size"


def _section_blocks(text: str) -> list[tuple[str, str]]:
    """Split doc text into (section_label, block_text) runs based on
    [SECTION] markers."""
    blocks: list[tuple[str, str]] = []
    current_section = "unknown"
    current_lines: list[str] = []

    def flush():
        if current_lines:
            blocks.append((current_section, "\n".join(current_lines)))

    for line in text.splitlines():
        m = _SECTION_MARKER_RE.match(line)
        if m:
            flush()
            current_section = m.group(1).strip() or current_section
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return blocks


def _chunk_block(block_text: str, chunk_size: int, overlap: int) -> list[str]:
    tokens = _ENCODING.encode(block_text)
    if not tokens:
        return []
    step = max(chunk_size - overlap, 1)
    chunks = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        text = _ENCODING.decode(window).strip()
        if text:
            chunks.append(text)
        if start + chunk_size >= len(tokens):
            break
    return chunks


def chunk_document(
    text: str,
    ticker: str,
    source_file: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for section, block_text in _section_blocks(text):
        for chunk_text in _chunk_block(block_text, chunk_size, overlap):
            if not chunk_text.strip():
                continue  # data-quality: drop empty/whitespace-only chunks
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    ticker=ticker,
                    source_file=source_file,
                    section=section,
                    chunk_index=idx,
                    text=chunk_text,
                )
            )
            idx += 1
    return chunks


def chunk_all(processed_dir: Path = config.PROCESSED_DIR) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for txt_path in sorted(processed_dir.glob("*.txt")):
        ticker = txt_path.stem.split("_10K_")[0]
        text = txt_path.read_text()
        doc_chunks = chunk_document(text, ticker=ticker, source_file=txt_path.name)
        all_chunks.extend(doc_chunks)
        print(f"[chunk] {txt_path.name}: {len(doc_chunks)} chunks")
    return all_chunks


if __name__ == "__main__":
    result = chunk_all()
    print(f"Total chunks: {len(result)}")
