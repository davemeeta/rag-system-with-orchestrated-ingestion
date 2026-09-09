"""Two chunking strategies, compared in Phase 3:

- "fixed_size": slides a fixed-size token window with overlap over each
  section's text, with no regard for sentence/paragraph boundaries — a chunk
  can start or end mid-sentence.
- "semantic": recursively splits each section on paragraph, then line, then
  sentence, then word boundaries, merging pieces back up to the token budget.
  This keeps chunks aligned to natural text boundaries instead of arbitrary
  token offsets. ("Semantic" here means structure-aware/recursive splitting,
  not embedding-similarity-based segmentation — the common looser usage of
  the term, e.g. LangChain's RecursiveCharacterTextSplitter.)

Both strategies split each document into per-section runs first, using the
[SECTION] markers left by extract.py.
"""
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from ragpipeline import config

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
MIN_CHUNK_TOKENS = 15  # drops table-of-contents/page-number fragments

_ENCODING = tiktoken.get_encoding("cl100k_base")
_SECTION_MARKER_RE = re.compile(r"^\[SECTION\]\s*(.*)$")

# Namespace for deterministic chunk IDs: re-materializing the same
# (ticker, source_file, strategy, chunk_index) always yields the same ID, so
# re-running ingestion overwrites the existing Qdrant point instead of
# inserting a duplicate. This is what makes the Dagster load asset idempotent.
_ID_NAMESPACE = uuid.UUID("7f6f6e8a-2f6f-4a3f-9a3f-6f6f6e8a2f6f")


def _chunk_id(ticker: str, source_file: str, strategy: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, f"{ticker}:{source_file}:{strategy}:{chunk_index}"))


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


def _fixed_size_blocks(block_text: str, chunk_size: int, overlap: int) -> list[str]:
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


# Priority order for recursive splitting: try to break on paragraph, then
# line, then sentence, then word boundaries before falling back to a hard
# token cutoff.
_SEMANTIC_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(_ENCODING.encode(text)) <= chunk_size:
        return [text]
    if not separators:
        tokens = _ENCODING.encode(text)
        return [_ENCODING.decode(tokens[:chunk_size]).strip()]

    sep, *rest_seps = separators
    parts = [p for p in text.split(sep) if p.strip()]
    if len(parts) <= 1:
        return _split_recursive(text, chunk_size, rest_seps)

    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{sep}{part}" if current else part
        if len(_ENCODING.encode(candidate)) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(_ENCODING.encode(part)) > chunk_size:
            chunks.extend(_split_recursive(part, chunk_size, rest_seps))
            current = ""
        else:
            current = part
    if current:
        chunks.append(current)
    return chunks


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Prepend a tail slice of the previous chunk to each chunk, so adjacent
    semantic chunks share context the same way fixed_size's sliding window
    does — without this, semantic chunk boundaries would have zero overlap."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tokens = _ENCODING.encode(chunks[i - 1])
        tail = _ENCODING.decode(prev_tokens[-overlap:]) if len(prev_tokens) > overlap else chunks[i - 1]
        result.append(f"{tail}\n{chunks[i]}".strip())
    return result


def _semantic_blocks(block_text: str, chunk_size: int, overlap: int) -> list[str]:
    return _add_overlap(_split_recursive(block_text, chunk_size, _SEMANTIC_SEPARATORS), overlap)


_STRATEGIES = {
    "fixed_size": _fixed_size_blocks,
    "semantic": _semantic_blocks,
}


def chunk_document(
    text: str,
    ticker: str,
    source_file: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
    strategy: str = "fixed_size",
) -> list[Chunk]:
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}. Choices: {list(_STRATEGIES)}")
    split_fn = _STRATEGIES[strategy]

    chunks: list[Chunk] = []
    idx = 0
    for section, block_text in _section_blocks(text):
        for chunk_text in split_fn(block_text, chunk_size, overlap):
            if not chunk_text.strip():
                continue  # data-quality: drop empty/whitespace-only chunks
            if len(_ENCODING.encode(chunk_text)) < MIN_CHUNK_TOKENS:
                continue  # data-quality: drop near-empty fragments (TOC lines, page numbers)
            chunks.append(
                Chunk(
                    id=_chunk_id(ticker, source_file, strategy, idx),
                    ticker=ticker,
                    source_file=source_file,
                    section=section,
                    chunk_index=idx,
                    text=chunk_text,
                    strategy=strategy,
                )
            )
            idx += 1
    return chunks


def chunk_all(processed_dir: Path = config.PROCESSED_DIR, strategy: str = "fixed_size") -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for txt_path in sorted(processed_dir.glob("*.txt")):
        ticker = txt_path.stem.split("_10K_")[0]
        text = txt_path.read_text()
        doc_chunks = chunk_document(text, ticker=ticker, source_file=txt_path.name, strategy=strategy)
        all_chunks.extend(doc_chunks)
        print(f"[chunk:{strategy}] {txt_path.name}: {len(doc_chunks)} chunks")
    return all_chunks


if __name__ == "__main__":
    import sys

    strat = sys.argv[1] if len(sys.argv) > 1 else "fixed_size"
    result = chunk_all(strategy=strat)
    print(f"Total chunks ({strat}): {len(result)}")
