"""Hybrid retrieval: dense (Qdrant cosine) + sparse (BM25) combined via
Reciprocal Rank Fusion (RRF). This is Phase 3's comparison variant against
the dense-only baseline in dense.py.

RRF is used instead of normalizing and summing raw scores because cosine
similarity and BM25 scores live on incomparable scales — RRF only needs each
method's *rank order*, which sidesteps that entirely and is the standard
approach for combining dense + sparse results.
"""
import re
from dataclasses import dataclass

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from ragpipeline import config
from ragpipeline.ingestion.embed import embed_text
from ragpipeline.retrieval.dense import RetrievedChunk, strategy_filter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# Per-strategy BM25 index cache: rebuilding it from a full Qdrant scroll on
# every query would dominate eval-harness runtime across dozens of questions.
_bm25_cache: dict[str, tuple[BM25Okapi, list[str], list[dict]]] = {}


def _scroll_all(strategy: str, client: QdrantClient) -> list[tuple[str, dict]]:
    points: list[tuple[str, dict]] = []
    next_offset = None
    flt = strategy_filter(strategy)
    while True:
        batch, next_offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            scroll_filter=flt,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend((str(p.id), p.payload) for p in batch)
        if next_offset is None:
            break
    return points


def _get_bm25_index(strategy: str, client: QdrantClient) -> tuple[BM25Okapi, list[str], list[dict]]:
    if strategy not in _bm25_cache:
        pairs = _scroll_all(strategy, client)
        ids = [pid for pid, _ in pairs]
        payloads = [payload for _, payload in pairs]
        bm25 = BM25Okapi([_tokenize(p["text"]) for p in payloads])
        _bm25_cache[strategy] = (bm25, ids, payloads)
    return _bm25_cache[strategy]


def _rrf_fuse(*ranked_id_lists: list[str], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def hybrid_search(
    query: str,
    top_k: int = 5,
    strategy: str = "fixed_size",
    dense_k: int = 20,
    sparse_k: int = 20,
    client: QdrantClient | None = None,
) -> list[RetrievedChunk]:
    client = client or QdrantClient(url=config.QDRANT_URL)

    dense_hits = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=embed_text(query),
        query_filter=strategy_filter(strategy),
        limit=dense_k,
        with_payload=True,
    ).points
    dense_ids = [str(h.id) for h in dense_hits]
    payload_by_id = {str(h.id): h.payload for h in dense_hits}

    bm25, all_ids, all_payloads = _get_bm25_index(strategy, client)
    bm25_scores = bm25.get_scores(_tokenize(query))
    top_sparse_idx = sorted(range(len(all_ids)), key=lambda i: bm25_scores[i], reverse=True)[:sparse_k]
    sparse_ids = [all_ids[i] for i in top_sparse_idx]
    for i in top_sparse_idx:
        payload_by_id.setdefault(all_ids[i], all_payloads[i])

    fused_scores = _rrf_fuse(dense_ids, sparse_ids)
    top_ids = sorted(fused_scores, key=lambda doc_id: fused_scores[doc_id], reverse=True)[:top_k]

    return [
        RetrievedChunk(
            id=doc_id,
            text=payload_by_id[doc_id]["text"],
            score=fused_scores[doc_id],
            ticker=payload_by_id[doc_id]["ticker"],
            source_file=payload_by_id[doc_id]["source_file"],
            section=payload_by_id[doc_id]["section"],
        )
        for doc_id in top_ids
    ]


if __name__ == "__main__":
    results = hybrid_search("What are the main risk factors related to supply chain?", top_k=3)
    for r in results:
        print(f"[{r.score:.4f}] {r.ticker} / {r.section}\n{r.text[:200]}...\n")
