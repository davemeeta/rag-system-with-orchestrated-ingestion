"""Dense-only retrieval baseline: embed the query, search Qdrant by cosine similarity."""
from dataclasses import dataclass

from qdrant_client import QdrantClient

from ragpipeline import config
from ragpipeline.ingestion.embed import embed_text


@dataclass
class RetrievedChunk:
    text: str
    score: float
    ticker: str
    source_file: str
    section: str


def dense_search(query: str, top_k: int = 5, client: QdrantClient | None = None) -> list[RetrievedChunk]:
    client = client or QdrantClient(url=config.QDRANT_URL)
    query_vector = embed_text(query)

    hits = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
    ).points

    return [
        RetrievedChunk(
            text=hit.payload["text"],
            score=hit.score,
            ticker=hit.payload["ticker"],
            source_file=hit.payload["source_file"],
            section=hit.payload["section"],
        )
        for hit in hits
    ]


if __name__ == "__main__":
    results = dense_search("What are the main risk factors related to supply chain?", top_k=3)
    for r in results:
        print(f"[{r.score:.3f}] {r.ticker} / {r.section}\n{r.text[:200]}...\n")
