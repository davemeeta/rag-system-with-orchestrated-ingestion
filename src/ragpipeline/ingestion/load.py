"""Create/reset the Qdrant collection and upsert embedded chunks into it."""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ragpipeline import config
from ragpipeline.ingestion.chunk import Chunk
from ragpipeline.ingestion.embed import EMBED_DIM, embed_texts

_UPSERT_BATCH_SIZE = 64


def get_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL)


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    exists = client.collection_exists(config.QDRANT_COLLECTION)
    if exists and recreate:
        client.delete_collection(config.QDRANT_COLLECTION)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )


def load_chunks(chunks: list[Chunk], client: QdrantClient | None = None, recreate: bool = False) -> int:
    client = client or get_client()
    ensure_collection(client, recreate=recreate)

    total = 0
    for i in range(0, len(chunks), _UPSERT_BATCH_SIZE):
        batch = chunks[i : i + _UPSERT_BATCH_SIZE]
        vectors = embed_texts([c.text for c in batch])

        # data-quality check: every vector must match the collection's dimension
        for c, v in zip(batch, vectors):
            if len(v) != EMBED_DIM:
                raise ValueError(f"Embedding for chunk {c.id} has dim {len(v)}, expected {EMBED_DIM}")

        points = [
            PointStruct(
                id=c.id,
                vector=v,
                payload={
                    "ticker": c.ticker,
                    "source_file": c.source_file,
                    "section": c.section,
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                    "strategy": c.strategy,
                },
            )
            for c, v in zip(batch, vectors)
        ]
        client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
        total += len(points)
        print(f"[load] upserted {total}/{len(chunks)} chunks")

    return total


if __name__ == "__main__":
    from ragpipeline.ingestion.chunk import chunk_all

    all_chunks = chunk_all()
    n = load_chunks(all_chunks, recreate=True)
    print(f"Loaded {n} chunks into collection '{config.QDRANT_COLLECTION}'")
