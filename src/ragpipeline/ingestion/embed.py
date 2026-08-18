"""Thin wrapper around the local Ollama embedding endpoint."""
from ollama import Client

from ragpipeline import config

EMBED_DIM = 768  # nomic-embed-text output dimension
_BATCH_SIZE = 32

_client = Client(host=config.OLLAMA_HOST)


def embed_texts(texts: list[str], batch_size: int = _BATCH_SIZE) -> list[list[float]]:
    """Embed a list of texts, batching requests to Ollama."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = _client.embed(model=config.EMBED_MODEL, input=batch)
        vectors.extend(resp["embeddings"])
    return vectors


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


if __name__ == "__main__":
    vecs = embed_texts(["hello world", "another test sentence"])
    print(f"Embedded {len(vecs)} texts, dim={len(vecs[0])}")
