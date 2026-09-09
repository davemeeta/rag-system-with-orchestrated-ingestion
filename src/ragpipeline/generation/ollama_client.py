"""Build a grounded RAG prompt and call the local Ollama chat model."""
from dataclasses import dataclass

from ollama import Client

from ragpipeline import config
from ragpipeline.retrieval.dense import RetrievedChunk

_client = Client(host=config.OLLAMA_HOST)


@dataclass
class GenerationResult:
    answer: str
    prompt_tokens: int | None
    completion_tokens: int | None

_SYSTEM_PROMPT = (
    "You are a financial research assistant. Answer the user's question using "
    "ONLY the provided context excerpts from SEC 10-K filings. Cite the ticker "
    "for any fact you use (e.g. 'AAPL 10-K'). If the context does not contain "
    "enough information to answer, say so explicitly instead of guessing."
)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] ({c.ticker}, {c.section})\n{c.text}")
    return "\n\n".join(parts)


def generate_answer_detailed(question: str, chunks: list[RetrievedChunk]) -> GenerationResult:
    context = _format_context(chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    resp = _client.chat(
        model=config.GEN_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return GenerationResult(
        answer=resp["message"]["content"],
        prompt_tokens=resp.get("prompt_eval_count"),
        completion_tokens=resp.get("eval_count"),
    )


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """Back-compat wrapper for callers (eval harness, Phase 1 script) that
    only need the answer text, not token counts."""
    return generate_answer_detailed(question, chunks).answer
