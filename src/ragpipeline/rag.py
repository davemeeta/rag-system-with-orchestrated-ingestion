"""End-to-end query path for the Phase 3 winning config (fixed_size + dense),
with a confidence-based refusal gate in front of generation.

Phase 4 finding (see README): a single retrieval-score threshold reliably
catches genuinely off-topic questions ("what's the capital of France")
because their top score sits well below every real 10-K question's, but it
does NOT catch out-of-corpus questions that are topically similar to what's
in the corpus (e.g. asking about Netflix's risk factors) -- those score
almost as high as in-corpus questions, because the retrieved chunks are
about the right *topic* (risk factors) even though they're the wrong
*company*. This gate is intentionally scoped to the failure mode it can
actually catch; see the README's Phase 4 section for what it doesn't.
"""
from dataclasses import dataclass

from ragpipeline.generation.ollama_client import generate_answer
from ragpipeline.retrieval.dense import RetrievedChunk, dense_search

# Chosen from the empirical score gap in the Phase 4 failure-analysis run:
# genuinely off-topic questions scored 0.40-0.47, every real 10-K question
# (in-corpus or topically-similar-but-wrong-company) scored 0.63+.
CONFIDENCE_THRESHOLD = 0.55

REFUSAL_MESSAGE = (
    "I don't have enough relevant information in the indexed 10-K filings "
    "(AAPL, MSFT, JPM, XOM, PFE, TSLA) to answer that confidently, so I'd "
    "rather not guess."
)


@dataclass
class RagAnswer:
    question: str
    answer: str
    refused: bool
    top_score: float
    chunks: list[RetrievedChunk]


def answer_question(question: str, top_k: int = 5, threshold: float = CONFIDENCE_THRESHOLD) -> RagAnswer:
    chunks = dense_search(question, top_k=top_k, strategy="fixed_size")
    top_score = chunks[0].score if chunks else 0.0

    if top_score < threshold:
        return RagAnswer(question=question, answer=REFUSAL_MESSAGE, refused=True, top_score=top_score, chunks=chunks)

    answer = generate_answer(question, chunks)
    return RagAnswer(question=question, answer=answer, refused=False, top_score=top_score, chunks=chunks)
