"""Phase 5: FastAPI wrapper around the Phase 4 guarded query path
(fixed_size + dense retrieval, confidence-gated generation), with structured
per-request logging (latency per stage, tokens, retrieved doc IDs).

    uvicorn ragpipeline.serving.main:app --reload
"""
import time
import uuid

from fastapi import FastAPI

from ragpipeline.rag import answer_question
from ragpipeline.serving.logging_config import get_request_logger, log_request
from ragpipeline.serving.schemas import QueryRequest, QueryResponse, RetrievedChunkOut

app = FastAPI(title="RAG Pipeline API", version="0.1.0")
_logger = get_request_logger()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    request_id = str(uuid.uuid4())
    t0 = time.time()

    result = answer_question(request.question, top_k=request.top_k)
    total_time_s = time.time() - t0

    log_request(
        _logger,
        request_id=request_id,
        question=request.question,
        refused=result.refused,
        top_score=round(result.top_score, 4),
        retrieved_doc_ids=[c.id for c in result.chunks],
        retrieved_tickers=[c.ticker for c in result.chunks],
        retrieval_time_s=round(result.retrieval_time_s, 3),
        generation_time_s=round(result.generation_time_s, 3),
        total_time_s=round(total_time_s, 3),
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        refused=result.refused,
        top_score=result.top_score,
        retrieved_chunks=[
            RetrievedChunkOut(id=c.id, ticker=c.ticker, section=c.section, score=c.score) for c in result.chunks
        ],
        retrieval_time_s=result.retrieval_time_s,
        generation_time_s=result.generation_time_s,
        total_time_s=total_time_s,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )
