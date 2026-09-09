from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedChunkOut(BaseModel):
    id: str
    ticker: str
    section: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    top_score: float
    retrieved_chunks: list[RetrievedChunkOut]
    retrieval_time_s: float
    generation_time_s: float
    total_time_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
