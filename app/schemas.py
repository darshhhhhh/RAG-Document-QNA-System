from typing import Any

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    file_name: str
    page_number: int
    chunk_number: int
    chunk_id: str
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: str
    best_score: float
    answer_type: str
    sources: list[Source]
    retrieved_context: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    message: str


class UploadResponse(BaseModel):
    status: str
    message: str
    file_name: str
    total_pages: int
    total_chunks: int
    inserted_chunks: int
    embedding_model: str
    embedding_dimension: int
    vector_db_count: int


class ResetResponse(BaseModel):
    status: str
    message: str
    vector_db_count: int