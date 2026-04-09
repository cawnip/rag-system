from pydantic import BaseModel
from typing import List


class UploadResponse(BaseModel):
    message: str
    files_processed: List[str]
    total_chunks: int


class AskRequest(BaseModel):
    question: str


class Citation(BaseModel):
    source: str
    page: int


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
