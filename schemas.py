from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class DocIn(BaseModel):
    doc_id: str
    title: str
    domain: str
    text: str
    source: Optional[str] = None
    updated_at: Optional[str] = None
    version: Optional[str] = None


class IngestRequest(BaseModel):
    # If omitted, the API will load `data/sample_docs.json`.
    docs: Optional[List[DocIn]] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 8


class AskRequest(BaseModel):
    question: str
    k: int = 5


class ContextHit(BaseModel):
    doc_id: str
    title: str
    domain: str
    source: Optional[str] = None
    updated_at: Optional[str] = None
    text: str
    score: float
    signals: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    query: str
    hits: List[ContextHit]


class AskResponse(BaseModel):
    question: str
    hits: List[ContextHit]
    prompt: Dict[str, Any]
