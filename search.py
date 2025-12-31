from fastapi import APIRouter, HTTPException

from ..schemas import SearchRequest, SearchResponse, ContextHit
from ..services import rag_service

router = APIRouter()

@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    try:
        hits = rag_service.search(req.query, k=req.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "query": req.query,
        "hits": [ContextHit(**h) for h in hits]
    }
