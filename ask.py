from fastapi import APIRouter, HTTPException

from ..schemas import AskRequest, AskResponse, ContextHit
from ..services import rag_service

router = APIRouter()

@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        hits = rag_service.search(req.question, k=req.k)
        prompt = rag_service.build_prompt(req.question, hits)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "question": req.question,
        "hits": [ContextHit(**h) for h in hits],
        "prompt": prompt,
    }
