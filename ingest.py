from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from ..schemas import IngestRequest, DocIn
from ..services import rag_service

router = APIRouter()


def _load_default_docs() -> List[Dict[str, Any]]:
    """Load the synthetic-but-messy corpus shipped with the repo."""
    p = Path("data/sample_docs.json")
    if not p.exists():
        raise FileNotFoundError("data/sample_docs.json not found")
    docs = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(docs, list) or not docs:
        raise ValueError("data/sample_docs.json is empty or invalid")
    return docs


@router.post("/ingest")
def ingest(req: Optional[IngestRequest] = None) -> Dict[str, Any]:
    """Build and persist the index.

    - If `docs` is provided, we index those.
    - Otherwise we load `data/sample_docs.json`.
    """
    try:
        if req is None or req.docs is None:
            docs = _load_default_docs()
        else:
            docs = [d.model_dump() for d in req.docs]
        rag_service.build_and_persist(docs)
        return {"ok": True, "docs": len(docs), "index_dir": str(rag_service.index_dir)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
