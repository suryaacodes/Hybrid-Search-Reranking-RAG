from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import ingest, search, ask
from .services import rag_service

app = FastAPI(title="Enterprise RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup():
    rag_service.ensure_loaded()

app.include_router(ingest.router, tags=["ingest"])
app.include_router(search.router, tags=["search"])
app.include_router(ask.router, tags=["ask"])

@app.get("/healthz")
def healthz():
    return {"ok": True}
