from pydantic import BaseModel, Field

class Settings(BaseModel):
    # Models
    embed_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    rerank_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Retrieval knobs (tune these, don’t hardcode in random places)
    chunk_size: int = 650
    chunk_overlap: int = 120
    bm25_weight: float = 0.35
    dense_k: int = 25
    bm25_k: int = 40
    rerank_k: int = 8

    # Storage (simple local persistence for portfolio)
    data_dir: str = "data"
    index_dir: str = "data/index"

settings = Settings()
