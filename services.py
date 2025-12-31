from pathlib import Path
from typing import Any, Dict, List

from .core.config import settings
from .core.index import RagIndex
from .core.prompting import build_prompt_payload

class RagService:
    def __init__(self):
        self.index = RagIndex(settings)
        self.index_dir = settings.index_dir

    def ensure_loaded(self) -> None:
        p = Path(self.index_dir)
        if (p / "chunks.jsonl").exists():
            self.index.load(self.index_dir)

    def build_and_persist(self, docs: List[Dict[str, Any]]) -> None:
        self.index.build(docs)
        self.index.save(self.index_dir)

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        return self.index.search(query, k=k)

    def build_prompt(self, question: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
        return build_prompt_payload(question, hits)

rag_service = RagService()
