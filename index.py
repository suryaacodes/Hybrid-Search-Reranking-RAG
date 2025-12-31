import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from .text_utils import tokenize, normalize_ws
from .chunking import chunk_doc

# Optional heavy deps (we fail loudly with clear message)
try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception as e:  # pragma: no cover
    np = None
    faiss = None
    SentenceTransformer = None
    CrossEncoder = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None

@dataclass
class BuiltIndex:
    chunks: List[Dict[str, Any]]
    bm25: BM25Okapi
    dense_index: Any  # faiss index
    embeddings: Any   # np.ndarray
    embedder: Any
    reranker: Any

class RagIndex:
    def __init__(self, settings):
        self.s = settings
        self._built: BuiltIndex | None = None

    def build(self, docs: List[Dict[str, Any]]) -> None:
        if _IMPORT_ERR is not None:
            raise RuntimeError(
                "Dense/reranker deps failed to import. "
                "Install pinned requirements and restart. Original error: "
                f"{_IMPORT_ERR}"
            )

        chunks: List[Dict[str, Any]] = []
        for d in docs:
            d = {**d, "text": normalize_ws(d["text"])}
            chunks.extend(chunk_doc(d, self.s.chunk_size, self.s.chunk_overlap))

        tokenized = [tokenize(c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized)

        embedder = SentenceTransformer(self.s.embed_model)
        chunk_texts = [c["text"] for c in chunks]
        embs = embedder.encode(chunk_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        embs = np.asarray(embs, dtype="float32")

        dim = embs.shape[1]
        dense_index = faiss.IndexFlatIP(dim)
        dense_index.add(embs)

        reranker = CrossEncoder(self.s.rerank_model)

        self._built = BuiltIndex(
            chunks=chunks,
            bm25=bm25,
            dense_index=dense_index,
            embeddings=embs,
            embedder=embedder,
            reranker=reranker,
        )

    def save(self, index_dir: str) -> None:
        assert self._built is not None, "Index not built"
        p = Path(index_dir)
        p.mkdir(parents=True, exist_ok=True)

        (p / "chunks.jsonl").write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in self._built.chunks),
            encoding="utf-8",
        )

        faiss.write_index(self._built.dense_index, str(p / "faiss.index"))
        np.save(p / "embeddings.npy", self._built.embeddings)

        (p / "meta.json").write_text(json.dumps({
            "embed_model": self.s.embed_model,
            "rerank_model": self.s.rerank_model,
            "chunk_size": self.s.chunk_size,
            "chunk_overlap": self.s.chunk_overlap,
            "bm25_weight": self.s.bm25_weight,
        }, indent=2), encoding="utf-8")

    def load(self, index_dir: str) -> None:
        if _IMPORT_ERR is not None:
            raise RuntimeError(
                "Dense/reranker deps failed to import. "
                "Install pinned requirements and restart. Original error: "
                f"{_IMPORT_ERR}"
            )
        p = Path(index_dir)
        chunks = [json.loads(line) for line in (p / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        tokenized = [tokenize(c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized)

        embedder = SentenceTransformer(self.s.embed_model)
        reranker = CrossEncoder(self.s.rerank_model)

        dense_index = faiss.read_index(str(p / "faiss.index"))
        embs = np.load(p / "embeddings.npy")

        self._built = BuiltIndex(
            chunks=chunks,
            bm25=bm25,
            dense_index=dense_index,
            embeddings=embs,
            embedder=embedder,
            reranker=reranker,
        )

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        assert self._built is not None, "Index not ready"

        q_tokens = tokenize(query)
        bm25_scores = self._built.bm25.get_scores(q_tokens)

        q_emb = self._built.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)
        q_emb = np.asarray(q_emb, dtype="float32")
        dense_scores, dense_ids = self._built.dense_index.search(q_emb, self.s.dense_k)

        dense_map = {int(i): float(s) for i, s in zip(dense_ids[0], dense_scores[0]) if i != -1}

        bm25 = np.asarray(bm25_scores, dtype="float32")
        if bm25.max() > bm25.min():
            bm25_n = (bm25 - bm25.min()) / (bm25.max() - bm25.min())
        else:
            bm25_n = bm25 * 0.0

        dense = np.zeros_like(bm25_n)
        for i, s in dense_map.items():
            dense[i] = s
        if dense.max() > dense.min():
            dense_n = (dense - dense.min()) / (dense.max() - dense.min())
        else:
            dense_n = dense * 0.0

        w = self.s.bm25_weight
        hybrid = (w * bm25_n) + ((1 - w) * dense_n)

        cand_ids = np.argsort(-hybrid)[: max(self.s.bm25_k, self.s.dense_k)].tolist()
        cands = [self._built.chunks[i] for i in cand_ids]

        pairs = [(query, c["text"]) for c in cands]
        rr_scores = self._built.reranker.predict(pairs)
        rr_scores = [float(x) for x in rr_scores]

        reranked = sorted(zip(cands, rr_scores), key=lambda x: x[1], reverse=True)[:k]

        hits = []
        for c, s in reranked:
            hits.append({
                **c,
                "score": s,
                "signals": {"bm25_weight": self.s.bm25_weight, "candidate_pool": len(cands)},
            })
        return hits
