"""Lightweight retrieval evaluation.

Extracted from the notebook into a real module on purpose:
- easy to run locally
- easy to wire into CI later
- keeps the notebook focused on exploration
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .config import Settings
from .index import RagIndex


@dataclass
class EvalCase:
    id: str
    question: str
    gold_phrase: str
    gold_sources: List[str]


def load_eval_set(path: str | Path) -> List[EvalCase]:
    p = Path(path)
    cases: List[EvalCase] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cases.append(EvalCase(
            id=obj["id"],
            question=obj["question"],
            gold_phrase=obj["gold_phrase"],
            gold_sources=list(obj.get("gold_sources", [])),
        ))
    return cases


def precision_at_k(hits: List[Dict], gold_sources: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = hits[:k]
    if not top:
        return 0.0
    good = 0
    gold = set(gold_sources)
    for h in top:
        if h.get("source") in gold:
            good += 1
    return good / len(top)


def mrr(hits: List[Dict], gold_sources: List[str]) -> float:
    gold = set(gold_sources)
    for i, h in enumerate(hits, start=1):
        if h.get("source") in gold:
            return 1.0 / i
    return 0.0


def phrase_hit(hits: List[Dict], phrase: str, k: int) -> bool:
    phrase_l = phrase.lower()
    for h in hits[:k]:
        if phrase_l in (h.get("text", "") or "").lower():
            return True
    return False


def run_retrieval_eval(
    settings: Settings,
    eval_set_path: str | Path,
    *,
    k: int = 5,
) -> Dict[str, float]:
    """Run retrieval-only checks (no generation).

    We score two things:
    - does the correct SOURCE show up? (Precision@k, MRR)
    - does the correct PHRASE appear in any top-k chunk? (phrase_recall@k)

    This is intentionally simple: a real system would use labeled passage relevance.
    """
    index = RagIndex(settings)
    index.load(settings.index_dir)

    cases = load_eval_set(eval_set_path)
    if not cases:
        raise ValueError(f"No eval cases found in {eval_set_path}")

    p_at_k = []
    mrrs = []
    phrase_recall = []

    for c in cases:
        hits = index.search(c.question, k=k)
        p_at_k.append(precision_at_k(hits, c.gold_sources, k))
        mrrs.append(mrr(hits, c.gold_sources))
        phrase_recall.append(1.0 if phrase_hit(hits, c.gold_phrase, k) else 0.0)

    return {
        "precision_at_k": sum(p_at_k) / len(p_at_k),
        "mrr": sum(mrrs) / len(mrrs),
        "phrase_recall_at_k": sum(phrase_recall) / len(phrase_recall),
        "num_cases": float(len(cases)),
    }
