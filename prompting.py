from typing import Any, Dict, List

def build_prompt_payload(question: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    citations = [{"doc_id": h["doc_id"], "chunk_id": h["chunk_id"], "title": h["title"]} for h in hits]
    context_text = "\n\n".join(
        [f"[{i+1}] {h['title']} ({h['domain']})\n{h['text']}" for i, h in enumerate(hits)]
    )

    system = (
        "You are an internal assistant. Answer using ONLY the provided context. "
        "If the answer isn't in context, say you don't know and ask for the missing policy/runbook."
    )
    user = f"Question: {question}\n\nContext:\n{context_text}\n\nReturn a concise answer + cite sources like [1], [2]."

    return {"system": system, "user": user, "citations": citations}
