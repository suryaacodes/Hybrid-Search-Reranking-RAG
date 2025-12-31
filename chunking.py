from typing import Dict, List, Any

def chunk_doc(doc: Dict[str, Any], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    text = doc["text"]
    # Simple token-ish chunking (works fine for portfolio); swap for semantic chunking later.
    words = text.split()
    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()
        chunks.append({
            "chunk_id": f"{doc['doc_id']}::c{idx}",
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "domain": doc.get("domain", "Unknown"),
            "text": chunk_text,
            "start_word": start,
            "end_word": end,
            "source": doc.get("source"),
            "updated_at": doc.get("updated_at"),
        })
        idx += 1
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks
