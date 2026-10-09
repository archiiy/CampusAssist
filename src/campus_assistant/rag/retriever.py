"""
Retrieval helper — given a desk name and a query, returns the top-k
most relevant chunks from that desk's FAISS index (built by ingest.py).
"""

import pickle

import faiss
import numpy as np

from campus_assistant.config import settings
from campus_assistant.llm.factory import get_embeddings

_cache = {}   # desk -> (faiss_index, chunks) — loaded once per process


def _load_desk_index(desk: str):
    if desk in _cache:
        return _cache[desk]

    index_path = settings.vector_store_dir / f"{desk}.index"
    meta_path = settings.vector_store_dir / f"{desk}.meta.pkl"
    if not (index_path.exists() and meta_path.exists()):
        raise FileNotFoundError(
            f"No vector store found for desk '{desk}'. Run `python scripts/ingest.py` first."
        )

    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)

    _cache[desk] = (index, chunks)
    return index, chunks


def retrieve(desk: str, query: str, top_k: int = 3):
    index, chunks = _load_desk_index(desk)
    embeddings = get_embeddings()

    vec = np.array([embeddings.embed_query(query)], dtype="float32")
    faiss.normalize_L2(vec)

    scores, indices = index.search(vec, min(top_k, len(chunks)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append({"text": chunk["text"], "source": chunk["source"], "score": float(score)})
    return results


if __name__ == "__main__":
    import sys
    desk = sys.argv[1] if len(sys.argv) > 1 else "hostel"
    query = sys.argv[2] if len(sys.argv) > 2 else "when is the hostel fee due"
    for r in retrieve(desk, query):
        print(f"[{r['score']:.3f}] ({r['source']}) {r['text'][:150]}...")
