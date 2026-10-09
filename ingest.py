"""
Phase 2 — Knowledge Base Ingestion.

Splits each desk's documents into chunks, embeds them, and builds a
per-desk FAISS index. This replaces the trip concierge's static
dictionary lookup with a real retrieval step: the desk agent searches
this index instead of guessing from memory.

Run this once (or whenever the knowledge_base/ documents change):
    python ingest.py
"""

import os
import glob
import pickle
import numpy as np
import faiss

from llm import get_embeddings

KB_DIR = "knowledge_base"
INDEX_DIR = "vector_store"
DESKS = ["academic", "hostel", "examcell", "placement", "fee"]

CHUNK_SIZE = 500     # characters per chunk
CHUNK_OVERLAP = 80


def chunk_text(text: str, source: str):
    """Simple sliding-window chunker — good enough for short circulars.
    Swap for a smarter splitter (e.g. RecursiveCharacterTextSplitter)
    once you're working with real, longer college PDFs."""
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_desk_index(desk: str, embeddings):
    doc_paths = glob.glob(os.path.join(KB_DIR, desk, "*.txt"))
    all_chunks = []
    for path in doc_paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        all_chunks.extend(chunk_text(text, source=os.path.basename(path)))

    if not all_chunks:
        print(f"  [!] No documents found for desk '{desk}' — skipping.")
        return

    vectors = embeddings.embed_documents([c["text"] for c in all_chunks])
    matrix = np.array(vectors, dtype="float32")

    index = faiss.IndexFlatIP(matrix.shape[1])   # cosine similarity via inner product on normalized vectors
    faiss.normalize_L2(matrix)
    index.add(matrix)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, f"{desk}.index"))
    with open(os.path.join(INDEX_DIR, f"{desk}.meta.pkl"), "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"  [ok] {desk}: {len(all_chunks)} chunks indexed")


def build_all_indexes():
    embeddings = get_embeddings()
    print("Building vector stores...")
    for desk in DESKS:
        build_desk_index(desk, embeddings)
    print("Done.")


if __name__ == "__main__":
    build_all_indexes()
