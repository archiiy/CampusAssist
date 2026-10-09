"""
Knowledge Base Ingestion.

Loads each desk's documents (.txt and .pdf), splits them into chunks,
embeds them, and builds a per-desk FAISS index. Run this once (or
whenever knowledge_base/ documents change):

    python scripts/ingest.py
"""

import glob
import os
import pickle

import faiss
import numpy as np
from pypdf import PdfReader

from campus_assistant.config import settings
from campus_assistant.llm.factory import get_embeddings
from campus_assistant.rag.chunking import chunk_text

DESKS = ["academic", "hostel", "examcell", "placement", "fee"]


def _load_txt(path: str) -> list[tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return [(f.read(), os.path.basename(path))]


def _load_pdf(path: str) -> list[tuple[str, str]]:
    reader = PdfReader(path)
    filename = os.path.basename(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            pages.append((page_text, f"{filename} (page {i})"))
    return pages


def load_documents(desk: str) -> list[dict]:
    """Returns [{"text": ..., "source": ...}, ...] for every chunk across
    every .txt/.pdf document found in knowledge_base/<desk>/."""
    desk_dir = settings.kb_dir / desk
    paths = sorted(glob.glob(str(desk_dir / "*.txt"))) + sorted(glob.glob(str(desk_dir / "*.pdf")))

    all_chunks = []
    for path in paths:
        loader = _load_pdf if path.lower().endswith(".pdf") else _load_txt
        for text, source in loader(path):
            all_chunks.extend(chunk_text(text, source=source))
    return all_chunks


def build_desk_index(desk: str, embeddings):
    all_chunks = load_documents(desk)

    if not all_chunks:
        print(f"  [!] No documents found for desk '{desk}' — skipping.")
        return

    vectors = embeddings.embed_documents([c["text"] for c in all_chunks])
    matrix = np.array(vectors, dtype="float32")

    index = faiss.IndexFlatIP(matrix.shape[1])   # cosine similarity via inner product on normalized vectors
    faiss.normalize_L2(matrix)
    index.add(matrix)

    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(settings.vector_store_dir / f"{desk}.index"))
    with open(settings.vector_store_dir / f"{desk}.meta.pkl", "wb") as f:
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
