"""
Text chunking for knowledge-base ingestion.

Uses LangChain's RecursiveCharacterTextSplitter (paragraph/sentence-aware)
instead of the original naive fixed-size sliding window, so chunk
boundaries respect natural text structure. Deliberately has no
FAISS/Ollama imports — pure string logic, so it's directly unit-testable
without a vector store or LLM.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from campus_assistant.config import settings


def chunk_text(text: str, source: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[dict]:
    chunk_size = settings.chunk_size if chunk_size is None else chunk_size
    chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap

    text = text.strip()
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = [p.strip() for p in splitter.split_text(text)]
    return [{"text": p, "source": source} for p in pieces if p]
