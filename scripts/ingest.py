"""
CLI wrapper for knowledge-base ingestion.

    python scripts/ingest.py
"""

from campus_assistant.rag.ingest import build_all_indexes

if __name__ == "__main__":
    build_all_indexes()
