"""
Shared test fixtures.

Forces offline mock mode and in-memory checkpointing so the full suite
runs without a live Ollama server and without leaving .sqlite files
behind. Also points the vector store at a disposable temp directory
built fresh with mock embeddings (256-dim) — the committed
vector_store/ in the project root is built with real Ollama embeddings
(768-dim, nomic-embed-text) for actual deployment use, so tests must
not share it: querying a 768-dim index with a 256-dim mock embedding
(or vice versa) raises a FAISS dimension-mismatch error.
"""

import atexit
import os
import shutil
import tempfile
from pathlib import Path

os.environ["CAMPUS_ASSISTANT_MOCK"] = "1"
os.environ["CAMPUS_ASSISTANT_CHECKPOINTER_BACKEND"] = "memory"

_tmp_vector_store = Path(tempfile.mkdtemp(prefix="campus_assistant_test_vs_"))
os.environ["CAMPUS_ASSISTANT_VECTOR_STORE_DIR"] = str(_tmp_vector_store)
atexit.register(shutil.rmtree, _tmp_vector_store, True)

from campus_assistant.rag.ingest import build_all_indexes  # noqa: E402

build_all_indexes()
