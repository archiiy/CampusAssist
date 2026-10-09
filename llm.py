"""
Single place every module gets its LLM and embedding model from.

Set CAMPUS_ASSISTANT_MOCK=1 to run the whole project WITHOUT Ollama
installed, using small deterministic stand-ins. This is only for
exercising the graph wiring / testing logic offline — turn it off (the
default) to actually judge classification, RAG, and reflection quality
with the real qwen2.5:3b model.
"""

import os
import hashlib
import numpy as np


def get_llm(model: str = "qwen2.5:3b", temperature: float = 0.0):
    if os.environ.get("CAMPUS_ASSISTANT_MOCK") == "1":
        from mock_backends import MockChatModel
        return MockChatModel()

    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, temperature=temperature)


def get_embeddings(model: str = "nomic-embed-text"):
    """Returns an object with .embed_documents(list[str]) and .embed_query(str)."""
    if os.environ.get("CAMPUS_ASSISTANT_MOCK") == "1":
        return _MockEmbeddings()

    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=model)


class _MockEmbeddings:
    """
    Deterministic hash-based 'embedding' used only for offline testing.
    It has no real semantic understanding — it just needs to be stable
    and give related short texts a reasonable chance of ranking near
    each other so the retrieval pipeline can be exercised end-to-end
    without an Ollama embedding model.
    """
    DIM = 256

    def _vec(self, text: str) -> np.ndarray:
        words = text.lower().split()
        v = np.zeros(self.DIM, dtype="float32")
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            v[h % self.DIM] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)
