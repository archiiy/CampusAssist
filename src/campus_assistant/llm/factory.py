"""
Single place every module gets its LLM and embedding model from.

`mock` is accepted explicitly by both factories, defaulting to
`config.settings.mock_mode` when omitted. Callers that need to vary
mock-mode per invocation (e.g. the Streamlit UI's sidebar toggle)
should pass `mock=` explicitly rather than mutating a global env var
that this module reads implicitly — that indirection previously meant
Streamlit's `@st.cache_resource` was keyed on a parameter the cached
function body itself ignored.

Both factories are cached per (model, temperature/name, mock) so
repeated node calls within a run reuse one client instance instead of
constructing a fresh ChatOllama/OllamaEmbeddings object every time.
"""

from functools import lru_cache

import numpy as np

from campus_assistant.config import settings


def get_llm(model: str | None = None, temperature: float | None = None, mock: bool | None = None):
    model = model or settings.llm_model
    temperature = settings.llm_temperature if temperature is None else temperature
    mock = settings.mock_mode if mock is None else mock
    return _get_llm_cached(model, temperature, mock)


@lru_cache(maxsize=None)
def _get_llm_cached(model: str, temperature: float, mock: bool):
    if mock:
        from campus_assistant.llm.mock_backends import MockChatModel
        return MockChatModel()

    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, temperature=temperature)


def get_embeddings(model: str | None = None, mock: bool | None = None):
    """Returns an object with .embed_documents(list[str]) and .embed_query(str)."""
    model = model or settings.embedding_model
    mock = settings.mock_mode if mock is None else mock
    return _get_embeddings_cached(model, mock)


@lru_cache(maxsize=None)
def _get_embeddings_cached(model: str, mock: bool):
    if mock:
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
        import hashlib
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
