"""Text embeddings: a multilingual sentence-transformers model, plus a tiny one for tests."""

import hashlib
import re
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


class Embedder(Protocol):
    name: str

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Loads the model on first use; downloads it from Hugging Face into cache_dir once."""

    def __init__(self, name: str = DEFAULT_EMBEDDING_MODEL, cache_dir: Path | None = None):
        self.name = name
        self.cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()
        # E5 models were trained with these prefixes and lose accuracy without them.
        self._e5 = "e5" in name.lower()

    @property
    def model(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                folder = str(self.cache_dir) if self.cache_dir else None
                self._model = SentenceTransformer(self.name, device="cpu", cache_folder=folder)
            return self._model

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"passage: {t}" if self._e5 else t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([f"query: {text}" if self._e5 else text])[0]


class HashEmbedder:
    """Bag-of-words hashed into a fixed vector. No downloads; only good enough for tests."""

    name = "hash"
    dim = 256

    def _one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._one(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._one(text)
