"""Text embeddings: a multilingual sentence-transformers model, plus a tiny one for tests."""

import hashlib
import logging
import re
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

log = logging.getLogger(__name__)


def pick_device(device: str) -> str:
    """ "auto" is the NVIDIA GPU when there is one (tens of times faster at indexing), else the
    CPU. Apple's MPS is left out: small batches there are often slower than the CPU."""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # a broken CUDA setup must not stop indexing
        return "cpu"


class Embedder(Protocol):
    name: str

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Loads the model on first use; downloads it from Hugging Face into cache_dir once."""

    def __init__(
        self,
        name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: Path | None = None,
        hf_token: str | None = None,
        device: str = "auto",
    ):
        self.name = name
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.device = device
        self._model = None
        self._lock = threading.Lock()
        # E5 models were trained with these prefixes and lose accuracy without them.
        self._e5 = "e5" in name.lower()

    @property
    def model(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                device = pick_device(self.device)
                options = {
                    "device": device,
                    "cache_folder": str(self.cache_dir) if self.cache_dir else None,
                }
                try:
                    # Once downloaded, load without asking the Hub for updates: faster, works
                    # offline and avoids the "unauthenticated requests" warning.
                    self._model = SentenceTransformer(self.name, local_files_only=True, **options)
                except OSError:
                    self._model = SentenceTransformer(self.name, token=self.hf_token, **options)
                log.info("Embedding model %s loaded on %s", self.name, device)
            return self._model

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self.model
        # A GPU is only fast with big batches; on the CPU they gain little and use more memory.
        batch_size = 32 if model.device.type == "cpu" else 256
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True)

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
