"""Knowledge base: collections of documents, split into chunks and indexed for vector search.

Everything lives in one SQLite file; vectors are stored with the sqlite-vec extension.
"""

import hashlib
import re
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from engine.rag.chunking import chunk_text
from engine.rag.embeddings import Embedder

COLLECTION_NAME = re.compile(r"^[\w-]{1,64}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS collections (name TEXT PRIMARY KEY, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    collection TEXT NOT NULL REFERENCES collections(name) ON DELETE CASCADE,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    chars INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (collection, sha256)
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    text TEXT NOT NULL
);
"""


class CollectionNotFound(Exception):
    pass


class InvalidName(Exception):
    pass


@dataclass
class Hit:
    chunk_id: int
    document_id: int
    document: str
    source: str
    text: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class KnowledgeStore:
    def __init__(self, path: Path, embedder: Embedder):
        import sqlite_vec

        path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self._lock = threading.Lock()
        self._vectors_ready = False

    def _vectors(self) -> None:
        """Creates the vector table on first use, sized for the embedding model."""
        if self._vectors_ready:
            return
        meta = dict(self.db.execute("SELECT key, value FROM meta").fetchall())
        if "embedding_model" in meta and meta["embedding_model"] != self.embedder.name:
            raise RuntimeError(
                f"The knowledge base was built with {meta['embedding_model']}, "
                f"not {self.embedder.name}. Use that model or start a new data folder."
            )
        dim = self.embedder.dim
        self.db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0("
            "chunk_id INTEGER PRIMARY KEY, collection TEXT PARTITION KEY, "
            f"embedding FLOAT[{dim}] distance_metric=cosine)"
        )
        self.db.execute(
            "INSERT OR IGNORE INTO meta VALUES ('embedding_model', ?)", (self.embedder.name,)
        )
        self.db.commit()
        self._vectors_ready = True

    def _require(self, collection: str) -> None:
        row = self.db.execute("SELECT 1 FROM collections WHERE name = ?", (collection,))
        if row.fetchone() is None:
            raise CollectionNotFound(collection)

    def collections(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT c.name, c.created_at, COUNT(DISTINCT d.id) AS documents, "
            "COUNT(k.id) AS chunks FROM collections c "
            "LEFT JOIN documents d ON d.collection = c.name "
            "LEFT JOIN chunks k ON k.document_id = d.id GROUP BY c.name ORDER BY c.name"
        )
        return [dict(r) for r in rows]

    def create_collection(self, name: str) -> dict:
        if not COLLECTION_NAME.match(name):
            raise InvalidName("Use up to 64 letters, digits, '_' or '-'.")
        with self._lock:
            self.db.execute("INSERT OR IGNORE INTO collections VALUES (?, ?)", (name, now()))
            self.db.commit()
        return next(c for c in self.collections() if c["name"] == name)

    def delete_collection(self, name: str) -> None:
        with self._lock:
            self._require(name)
            if self._has_vectors():
                self.db.execute("DELETE FROM chunk_vectors WHERE collection = ?", (name,))
            self.db.execute("DELETE FROM collections WHERE name = ?", (name,))
            self.db.commit()

    def documents(self, collection: str) -> list[dict]:
        self._require(collection)
        rows = self.db.execute(
            "SELECT d.id, d.name, d.source, d.chars, d.created_at, COUNT(k.id) AS chunks "
            "FROM documents d LEFT JOIN chunks k ON k.document_id = d.id "
            "WHERE d.collection = ? GROUP BY d.id ORDER BY d.id",
            (collection,),
        )
        return [dict(r) for r in rows]

    def add_document(self, collection: str, name: str, text: str, source: str = "") -> dict:
        """Indexes a document; the same text added twice to a collection is stored once."""
        self._require(collection)
        digest = hashlib.sha256(text.encode()).hexdigest()
        existing = self.db.execute(
            "SELECT id FROM documents WHERE collection = ? AND sha256 = ?", (collection, digest)
        ).fetchone()
        if existing:
            return {"id": existing["id"], "name": name, "chunks": 0, "duplicate": True}
        chunks = chunk_text(text)
        # Embedding is the slow part; do it before taking the lock.
        vectors = self.embedder.embed_documents(chunks) if chunks else np.zeros((0, 0))
        with self._lock:
            self._vectors()
            cursor = self.db.execute(
                "INSERT INTO documents (collection, name, source, sha256, chars, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (collection, name, source or name, digest, len(text), now()),
            )
            document_id = cursor.lastrowid
            for position, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                chunk_id = self.db.execute(
                    "INSERT INTO chunks (document_id, position, text) VALUES (?, ?, ?)",
                    (document_id, position, chunk),
                ).lastrowid
                self.db.execute(
                    "INSERT INTO chunk_vectors (chunk_id, collection, embedding) VALUES (?, ?, ?)",
                    (chunk_id, collection, np.asarray(vector, dtype=np.float32).tobytes()),
                )
            self.db.commit()
        return {"id": document_id, "name": name, "chunks": len(chunks), "duplicate": False}

    def delete_document(self, collection: str, document_id: int) -> bool:
        with self._lock:
            ids = [
                r["id"]
                for r in self.db.execute(
                    "SELECT k.id FROM chunks k JOIN documents d ON d.id = k.document_id "
                    "WHERE d.id = ? AND d.collection = ?",
                    (document_id, collection),
                )
            ]
            if ids and self._has_vectors():
                marks = ",".join("?" * len(ids))
                self.db.execute(f"DELETE FROM chunk_vectors WHERE chunk_id IN ({marks})", ids)
            deleted = self.db.execute(
                "DELETE FROM documents WHERE id = ? AND collection = ?", (document_id, collection)
            ).rowcount
            self.db.commit()
        return bool(deleted)

    def search(self, collection: str, query: str, k: int = 4) -> list[Hit]:
        self._require(collection)
        if not self._has_vectors():
            return []
        self._vectors()
        vector = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        rows = self.db.execute(
            "SELECT v.chunk_id, v.distance, k.text, d.id AS document_id, d.name, d.source "
            "FROM chunk_vectors v JOIN chunks k ON k.id = v.chunk_id "
            "JOIN documents d ON d.id = k.document_id "
            "WHERE v.embedding MATCH ? AND v.k = ? AND v.collection = ? ORDER BY v.distance",
            (vector.tobytes(), k, collection),
        )
        return [
            Hit(
                r["chunk_id"],
                r["document_id"],
                r["name"],
                r["source"],
                r["text"],
                round(1 - r["distance"], 4),
            )
            for r in rows
        ]

    def _has_vectors(self) -> bool:
        row = self.db.execute("SELECT 1 FROM sqlite_master WHERE name = 'chunk_vectors'").fetchone()
        return row is not None
