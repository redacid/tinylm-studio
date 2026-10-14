"""Knowledge base: collections of documents, split into chunks and indexed for hybrid search.

Everything lives in one SQLite file. Chunks are found two ways: by meaning (vectors stored with
the sqlite-vec extension) and by words (an FTS5 full-text index ranked with BM25). The second
catches exact names such as "Borderlands 3" that a small embedding model blurs. Both rankings
are merged with reciprocal rank fusion.
"""

import hashlib
import re
import sqlite3
import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np

from engine.rag.chunking import chunk_text
from engine.rag.embeddings import Embedder

COLLECTION_NAME = re.compile(r"^[\w-]{1,64}$")
DocumentKind = Literal["all", "file", "web"]
DocumentSort = Literal["oldest", "newest", "name", "largest"]
DOCUMENT_ORDER = {
    "oldest": "d.id",
    "newest": "d.id DESC",
    "name": "d.name COLLATE NOCASE, d.id",
    "largest": "d.chars DESC, d.id",
}
# Reciprocal rank fusion constant: the usual value from the original RRF paper.
RRF_K = 60
# Each ranking contributes this many candidates per requested hit before fusion.
CANDIDATES_PER_HIT = 5

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

# Full-text index over chunks.text, kept in sync by triggers (they also fire on cascade deletes).
FULL_TEXT_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_text USING fts5(
    text, content='chunks', content_rowid='id', tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunk_text(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunk_text(chunk_text, rowid, text) VALUES ('delete', old.id, old.text);
END;
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


def full_text_query(text: str) -> str:
    """Turns free text into an FTS5 query that matches any of its words.

    Long words also match by prefix, so inflected forms ("Відьмака", "Відьмаку") find each other.
    """
    terms = []
    for word in dict.fromkeys(re.findall(r"\w+", text.lower())):
        if len(word) >= 6:
            terms.append(f'"{word[: max(4, len(word) - 2)]}"*')
        elif len(word) >= 2 or word.isdigit():
            terms.append(f'"{word}"')
    return " OR ".join(terms)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class KnowledgeStore:
    def __init__(self, path: Path, embedder: Embedder):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.embedder = embedder
        self._local = threading.local()
        # Write-ahead logging lets readers (the UI polling) work while a crawl or upload writes.
        # Switching to it fails at once if another process holds a write lock.
        try:
            self.db.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError as e:
            raise RuntimeError(
                f"{path} is locked by another process, probably an engine that is still running. "
                "Stop it (pkill -f 'python -m engine') and start again."
            ) from e
        self.db.executescript(SCHEMA)
        self._full_text()
        # Serializes writes inside this process; reentrant so helpers can take it again.
        self._lock = threading.RLock()
        self._vectors_ready = False

    @property
    def db(self) -> sqlite3.Connection:
        """This thread's connection: one sqlite3 connection must not be shared between threads
        (the API, crawls and uploads all run in their own threads)."""
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = self._connect()
            self._local.connection = connection
        return connection

    def _connect(self) -> sqlite3.Connection:
        import sqlite_vec

        # Wait for another writer (thread or process) instead of failing with "database is locked".
        connection = sqlite3.connect(self.path, timeout=60)
        connection.row_factory = sqlite3.Row
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 60000")
        connection.create_function("casefold", 1, str.casefold, deterministic=True)
        return connection

    def _vectors(self) -> None:
        """Creates the vector table on first use, sized for the embedding model."""
        if self._vectors_ready:
            return
        with self._lock:
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

    def _full_text(self) -> None:
        """Creates the full-text index; a database made before it existed gets indexed once."""
        exists = self.db.execute("SELECT 1 FROM sqlite_master WHERE name = 'chunk_text'").fetchone()
        self.db.executescript(FULL_TEXT_SCHEMA)
        if not exists:
            self.db.execute("INSERT INTO chunk_text(chunk_text) VALUES ('rebuild')")
        self.db.commit()

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

    def documents(
        self,
        collection: str,
        query: str = "",
        kind: DocumentKind = "all",
        sort: DocumentSort = "oldest",
        offset: int = 0,
        limit: int | None = None,
    ) -> list[dict]:
        """Documents of a collection, optionally filtered and one page at a time.

        query matches the name or source (file path or web address); kind keeps only uploaded
        files or only web pages.
        """
        where, params = self._document_filter(collection, query, kind)
        rows = self.db.execute(
            "SELECT d.id, d.name, d.source, d.chars, d.created_at, "
            "(SELECT COUNT(*) FROM chunks k WHERE k.document_id = d.id) AS chunks "
            f"FROM documents d WHERE {where} ORDER BY {DOCUMENT_ORDER[sort]} LIMIT ? OFFSET ?",
            (*params, -1 if limit is None else limit, offset),
        )
        return [dict(r) for r in rows]

    def count_documents(self, collection: str, query: str = "", kind: DocumentKind = "all") -> int:
        where, params = self._document_filter(collection, query, kind)
        return self.db.execute(
            f"SELECT COUNT(*) FROM documents d WHERE {where}", params
        ).fetchone()[0]

    def _document_filter(self, collection: str, query: str, kind: DocumentKind) -> tuple[str, list]:
        self._require(collection)
        where, params = ["d.collection = ?"], [collection]
        if query.strip():
            # SQLite's LIKE and lower() only fold ASCII; casefold also handles Cyrillic.
            where.append("(instr(casefold(d.name), ?) OR instr(casefold(d.source), ?))")
            params += [query.strip().casefold()] * 2
        if kind == "web":
            where.append("(d.source LIKE 'http://%' OR d.source LIKE 'https://%')")
        elif kind == "file":
            where.append("NOT (d.source LIKE 'http://%' OR d.source LIKE 'https://%')")
        return " AND ".join(where), params

    def add_document(self, collection: str, name: str, text: str, source: str = "") -> dict:
        """Indexes a document; the same text added twice to a collection is stored once."""
        self._require(collection)
        digest = hashlib.sha256(text.encode()).hexdigest()
        existing = self.db.execute(
            "SELECT id FROM documents WHERE collection = ? AND sha256 = ?", (collection, digest)
        ).fetchone()
        if existing:
            return {"id": existing["id"], "name": name, "chunks": 0, "duplicate": True}
        return self._insert(collection, name, source or name, digest, len(text), chunk_text(text))

    def add_chunks(self, collection: str, name: str, source: str, chunks: list[str]) -> dict:
        """Indexes a document already split into chunks, e.g. one exported with a model."""
        self._require(collection)
        digest = hashlib.sha256("\n\n".join(chunks).encode()).hexdigest()
        existing = self.db.execute(
            "SELECT id FROM documents WHERE collection = ? AND sha256 = ?", (collection, digest)
        ).fetchone()
        if existing:
            return {"id": existing["id"], "name": name, "chunks": 0, "duplicate": True}
        return self._insert(collection, name, source or name, digest, sum(map(len, chunks)), chunks)

    def export_documents(self, collection: str) -> Iterator[dict]:
        """Every document of the collection with its chunks, for add_chunks elsewhere."""
        self._require(collection)
        documents = self.db.execute(
            "SELECT id, name, source FROM documents WHERE collection = ? ORDER BY id",
            (collection,),
        ).fetchall()
        for d in documents:
            chunks = self.db.execute(
                "SELECT text FROM chunks WHERE document_id = ? ORDER BY position", (d["id"],)
            )
            yield {"name": d["name"], "source": d["source"], "chunks": [r[0] for r in chunks]}

    def _insert(
        self, collection: str, name: str, source: str, digest: str, chars: int, chunks: list[str]
    ) -> dict:
        # Embedding is the slow part; do it before taking the lock.
        vectors = self.embedder.embed_documents(chunks) if chunks else np.zeros((0, 0))
        with self._lock:
            self._vectors()
            cursor = self.db.execute(
                "INSERT INTO documents (collection, name, source, sha256, chars, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (collection, name, source, digest, chars, now()),
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
        """The best chunks by meaning and by words, merged. Higher score means a better match."""
        self._require(collection)
        if not self._has_vectors():
            return []
        candidates = k * CANDIDATES_PER_HIT
        rankings = [
            self._search_vectors(collection, query, candidates),
            self._search_words(collection, query, candidates),
        ]
        scores: dict[int, float] = {}
        rows: dict[int, sqlite3.Row] = {}
        for ranking in rankings:
            for rank, row in enumerate(ranking, 1):
                scores[row["chunk_id"]] = scores.get(row["chunk_id"], 0.0) + 1 / (RRF_K + rank)
                rows.setdefault(row["chunk_id"], row)
        best = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
        return [
            Hit(
                chunk_id,
                rows[chunk_id]["document_id"],
                rows[chunk_id]["name"],
                rows[chunk_id]["source"],
                rows[chunk_id]["text"],
                round(scores[chunk_id], 4),
            )
            for chunk_id in best
        ]

    def _search_vectors(self, collection: str, query: str, limit: int) -> list[sqlite3.Row]:
        self._vectors()
        vector = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        return self.db.execute(
            "SELECT v.chunk_id, k.text, d.id AS document_id, d.name, d.source "
            "FROM chunk_vectors v JOIN chunks k ON k.id = v.chunk_id "
            "JOIN documents d ON d.id = k.document_id "
            "WHERE v.embedding MATCH ? AND v.k = ? AND v.collection = ? ORDER BY v.distance",
            (vector.tobytes(), limit, collection),
        ).fetchall()

    def _search_words(self, collection: str, query: str, limit: int) -> list[sqlite3.Row]:
        match = full_text_query(query)
        if not match:
            return []
        return self.db.execute(
            "SELECT k.id AS chunk_id, k.text, d.id AS document_id, d.name, d.source "
            "FROM chunk_text f JOIN chunks k ON k.id = f.rowid "
            "JOIN documents d ON d.id = k.document_id "
            "WHERE chunk_text MATCH ? AND d.collection = ? ORDER BY bm25(chunk_text) LIMIT ?",
            (match, collection, limit),
        ).fetchall()

    def _has_vectors(self) -> bool:
        row = self.db.execute("SELECT 1 FROM sqlite_master WHERE name = 'chunk_vectors'").fetchone()
        return row is not None
