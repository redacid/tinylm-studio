"""Moving a collection between engines: from this computer to a RunPod pod, for example.

An export is JSON lines: a header with the collection's name, size and embedding model, then
each document with its chunks and their vectors (float32, base64), as the knowledge store keeps
them. An import adds the documents to a collection (made if missing); documents it already has
are skipped. When both engines use the same embedding model the vectors are stored as they are,
so nothing is computed; otherwise the chunks are embedded again with the importing engine's
model. Imports run in the background: an upload through RunPod's proxy must answer within 100
seconds.
"""

import base64
import json
import logging
import threading
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np

from engine.rag.store import KnowledgeStore

log = logging.getLogger(__name__)

FORMAT = "tinylm-collection/1"
# Chunks embedded and written together: big enough to keep a GPU busy, small enough that the
# progress moves and memory stays low.
BATCH_CHUNKS = 512


def batches(lines: Iterable[str], vectors: bool = False) -> Iterator[list[dict]]:
    """The documents of an export's lines (the header skipped), a batch of about BATCH_CHUNKS
    chunks at a time. With vectors, a document whose chunks all have one keeps them, decoded;
    otherwise they are dropped and the chunks get embedded."""
    batch: list[dict] = []
    size = 0
    for line in lines:
        if not line.strip():
            continue
        d = json.loads(line)
        if "chunks" not in d:
            continue  # the header
        encoded = d.pop("vectors", None)
        if vectors and encoded and len(encoded) == len(d["chunks"]) and all(encoded):
            rows = [np.frombuffer(base64.b64decode(v), dtype=np.float32) for v in encoded]
            d["vectors"] = np.stack(rows)
        batch.append(d)
        size += len(d["chunks"])
        if size >= BATCH_CHUNKS:
            yield batch
            batch, size = [], 0
    if batch:
        yield batch


def export_lines(store: KnowledgeStore, collection: str) -> Iterator[str]:
    """The collection as JSON lines, header first. Raises CollectionNotFound."""
    header = {
        "format": FORMAT,
        "collection": collection,
        "documents": store.count_documents(collection),
        "embedding_model": store.embedding_model(),
        "exported_at": now(),
    }
    yield json.dumps(header, ensure_ascii=False) + "\n"
    for document in store.export_documents(collection, vectors=True):
        document["vectors"] = [
            base64.b64encode(v).decode() if v else None for v in document["vectors"]
        ]
        yield json.dumps(document, ensure_ascii=False) + "\n"


def read_header(path: Path) -> dict:
    """The export's header; {} for a file without one (a model's collection.jsonl)."""
    with path.open(encoding="utf-8") as f:
        first = json.loads(f.readline() or "{}")
    return first if "format" in first else {}


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class ImportJob:
    collection: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: Literal["running", "done", "error"] = "running"
    total: int | None = None  # documents in the file, when its header says
    added: int = 0
    duplicates: int = 0
    error: str | None = None
    started_at: str = field(default_factory=now)
    finished_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ImportManager:
    def __init__(self, store: KnowledgeStore):
        self.store = store
        self.jobs: dict[str, ImportJob] = {}

    def start(self, collection: str, path: Path) -> ImportJob:
        """Imports the file at path (deleted when done) into the collection, made if missing."""
        header = read_header(path)
        job = ImportJob(collection, total=header.get("documents"))
        self.store.create_collection(collection)  # nothing if it exists
        self.jobs[job.id] = job
        # Vectors of another model would not match this engine's searches.
        exported_with = header.get("embedding_model")
        vectors = exported_with == self.store.embedding_model()
        log.info(
            "Importing %s documents into %s, %s",
            job.total or "the",
            collection,
            "with their vectors"
            if vectors
            else f"embedding them again (exported with {exported_with or 'no vectors'})",
        )
        threading.Thread(target=self._run, args=(job, path, vectors), daemon=True).start()
        return job

    def _run(self, job: ImportJob, path: Path, vectors: bool) -> None:
        try:
            with path.open(encoding="utf-8") as f:
                for batch in batches(f, vectors):
                    for result in self.store.add_chunked(batch, job.collection):
                        if result["duplicate"]:
                            job.duplicates += 1
                        else:
                            job.added += 1
            job.state = "done"
        except Exception as e:
            log.exception("Importing into %s failed", job.collection)
            job.state, job.error = "error", str(e)
        finally:
            job.finished_at = now()
            log.info(
                "Import into %s %s: %d documents added, %d already there",
                job.collection,
                job.state,
                job.added,
                job.duplicates,
            )
            path.unlink(missing_ok=True)
