"""Moving a collection between engines: from this computer to a RunPod pod, for example.

An export is JSON lines: a header with the collection's name and size, then each document with
its chunks, as the knowledge store keeps them. An import adds the documents to a collection
(made if missing) and embeds the chunks with the importing engine's own embedding model, so the
two engines need not use the same one; documents it already has are skipped. Embedding is the
slow part, so imports run in the background: an upload through RunPod's proxy must answer
within 100 seconds.
"""

import json
import logging
import threading
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from engine.rag.store import KnowledgeStore

log = logging.getLogger(__name__)

FORMAT = "tinylm-collection/1"


def export_lines(store: KnowledgeStore, collection: str) -> Iterator[str]:
    """The collection as JSON lines, header first. Raises CollectionNotFound."""
    header = {
        "format": FORMAT,
        "collection": collection,
        "documents": store.count_documents(collection),
        "exported_at": now(),
    }
    yield json.dumps(header, ensure_ascii=False) + "\n"
    for document in store.export_documents(collection):
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
        job = ImportJob(collection, total=read_header(path).get("documents"))
        self.store.create_collection(collection)  # nothing if it exists
        self.jobs[job.id] = job
        log.info("Importing %s documents into %s", job.total or "the", collection)
        threading.Thread(target=self._run, args=(job, path), daemon=True).start()
        return job

    def _run(self, job: ImportJob, path: Path) -> None:
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    if "chunks" not in d:
                        continue  # the header
                    result = self.store.add_chunks(
                        job.collection, d["name"], d.get("source", ""), d["chunks"]
                    )
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
