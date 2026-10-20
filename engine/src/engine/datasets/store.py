"""Datasets of training examples, kept in the workspace's SQLite file next to the collections.

Each dataset is split into a training part and a held-out evaluation part. The split is decided
by each example's id, so it stays the same when other examples are added, edited or deleted,
and changing the evaluation share only moves the examples near the boundary.
"""

import hashlib
import json
import re
from collections.abc import Iterator
from typing import Literal

from engine.datasets.formats import Messages, parse, validate
from engine.rag.store import KnowledgeStore, now

DATASET_NAME = re.compile(r"^[\w-]{1,64}$")
Split = Literal["all", "train", "eval"]
Origin = Literal["manual", "imported", "generated"]
ExampleSort = Literal["oldest", "newest", "longest", "shortest"]
EXAMPLE_ORDER = {
    "oldest": "e.id",
    "newest": "e.id DESC",
    # The length of the conversation's text (as stored, JSON).
    "longest": "length(e.messages) DESC, e.id",
    "shortest": "length(e.messages), e.id",
}
# A well-mixed 0-99 bucket per example id (Knuth's multiplicative hash).
BUCKET = "((e.id * 2654435761) % 4294967296 % 100)"
# Errors reported per import; a broken file would otherwise send thousands of lines.
MAX_REPORTED_ERRORS = 20

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    name TEXT PRIMARY KEY,
    eval_percent INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS examples (
    id INTEGER PRIMARY KEY,
    dataset TEXT NOT NULL REFERENCES datasets(name) ON DELETE CASCADE ON UPDATE CASCADE,
    messages TEXT NOT NULL,  -- JSON list of {role, content}
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    origin TEXT NOT NULL DEFAULT 'manual',  -- manual, imported or generated
    source TEXT,  -- for generated examples: the document they were written from
    UNIQUE (dataset, sha256)
);
"""
# Columns added after the first release, with their definitions.
LATER_COLUMNS = {
    "origin": "TEXT NOT NULL DEFAULT 'manual'",
    "source": "TEXT",
}


class DatasetNotFound(Exception):
    pass


class InvalidDataset(ValueError):
    pass


def digest(messages: Messages) -> str:
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()


class DatasetStore:
    def __init__(self, store: KnowledgeStore):
        self.store = store
        self._lock = store._lock
        self.db.executescript(SCHEMA)
        columns = {r["name"] for r in self.db.execute("PRAGMA table_info(examples)")}
        for name, definition in LATER_COLUMNS.items():
            if name not in columns:
                self.db.execute(f"ALTER TABLE examples ADD COLUMN {name} {definition}")
        self.db.commit()

    @property
    def db(self):
        return self.store.db

    # Datasets

    def datasets(self) -> list[dict]:
        rows = self.db.execute(
            f"SELECT d.name, d.eval_percent, d.created_at, COUNT(e.id) AS examples, "
            f"COALESCE(SUM({BUCKET} < d.eval_percent), 0) AS eval "
            "FROM datasets d LEFT JOIN examples e ON e.dataset = d.name "
            "GROUP BY d.name ORDER BY d.name"
        )
        return [{**dict(r), "train": r["examples"] - r["eval"]} for r in rows]

    def dataset(self, name: str) -> dict:
        found = next((d for d in self.datasets() if d["name"] == name), None)
        if found is None:
            raise DatasetNotFound(name)
        return found

    def create(self, name: str, eval_percent: int = 10) -> dict:
        if not DATASET_NAME.match(name):
            raise InvalidDataset("Use up to 64 letters, digits, '_' or '-'.")
        check_percent(eval_percent)
        with self._lock:
            self.db.execute(
                "INSERT OR IGNORE INTO datasets VALUES (?, ?, ?)", (name, eval_percent, now())
            )
            self.db.commit()
        return self.dataset(name)

    def update(self, name: str, eval_percent: int) -> dict:
        check_percent(eval_percent)
        with self._lock:
            self._require(name)
            self.db.execute(
                "UPDATE datasets SET eval_percent = ? WHERE name = ?", (eval_percent, name)
            )
            self.db.commit()
        return self.dataset(name)

    def delete(self, name: str) -> None:
        with self._lock:
            self._require(name)
            self.db.execute("DELETE FROM datasets WHERE name = ?", (name,))
            self.db.commit()

    def _require(self, name: str) -> None:
        if self.db.execute("SELECT 1 FROM datasets WHERE name = ?", (name,)).fetchone() is None:
            raise DatasetNotFound(name)

    # Examples

    def examples(
        self,
        dataset: str,
        query: str = "",
        split: Split = "all",
        offset: int = 0,
        limit: int | None = None,
        origin: Origin | None = None,
        sort: ExampleSort = "oldest",
    ) -> list[dict]:
        where, params = self._filter(dataset, query, split, origin)
        rows = self.db.execute(
            "SELECT e.id, e.messages, e.created_at, e.origin, e.source, "
            f"{BUCKET} < d.eval_percent AS held_out "
            f"FROM examples e JOIN datasets d ON d.name = e.dataset WHERE {where} "
            f"ORDER BY {EXAMPLE_ORDER[sort]} LIMIT ? OFFSET ?",
            (*params, -1 if limit is None else limit, offset),
        )
        return [
            {
                "id": r["id"],
                "messages": json.loads(r["messages"]),
                "split": "eval" if r["held_out"] else "train",
                "origin": r["origin"],
                "source": r["source"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def count(
        self, dataset: str, query: str = "", split: Split = "all", origin: Origin | None = None
    ) -> int:
        where, params = self._filter(dataset, query, split, origin)
        return self.db.execute(
            f"SELECT COUNT(*) FROM examples e JOIN datasets d ON d.name = e.dataset WHERE {where}",
            params,
        ).fetchone()[0]

    def _filter(
        self, dataset: str, query: str, split: Split, origin: Origin | None = None
    ) -> tuple[str, list]:
        self._require(dataset)
        where, params = ["e.dataset = ?"], [dataset]
        if origin:
            where.append("e.origin = ?")
            params.append(origin)
        if query.strip():
            # Searches the JSON text; casefold also handles Cyrillic, unlike SQLite's lower().
            where.append("instr(casefold(e.messages), ?)")
            params.append(json.dumps(query.strip().casefold(), ensure_ascii=False)[1:-1])
        if split == "train":
            where.append(f"{BUCKET} >= d.eval_percent")
        elif split == "eval":
            where.append(f"{BUCKET} < d.eval_percent")
        return " AND ".join(where), params

    def add(
        self,
        dataset: str,
        messages: Messages,
        origin: Origin = "manual",
        source: str | None = None,
    ) -> dict:
        """Adds one example; the same example twice is stored once (duplicate=True)."""
        messages = validate(messages)
        with self._lock:
            self._require(dataset)
            added = self._insert(dataset, messages, origin, source)
            self.db.commit()
        return {"id": added, "duplicate": added is None}

    def _insert(
        self, dataset: str, messages: Messages, origin: Origin = "manual", source: str | None = None
    ) -> int | None:
        cursor = self.db.execute(
            "INSERT OR IGNORE INTO examples "
            "(dataset, messages, sha256, created_at, origin, source) VALUES (?, ?, ?, ?, ?, ?)",
            (
                dataset,
                json.dumps(messages, ensure_ascii=False),
                digest(messages),
                now(),
                origin,
                source,
            ),
        )
        return cursor.lastrowid if cursor.rowcount else None

    def edit(self, dataset: str, example_id: int, messages: Messages) -> bool:
        messages = validate(messages)
        with self._lock:
            updated = self.db.execute(
                "UPDATE OR IGNORE examples SET messages = ?, sha256 = ? "
                "WHERE id = ? AND dataset = ?",
                (json.dumps(messages, ensure_ascii=False), digest(messages), example_id, dataset),
            ).rowcount
            self.db.commit()
        if not updated and self._exists(dataset, example_id):
            raise InvalidDataset("The same example is already in the dataset.")
        return bool(updated)

    def _exists(self, dataset: str, example_id: int) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM examples WHERE id = ? AND dataset = ?", (example_id, dataset)
        ).fetchone()
        return row is not None

    def remove(self, dataset: str, example_id: int) -> bool:
        with self._lock:
            deleted = self.db.execute(
                "DELETE FROM examples WHERE id = ? AND dataset = ?", (example_id, dataset)
            ).rowcount
            self.db.commit()
        return bool(deleted)

    def import_file(self, dataset: str, filename: str, data: bytes) -> dict:
        """Adds every valid example of a JSONL, JSON or CSV file. Returns counts and the first
        errors with their line numbers."""
        self._require(dataset)
        added = duplicates = invalid = 0
        errors = []
        with self._lock:
            for record in parse(filename, data):
                if record.messages is None:
                    invalid += 1
                    if len(errors) < MAX_REPORTED_ERRORS:
                        errors.append({"line": record.line, "error": record.error})
                elif (
                    self._insert(
                        dataset, record.messages, record.origin or "imported", record.source
                    )
                    is None
                ):
                    duplicates += 1
                else:
                    added += 1
            self.db.commit()
        return {
            "name": filename,
            "added": added,
            "duplicates": duplicates,
            "invalid": invalid,
            "errors": errors,
        }

    def export_lines(self, dataset: str, split: Split = "all") -> Iterator[str]:
        """The examples as JSON lines that import_file reads back, keeping origin and source."""
        for example in self.examples(dataset, split=split):
            record = {"messages": example["messages"], "origin": example["origin"]}
            if example["source"]:
                record["source"] = example["source"]
            yield json.dumps(record, ensure_ascii=False) + "\n"

    def split(self, dataset: str) -> tuple[list[Messages], list[Messages]]:
        """All examples as (train, eval) message lists, for a training run."""
        examples = self.examples(dataset)
        train = [e["messages"] for e in examples if e["split"] == "train"]
        held_out = [e["messages"] for e in examples if e["split"] == "eval"]
        return train, held_out


def check_percent(eval_percent: int) -> None:
    if not 0 <= eval_percent <= 50:
        raise InvalidDataset("The evaluation share must be between 0 and 50%.")
