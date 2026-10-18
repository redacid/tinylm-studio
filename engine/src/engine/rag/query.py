"""SQL queries over the data (collections, documents, chunks, websites), for the SQL tab.

A query runs on its own read-only connection that may only read the data's tables: anything
else (writing, PRAGMA, ATTACH, other tables such as the teachers with their API keys) is refused
by SQLite's authorizer before it runs. Queries are stopped after a time limit, and only the
first rows are returned.

DELETE FROM one of the data's tables is the one way to change them. Its WHERE runs as a SELECT
under the same rules, so it can only read the data, and the matching rows are then deleted by
the knowledge store the way the Data tab deletes them: their search vectors go too, and rows
that depend on them (a collection's documents, a document's chunks and crawled pages) follow.
It first says how many rows would go; only a confirmed DELETE deletes.
"""

import re
import sqlite3
import time

from engine.rag.store import KnowledgeStore, marks

# The tables a query may read, with what they hold and what their columns mean.
TABLES = {
    "collections": (
        "Collections of documents, as on the Data tab.",
        {
            "name": "The collection's name",
            "created_at": "When it was made (ISO 8601, UTC)",
        },
    ),
    "documents": (
        "Documents: uploaded files, imported ones and crawled pages, one row each.",
        {
            "id": "The document's number",
            "collection": "Its collection (collections.name)",
            "name": "The file name or the page title",
            "source": "Where it came from: a file name or a page's address",
            "sha256": "Hash of its text: the same text is not added twice to a collection",
            "chars": "Length of its text in characters",
            "created_at": "When it was added (ISO 8601, UTC)",
        },
    ),
    "chunks": (
        "Pieces of the documents' text that search finds and answers quote, in order.",
        {
            "id": "The chunk's number",
            "document_id": "Its document (documents.id)",
            "position": "Its place in the document, from 0",
            "text": "The text",
        },
    ),
    "chunk_text": (
        "Full-text index of chunks.text: chunk_text MATCH 'words' finds chunks by word, "
        "rowid is chunks.id, bm25(chunk_text) ranks them (lower is better).",
        {"text": "The chunk's text, searchable with MATCH"},
    ),
    "websites": (
        "Saved websites: their start addresses and crawl settings.",
        {
            "id": "The website's number",
            "collection": "The collection its pages go to",
            "name": "Its name",
            "urls": "Start addresses (a JSON list)",
            "max_pages": "New pages per crawl",
            "max_depth": "Link hops from a start address",
            "per_minute": "Requests per minute to the site",
            "use_browser": "1: pages are loaded in the browser (CDP)",
            "respect_robots": "1: robots.txt is followed",
            "only_under_start": "1: only pages under a start address",
            "exclude": "Link patterns never followed (a JSON list)",
            "created_at": "When it was saved",
            "last_crawled_at": "When it was last crawled; NULL if never",
        },
    ),
    "crawled_pages": (
        "Pages crawls have fetched, also the ones too short to index.",
        {
            "collection": "The collection",
            "url": "The page's address",
            "links": "Same-site links found on it (a JSON list)",
            "document_id": "Its document (documents.id); NULL when it was too short to index",
            "fetched_at": "When it was fetched",
        },
    ),
}

EXAMPLES = [
    (
        "Documents per collection",
        "SELECT collection, count(*) AS documents, sum(chars) AS characters\n"
        "FROM documents\nGROUP BY collection\nORDER BY documents DESC",
    ),
    (
        "Largest documents",
        "SELECT collection, name, chars, source\nFROM documents\nORDER BY chars DESC\nLIMIT 20",
    ),
    (
        "Documents added by day",
        "SELECT substr(created_at, 1, 10) AS day, count(*) AS documents\n"
        "FROM documents\nGROUP BY day\nORDER BY day DESC",
    ),
    (
        "Find chunks by words (full-text)",
        "SELECT d.collection, d.name, snippet(chunk_text, 0, '[', ']', '…', 12) AS found\n"
        "FROM chunk_text\nJOIN chunks c ON c.id = chunk_text.rowid\n"
        "JOIN documents d ON d.id = c.document_id\n"
        "WHERE chunk_text MATCH 'witcher'\nORDER BY bm25(chunk_text)\nLIMIT 20",
    ),
    (
        "Find text in any case (casefold works for Cyrillic too)",
        "SELECT d.name, c.position, substr(c.text, 1, 200) AS text\n"
        "FROM chunks c JOIN documents d ON d.id = c.document_id\n"
        "WHERE casefold(c.text) LIKE '%' || casefold('Відьмак') || '%'\nLIMIT 50",
    ),
    (
        "Documents with the most chunks",
        "SELECT d.collection, d.name, count(*) AS chunks, avg(length(c.text)) AS avg_chars\n"
        "FROM documents d JOIN chunks c ON c.document_id = d.id\n"
        "GROUP BY d.id\nORDER BY chunks DESC\nLIMIT 20",
    ),
    (
        "The same address twice in a collection",
        "SELECT collection, source, count(*) AS copies\nFROM documents\n"
        "WHERE source LIKE 'http%'\nGROUP BY collection, source\nHAVING copies > 1",
    ),
    (
        "Pages per site",
        "SELECT collection,\n"
        "       substr(url, 1, instr(substr(url, 9), '/') + 8) AS site,\n"
        "       count(*) AS pages, count(document_id) AS indexed\n"
        "FROM crawled_pages\nGROUP BY collection, site\nORDER BY pages DESC",
    ),
    (
        "Websites and their last crawl",
        "SELECT w.name, w.collection, w.last_crawled_at, json_array_length(w.urls) AS starts,\n"
        "       (SELECT count(*) FROM crawled_pages p WHERE p.collection = w.collection) AS pages\n"
        "FROM websites w\nORDER BY w.last_crawled_at DESC",
    ),
    (
        "Delete documents by address (asks first)",
        "DELETE FROM documents\nWHERE source LIKE '%example.com/old/%'",
    ),
]

DEFAULT_LIMIT = 500
MAX_LIMIT = 10_000
TIME_LIMIT = 15.0
# Shadow tables SQLite's full-text index reads for a MATCH.
FULL_TEXT_PREFIX = "chunk_text"
ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
# The full-text index asks whether the file changed before a MATCH; it only reads a counter.
ALLOWED_PRAGMAS = {"data_version"}
# Tables DELETE may delete from; the full-text index follows chunks by itself.
DELETABLE = ["collections", "documents", "chunks", "websites", "crawled_pages"]
DELETE = re.compile(
    r"DELETE\s+FROM\s+(?P<table>\w+)(?:\s+(?:AS\s+)?(?P<alias>(?!WHERE\b)\w+))?"
    r"(?:\s+(?P<where>WHERE\b.*))?",
    re.IGNORECASE | re.DOTALL,
)
# Rows deleted per statement, under SQLite's limit on bound values.
BATCH = 500


class QueryError(Exception):
    pass


def readable(table: str | None) -> bool:
    return table is not None and (table in TABLES or table.startswith(FULL_TEXT_PREFIX))


def authorize(action: int, arg1, arg2, database, trigger) -> int:
    if action in ALLOWED_ACTIONS:
        return sqlite3.SQLITE_OK
    # A read with no column is count(*) over a table or a WITH query: it sees no values.
    if action == sqlite3.SQLITE_READ and (readable(arg1) or arg2 == ""):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA and arg1 in ALLOWED_PRAGMAS and arg2 is None:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def cell(value):
    """A value as JSON can carry it: blobs as their size."""
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    return value


def is_delete(sql: str) -> bool:
    return re.match(r"\s*DELETE\b", sql, re.IGNORECASE) is not None


class DataQuery:
    def __init__(self, store: KnowledgeStore):
        self.store = store
        self.path = store.path

    def schema(self) -> dict:
        """The tables a query may read, their columns with types, and example queries."""
        tables = []
        with self._connect() as db:
            for name, (about, notes) in TABLES.items():
                columns = db.execute(f"PRAGMA table_info({name})").fetchall()
                if not columns:
                    continue  # not made yet (e.g. no website was ever saved)
                tables.append(
                    {
                        "name": name,
                        "about": about,
                        "deletable": name in DELETABLE,
                        "columns": [
                            {
                                "name": c[1],
                                "type": c[2] or "TEXT",
                                "not_null": bool(c[3]),
                                "primary_key": bool(c[5]),
                                "about": notes.get(c[1], ""),
                            }
                            for c in columns
                        ],
                    }
                )
        examples = [{"title": title, "sql": sql} for title, sql in EXAMPLES]
        return {"tables": tables, "examples": examples}

    def run(self, sql: str, limit: int = DEFAULT_LIMIT, confirm: bool = False) -> dict:
        """Runs one SELECT and returns its columns and up to limit rows, or a DELETE: without
        confirm it only counts the rows it would delete (in "delete"), with it deletes them."""
        sql = sql.strip().rstrip(";").strip()
        if not sql:
            raise QueryError("Write a query first.")
        started = time.monotonic()
        if is_delete(sql):
            table, rowids = self._matching(sql, started)
            deleted = self._delete(table, rowids) if confirm else 0
            result = {
                "columns": ["table", "deleted" if confirm else "rows to delete"],
                "rows": [[table, deleted if confirm else len(rowids)]],
                "truncated": False,
                "delete": {"table": table, "rows": len(rowids), "done": confirm},
            }
        else:
            limit = max(1, min(limit, MAX_LIMIT))
            columns, rows = self._select(sql, limit + 1, started)
            result = {
                "columns": columns,
                "rows": [[cell(v) for v in row] for row in rows[:limit]],
                "truncated": len(rows) > limit,
            }
        return {**result, "elapsed_ms": round((time.monotonic() - started) * 1000, 1)}

    def _select(self, sql: str, limit: int | None, started: float) -> tuple[list[str], list]:
        with self._connect() as db:
            db.set_authorizer(authorize)
            db.set_progress_handler(lambda: int(time.monotonic() - started > TIME_LIMIT), 10_000)
            try:
                cursor = db.execute(sql)
                rows = cursor.fetchall() if limit is None else cursor.fetchmany(limit)
            except sqlite3.DatabaseError as e:
                message = str(e)
                if message == "interrupted":
                    message = f"The query took longer than {TIME_LIMIT:.0f} s and was stopped."
                elif "not authorized" in message or "prohibited" in message:
                    message = (
                        "Not allowed: queries can only read the tables listed "
                        f"({', '.join(TABLES)}) and delete from "
                        f"{', '.join(DELETABLE)}."
                    )
                raise QueryError(message) from None
            except sqlite3.Warning as e:  # e.g. two statements at once
                raise QueryError(str(e)) from None
            return [d[0] for d in cursor.description or []], rows

    def _matching(self, sql: str, started: float) -> tuple[str, list[int]]:
        """The table of a DELETE and the rowids of the rows it would delete."""
        found = DELETE.fullmatch(sql)
        if not found:
            raise QueryError("Write a DELETE as: DELETE FROM table WHERE condition")
        table = found["table"].lower()
        if table not in DELETABLE:
            raise QueryError(f"Not allowed: DELETE works on {', '.join(DELETABLE)}.")
        alias = f" AS {found['alias']}" if found["alias"] else ""
        select = (
            f"SELECT {found['alias'] or table}.rowid FROM {table}{alias} {found['where'] or ''}"
        )
        _, rows = self._select(select, None, started)
        return table, [r[0] for r in rows]

    def _delete(self, table: str, rowids: list[int]) -> int:
        """Deletes rows by rowid on the store's connection, with their search vectors."""
        store, deleted = self.store, 0
        with store._lock:
            try:
                for start in range(0, len(rowids), BATCH):
                    batch = rowids[start : start + BATCH]
                    where = f"rowid IN ({marks(batch)})"
                    if store._has_vectors():
                        self._delete_vectors(table, where, batch)
                    deleted += store.db.execute(
                        f"DELETE FROM {table} WHERE {where}", batch
                    ).rowcount
                store.db.commit()
            except BaseException:
                store.db.rollback()
                raise
        return deleted

    def _delete_vectors(self, table: str, where: str, batch: list[int]) -> None:
        db = self.store.db
        if table == "collections":
            db.execute(
                f"DELETE FROM chunk_vectors WHERE collection IN "
                f"(SELECT name FROM collections WHERE {where})",
                batch,
            )
            return
        chunks = {
            "documents": f"SELECT id FROM chunks WHERE document_id IN "
            f"(SELECT id FROM documents WHERE {where})",
            "chunks": f"SELECT id FROM chunks WHERE {where}",
        }.get(table)
        if chunks:
            ids = [r[0] for r in db.execute(chunks, batch)]
            if ids:
                db.execute(f"DELETE FROM chunk_vectors WHERE chunk_id IN ({marks(ids)})", ids)

    def _connect(self) -> sqlite3.Connection:
        if not self.path.exists():
            raise QueryError("There is no data yet.")
        db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=10)
        db.execute("PRAGMA query_only = ON")
        db.create_function("casefold", 1, lambda s: s.casefold() if isinstance(s, str) else s)
        return db
