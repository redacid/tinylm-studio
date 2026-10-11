"""What the crawler keeps between runs: saved websites and the pages it has already downloaded.

Both live in the knowledge base's SQLite file, next to the collections they belong to, and go
away with their collection. Remembering a page's links lets a later crawl walk through pages it
already has without downloading them again, and spend its page budget only on new ones.
"""

import json
from dataclasses import asdict, dataclass

from engine.rag.store import KnowledgeStore, now

SCHEMA = """
CREATE TABLE IF NOT EXISTS websites (
    id INTEGER PRIMARY KEY,
    collection TEXT NOT NULL REFERENCES collections(name) ON DELETE CASCADE,
    name TEXT NOT NULL,
    urls TEXT NOT NULL,  -- JSON list
    max_pages INTEGER NOT NULL,
    max_depth INTEGER NOT NULL,
    per_minute REAL NOT NULL,
    use_browser INTEGER NOT NULL,
    respect_robots INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    last_crawled_at TEXT
);
CREATE TABLE IF NOT EXISTS crawled_pages (
    collection TEXT NOT NULL REFERENCES collections(name) ON DELETE CASCADE,
    url TEXT NOT NULL,
    links TEXT NOT NULL,  -- JSON list of same-site links found on the page
    -- The indexed document; NULL for pages that were too short to index (listings, menus).
    -- Deleting the document forgets the page, so the next crawl downloads it again.
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (collection, url)
);
"""


@dataclass
class Website:
    """A saved crawl: start addresses and settings that can be run again."""

    name: str
    urls: list[str]
    max_pages: int = 50
    max_depth: int = 2
    per_minute: float = 20
    use_browser: bool = False
    respect_robots: bool = True
    id: int | None = None
    collection: str = ""
    created_at: str | None = None
    last_crawled_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


FIELDS = ("name", "max_pages", "max_depth", "per_minute", "use_browser", "respect_robots")


class WebsiteStore:
    def __init__(self, store: KnowledgeStore):
        self.store = store
        self._lock = store._lock
        self.db.executescript(SCHEMA)

    @property
    def db(self):
        return self.store.db

    # Saved websites

    def websites(self, collection: str) -> list[Website]:
        self.store._require(collection)
        rows = self.db.execute(
            "SELECT * FROM websites WHERE collection = ? ORDER BY id", (collection,)
        )
        return [self._website(r) for r in rows]

    def website(self, website_id: int) -> Website | None:
        row = self.db.execute("SELECT * FROM websites WHERE id = ?", (website_id,)).fetchone()
        return self._website(row) if row else None

    def add_website(self, collection: str, website: Website) -> Website:
        self.store._require(collection)
        with self._lock:
            cursor = self.db.execute(
                f"INSERT INTO websites (collection, urls, created_at, {', '.join(FIELDS)}) "
                f"VALUES (?, ?, ?, {', '.join('?' * len(FIELDS))})",
                (collection, json.dumps(website.urls), now(), *self._values(website)),
            )
            self.db.commit()
        return self.website(cursor.lastrowid)

    def update_website(self, website_id: int, website: Website) -> Website | None:
        with self._lock:
            updated = self.db.execute(
                f"UPDATE websites SET urls = ?, {', '.join(f + ' = ?' for f in FIELDS)} "
                "WHERE id = ?",
                (json.dumps(website.urls), *self._values(website), website_id),
            ).rowcount
            self.db.commit()
        return self.website(website_id) if updated else None

    def delete_website(self, website_id: int) -> bool:
        with self._lock:
            deleted = self.db.execute("DELETE FROM websites WHERE id = ?", (website_id,)).rowcount
            self.db.commit()
        return bool(deleted)

    def mark_crawled(self, website_id: int) -> None:
        with self._lock:
            self.db.execute(
                "UPDATE websites SET last_crawled_at = ? WHERE id = ?", (now(), website_id)
            )
            self.db.commit()

    @staticmethod
    def _values(website: Website) -> tuple:
        return tuple(getattr(website, f) for f in FIELDS)

    @staticmethod
    def _website(row) -> Website:
        data = dict(row)
        data["urls"] = json.loads(data["urls"])
        data["use_browser"] = bool(data["use_browser"])
        data["respect_robots"] = bool(data["respect_robots"])
        return Website(**data)

    # Pages already downloaded

    def known_links(self, collection: str, url: str) -> list[str] | None:
        """The links of a page downloaded before, or None when it has to be downloaded."""
        row = self.db.execute(
            "SELECT links FROM crawled_pages WHERE collection = ? AND url = ?", (collection, url)
        ).fetchone()
        return None if row is None else json.loads(row["links"])

    def remember_page(
        self, collection: str, url: str, links: list[str], document_id: int | None
    ) -> None:
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO crawled_pages VALUES (?, ?, ?, ?, ?)",
                (collection, url, json.dumps(links), document_id, now()),
            )
            self.db.commit()
