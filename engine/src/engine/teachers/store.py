"""Teachers saved by the user: OpenAI-compatible or Anthropic API endpoints with their keys.

Keys are stored in the workspace's SQLite file on this computer and never sent back to clients.
Installed models are offered as local teachers without being saved.
"""

from dataclasses import dataclass

from engine.rag.store import KnowledgeStore, now
from engine.teachers.providers import (
    DEFAULT_BASE_URLS,
    AnthropicTeacher,
    LocalTeacher,
    OpenAICompatibleTeacher,
    Paced,
    TeacherKind,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS teachers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    base_url TEXT NOT NULL,
    model TEXT NOT NULL,
    api_key TEXT,
    per_minute REAL,
    created_at TEXT NOT NULL
);
"""
FIELDS = ("name", "kind", "base_url", "model", "api_key", "per_minute")


class TeacherNotFound(Exception):
    pass


@dataclass
class TeacherConfig:
    name: str
    kind: TeacherKind
    model: str
    base_url: str = ""
    api_key: str | None = None
    # Requests per minute; free OpenRouter models allow about 20.
    per_minute: float | None = None
    id: int | None = None
    created_at: str | None = None

    def __post_init__(self):
        self.base_url = (self.base_url or DEFAULT_BASE_URLS.get(self.kind, "")).rstrip("/")

    def to_dict(self) -> dict:
        """Without the key: only whether there is one."""
        return {
            "id": self.id,
            "ref": str(self.id),
            "name": self.name,
            "kind": self.kind,
            "base_url": self.base_url,
            "model": self.model,
            "has_key": bool(self.api_key),
            "per_minute": self.per_minute,
            "created_at": self.created_at,
        }


def client(config: TeacherConfig):
    if config.kind == "anthropic":
        return AnthropicTeacher(config.base_url, config.api_key, config.model, config.name)
    return OpenAICompatibleTeacher(config.base_url, config.api_key, config.model, config.name)


class TeacherStore:
    def __init__(self, store: KnowledgeStore, models):
        self.store = store
        self.models = models
        self._lock = store._lock
        self.db.executescript(SCHEMA)

    @property
    def db(self):
        return self.store.db

    def saved(self) -> list[TeacherConfig]:
        rows = self.db.execute("SELECT * FROM teachers ORDER BY id")
        return [TeacherConfig(**dict(r)) for r in rows]

    def get(self, teacher_id: int) -> TeacherConfig:
        row = self.db.execute("SELECT * FROM teachers WHERE id = ?", (teacher_id,)).fetchone()
        if row is None:
            raise TeacherNotFound(str(teacher_id))
        return TeacherConfig(**dict(row))

    def all(self) -> list[dict]:
        """Saved teachers, then installed models as local teachers ("local:<model id>")."""
        local = [
            {
                "id": None,
                "ref": f"local:{model_id}",
                "name": model_id,
                "kind": "local",
                "base_url": "",
                "model": model_id,
                "has_key": False,
                "per_minute": None,
                "created_at": None,
            }
            for model_id in self.models.installed()
            if model_id != "echo"
        ]
        return [t.to_dict() for t in self.saved()] + local

    def add(self, config: TeacherConfig) -> TeacherConfig:
        with self._lock:
            cursor = self.db.execute(
                f"INSERT INTO teachers (created_at, {', '.join(FIELDS)}) "
                f"VALUES (?, {', '.join('?' * len(FIELDS))})",
                (now(), *(getattr(config, f) for f in FIELDS)),
            )
            self.db.commit()
        return self.get(cursor.lastrowid)

    def update(self, teacher_id: int, config: TeacherConfig) -> TeacherConfig:
        """An empty api_key keeps the saved one."""
        current = self.get(teacher_id)
        config.api_key = config.api_key or current.api_key
        with self._lock:
            self.db.execute(
                f"UPDATE teachers SET {', '.join(f + ' = ?' for f in FIELDS)} WHERE id = ?",
                (*(getattr(config, f) for f in FIELDS), teacher_id),
            )
            self.db.commit()
        return self.get(teacher_id)

    def delete(self, teacher_id: int) -> None:
        with self._lock:
            if not self.db.execute("DELETE FROM teachers WHERE id = ?", (teacher_id,)).rowcount:
                raise TeacherNotFound(str(teacher_id))
            self.db.commit()

    def teacher(self, ref: str) -> Paced:
        """A ready-to-use teacher for "local:<model id>" or a saved teacher's id."""
        if ref.startswith("local:"):
            model_id = ref.removeprefix("local:")
            if model_id not in self.models.installed():
                raise TeacherNotFound(ref)
            return Paced(LocalTeacher(self.models, model_id), None)
        if not ref.isdigit():
            raise TeacherNotFound(ref)
        config = self.get(int(ref))
        return Paced(client(config), config.per_minute)
