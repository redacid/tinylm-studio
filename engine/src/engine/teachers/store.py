"""Teachers saved by the user: OpenAI-compatible or Anthropic API endpoints with their keys, or
a subscription's command-line tool (its program in base_url, no key).

Keys are stored in the workspace's SQLite file on this computer and never sent back to clients.
Installed models are offered as local teachers without being saved.
"""

import logging
from dataclasses import dataclass

from engine.rag.store import KnowledgeStore, now
from engine.teachers.providers import (
    DEFAULT_BASE_URLS,
    AnswerOptions,
    AnthropicTeacher,
    CliTeacher,
    LocalTeacher,
    OpenAICompatibleTeacher,
    Paced,
    TeacherKind,
    TeacherRuntime,
    ThinkingMode,
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
# Added after the first release: older databases get them when the engine starts.
ADDED_COLUMNS = {
    "max_tokens": "INTEGER",
    "temperature": "REAL",
    "thinking": "TEXT NOT NULL DEFAULT 'default'",
}
log = logging.getLogger(__name__)

# A saved teacher's model id in the chat API and /v1/models: "tinylm/teacher/<name>".
CHAT_PREFIX = "tinylm/teacher/"
FIELDS = ("name", "kind", "base_url", "model", "api_key", "per_minute", *ADDED_COLUMNS)


class TeacherNotFound(Exception):
    pass


class TeacherNameTaken(Exception):
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
    # Answer settings for every request (see AnswerOptions): empty leaves it to the request.
    max_tokens: int | None = None
    temperature: float | None = None
    thinking: ThinkingMode = "default"
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
            # The model id to chat with it through /v1/chat/completions and /v1/messages.
            "model_id": CHAT_PREFIX + self.name,
            "kind": self.kind,
            "base_url": self.base_url,
            "model": self.model,
            "has_key": bool(self.api_key),
            "per_minute": self.per_minute,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "thinking": self.thinking,
            "created_at": self.created_at,
        }

    @property
    def options(self) -> AnswerOptions:
        return AnswerOptions(self.max_tokens, self.temperature, self.thinking)


def client(config: TeacherConfig):
    if config.kind == "cli":
        return CliTeacher(config.base_url, config.model, config.name, options=config.options)
    kind = AnthropicTeacher if config.kind == "anthropic" else OpenAICompatibleTeacher
    return kind(config.base_url, config.api_key, config.model, config.name, options=config.options)


class TeacherStore:
    def __init__(self, store: KnowledgeStore, models, cli_tools=None):
        self.store = store
        self.models = models
        # The logins of cli teachers (CliTools); None leaves them to the tools' own folders.
        self.cli_tools = cli_tools
        self._lock = store._lock
        self.db.executescript(SCHEMA)
        columns = {r["name"] for r in self.db.execute("PRAGMA table_info(teachers)")}
        with self._lock:
            for name, definition in ADDED_COLUMNS.items():
                if name not in columns:
                    self.db.execute(f"ALTER TABLE teachers ADD COLUMN {name} {definition}")
            self.db.commit()
        # Saved before names had to be unique: the chat answers with the oldest of them.
        for row in self.db.execute(
            "SELECT name, COUNT(*) AS n FROM teachers GROUP BY name HAVING n > 1"
        ):
            message = "%d teachers are named %r; rename all but one to chat with them"
            log.warning(message, row["n"], row["name"])

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

    def by_name(self, name: str) -> TeacherConfig:
        row = self.db.execute(
            "SELECT * FROM teachers WHERE name = ? ORDER BY id LIMIT 1", (name,)
        ).fetchone()
        if row is None:
            raise TeacherNotFound(name)
        return TeacherConfig(**dict(row))

    def _check_name(self, name: str, teacher_id: int | None = None) -> None:
        taken = self.db.execute(
            "SELECT 1 FROM teachers WHERE name = ? AND id IS NOT ?", (name, teacher_id)
        ).fetchone()
        if taken:
            raise TeacherNameTaken(name)

    def all(self) -> list[dict]:
        """Saved teachers, then installed models as local teachers ("local:<model id>")."""
        local = [
            {
                "id": None,
                "ref": f"local:{model_id}",
                "name": model_id,
                "model_id": model_id,
                "kind": "local",
                "base_url": "",
                "model": model_id,
                "has_key": False,
                "per_minute": None,
                "max_tokens": None,
                "temperature": None,
                "thinking": "default",
                "created_at": None,
            }
            for model_id in self.models.installed()
            if model_id != "echo"
        ]
        return [t.to_dict() for t in self.saved()] + local

    def add(self, config: TeacherConfig) -> TeacherConfig:
        with self._lock:
            self._check_name(config.name)
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
            # Unchanged names pass, so teachers saved with a duplicate name stay editable.
            if config.name != current.name:
                self._check_name(config.name, teacher_id)
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

    def chat_runtime(self, name: str) -> TeacherRuntime:
        """A saved teacher as a runtime the chat answers with ("tinylm/teacher/<name>" there)."""
        return TeacherRuntime(self.client(self.by_name(name)))

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
        return Paced(self.client(config), config.per_minute)

    def client(self, config: TeacherConfig):
        """client(), with the engine's logins for a cli teacher."""
        teacher = client(config)
        if config.kind == "cli" and self.cli_tools is not None:
            teacher.environment = self.cli_tools.environment()
        return teacher
