"""Crawler settings: robots.txt, request rate, defaults for new crawls, and how pages are fetched.

The defaults come from TINYLM_* environment variables (see engine.config); what is changed in
the app's Settings is saved in <workspace>/crawl-settings.json and wins over them.
"""

import json
import threading
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from engine.config import Settings


class CrawlSettings(BaseModel):
    # Sent whole by the API: its schema lists every field as present.
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    # Off: robots.txt is not read at all, for every crawl. Only for sites you may crawl anyway.
    respect_robots: bool = True
    # Requests per minute to each site, for crawls that do not set their own.
    per_minute: float = Field(20, gt=0, le=120)
    # Defaults for new websites and crawls.
    max_pages: int = Field(50, ge=1, le=100000)
    max_depth: int = Field(2, ge=0, le=100)
    # Load pages in the browser at cdp_url, for new websites and pages added from search.
    use_browser: bool = False
    user_agent: str = Field(min_length=1, max_length=300)
    cdp_url: str

    @field_validator("user_agent", "cdp_url")
    @classmethod
    def stripped(cls, value: str) -> str:
        return value.strip()

    @field_validator("cdp_url")
    @classmethod
    def web_address(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("not an http:// or https:// address")
        return value.rstrip("/")


def env_defaults(settings: Settings) -> dict:
    return {
        "respect_robots": settings.crawl_respect_robots,
        "per_minute": settings.crawl_per_minute,
        "max_pages": settings.crawl_max_pages,
        "max_depth": settings.crawl_max_depth,
        "use_browser": settings.crawl_use_browser,
        "user_agent": settings.user_agent,
        "cdp_url": settings.cdp_url,
    }


class CrawlSettingsStore:
    def __init__(self, path: Path, settings: Settings):
        self.path = path
        self.settings = settings
        self._lock = threading.Lock()

    def _saved(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, ValueError):
            return {}

    def get(self) -> CrawlSettings:
        return CrawlSettings(**{**env_defaults(self.settings), **self._saved()})

    def update(self, changes: dict) -> CrawlSettings:
        """Saves the given fields; a field left out keeps its value."""
        with self._lock:
            saved = self._saved()
            merged = CrawlSettings(**{**env_defaults(self.settings), **saved, **changes})
            saved.update({name: getattr(merged, name) for name in changes})
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(saved, indent=2))
            tmp.replace(self.path)
            return merged
