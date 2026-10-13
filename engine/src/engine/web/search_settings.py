"""Search settings: the provider, its options and API key, and the domain filter.

The defaults come from TINYLM_* environment variables (see engine.config); what is changed in
the app's Settings is saved in <workspace>/search-settings.json and wins over them. API keys are
never sent back to clients: they only learn whether a key is set.
"""

import json
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from engine.config import Settings
from engine.web.search import (
    BraveSearch,
    DomainFilter,
    ExaSearch,
    PerplexitySearch,
    SearchError,
    SearchProvider,
    SearxngSearch,
    SerperSearch,
    TavilySearch,
    normalize_domain,
)

Provider = Literal["searxng", "brave", "tavily", "exa", "perplexity", "serper"]
KEY_FIELDS = (
    "brave_api_key",
    "tavily_api_key",
    "exa_api_key",
    "perplexity_api_key",
    "serper_api_key",
)
# Provider -> (title, the setting that holds its key).
NEEDS_KEY = {
    "brave": ("Brave Search", "brave_api_key"),
    "tavily": ("Tavily", "tavily_api_key"),
    "exa": ("Exa", "exa_api_key"),
    "perplexity": ("Perplexity", "perplexity_api_key"),
    "serper": ("Serper", "serper_api_key"),
}


class SearchSettings(BaseModel):
    provider: Provider = "searxng"
    include_domains: list[str] = []
    exclude_domains: list[str] = []
    searxng_url: str = "http://127.0.0.1:8888"
    brave_api_key: str | None = None
    tavily_api_key: str | None = None
    tavily_depth: Literal["basic", "advanced"] = "basic"
    tavily_topic: Literal["general", "news"] = "general"
    exa_api_key: str | None = None
    exa_type: Literal["auto", "fast", "instant"] = "auto"
    perplexity_api_key: str | None = None
    serper_api_key: str | None = None
    # Google country code for Serper results, e.g. "ua"; any when empty.
    serper_country: str | None = Field(None, pattern=r"^[a-z]{2}$")

    @field_validator("include_domains", "exclude_domains", mode="before")
    @classmethod
    def domains(cls, value):
        if isinstance(value, str):
            value = value.replace(",", "\n").split("\n")
        found, bad = [], []
        for text in value or []:
            if not str(text).strip():
                continue
            domain = normalize_domain(str(text))
            if domain is None:
                bad.append(str(text).strip())
            elif domain not in found:
                found.append(domain)
        if bad:
            raise ValueError(f"not a domain or pattern: {', '.join(bad)}")
        return found

    @field_validator("serper_country", *KEY_FIELDS, mode="before")
    @classmethod
    def empty_is_none(cls, value):
        return value.strip() or None if isinstance(value, str) else value

    @property
    def domain_filter(self) -> DomainFilter:
        return DomainFilter(tuple(self.include_domains), tuple(self.exclude_domains))

    def public(self) -> dict:
        """Without the API keys; `<name>_set` says whether each one is set."""
        data = self.model_dump()
        for name in KEY_FIELDS:
            data[f"{name}_set"] = bool(data.pop(name))
        return data


def env_defaults(settings: Settings) -> dict:
    return {
        "provider": settings.search_provider,
        "include_domains": settings.search_include_domains,
        "exclude_domains": settings.search_exclude_domains,
        "searxng_url": settings.searxng_url,
        **{name: getattr(settings, name) for name in KEY_FIELDS},
    }


class SearchSettingsStore:
    def __init__(self, path: Path, settings: Settings):
        self.path = path
        self.settings = settings
        self._lock = threading.Lock()

    def _saved(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, ValueError):
            return {}

    def get(self) -> SearchSettings:
        return SearchSettings(**{**env_defaults(self.settings), **self._saved()})

    def update(self, changes: dict) -> SearchSettings:
        """Saves the given fields; a field left out keeps its value, an empty key removes it."""
        with self._lock:
            saved = self._saved()
            merged = SearchSettings(**{**env_defaults(self.settings), **saved, **changes})
            for name in changes:
                value = getattr(merged, name)
                # An empty key is saved as "", so it also hides a key set in the environment.
                saved[name] = "" if name in KEY_FIELDS and value is None else value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(saved, indent=2))
            tmp.chmod(0o600)  # holds API keys
            tmp.replace(self.path)
            return merged

    def provider(self, search: SearchSettings | None = None) -> SearchProvider:
        return make_provider(search or self.get(), self.settings.proxy)


def make_provider(s: SearchSettings, proxy: str | None) -> SearchProvider:
    if s.provider in NEEDS_KEY:
        title, field = NEEDS_KEY[s.provider]
        if not getattr(s, field):
            raise SearchError(
                f"{title} needs an API key: add it in Settings > Search "
                f"or set TINYLM_{field.upper()}."
            )
    if s.provider == "brave":
        return BraveSearch(s.brave_api_key, proxy)
    if s.provider == "tavily":
        return TavilySearch(s.tavily_api_key, proxy, s.tavily_depth, s.tavily_topic)
    if s.provider == "exa":
        return ExaSearch(s.exa_api_key, proxy, s.exa_type)
    if s.provider == "perplexity":
        return PerplexitySearch(s.perplexity_api_key, proxy)
    if s.provider == "serper":
        return SerperSearch(s.serper_api_key, proxy, s.serper_country)
    return SearxngSearch(s.searxng_url)


def validation_message(error: ValidationError) -> str:
    """'exclude_domains: not a domain or pattern: x' instead of pydantic's report."""
    first = error.errors()[0]
    field = ".".join(str(p) for p in first["loc"])
    text = first["msg"].removeprefix("Value error, ")
    return f"{field}: {text}" if field else text
