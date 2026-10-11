"""Web search for finding pages and videos to add to a collection.

Two providers:
- SearXNG (default): a self-hosted metasearch engine. Free, no key; it queries Google, Bing,
  DuckDuckGo, YouTube and others itself. Its JSON output must be enabled (see docs/api.md).
- Brave Search API: a hosted service that needs an API key (free tier available).
"""

from dataclasses import asdict, dataclass
from typing import Literal, Protocol
from urllib.parse import urlsplit

import httpx

SearchKind = Literal["web", "video"]


class SearchError(Exception):
    pass


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    kind: SearchKind = "web"
    thumbnail: str | None = None
    source: str = ""  # the engine or site that found it

    @property
    def site(self) -> str:
        return urlsplit(self.url).hostname or ""

    def to_dict(self) -> dict:
        return {**asdict(self), "site": self.site}


class SearchProvider(Protocol):
    name: str

    def search(
        self, query: str, kind: SearchKind = "web", limit: int = 10, language: str | None = None
    ) -> list[SearchResult]: ...

    def close(self) -> None: ...


def unique(results: list[SearchResult], limit: int) -> list[SearchResult]:
    seen, kept = set(), []
    for result in results:
        if result.url.startswith(("http://", "https://")) and result.url not in seen:
            seen.add(result.url)
            kept.append(result)
    return kept[:limit]


class SearxngSearch:
    name = "searxng"

    def __init__(self, base_url: str, timeout: float = 20):
        # No proxy here: SearXNG usually runs locally and has its own outgoing proxy settings.
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout, trust_env=False)

    def search(
        self, query: str, kind: SearchKind = "web", limit: int = 10, language: str | None = None
    ) -> list[SearchResult]:
        params = {
            "q": query,
            "format": "json",
            "categories": "videos" if kind == "video" else "general",
        }
        if language:
            params["language"] = language
        try:
            response = self.client.get(f"{self.base_url}/search", params=params)
        except httpx.HTTPError as e:
            raise SearchError(
                f"SearXNG is not reachable at {self.base_url} ({type(e).__name__}). "
                "Start it (make searxng) or set TINYLM_SEARXNG_URL."
            ) from None
        if response.status_code == 403:
            raise SearchError(
                "SearXNG refused the JSON format: add 'json' to search.formats in its settings.yml."
            )
        if response.status_code != 200:
            raise SearchError(f"SearXNG answered HTTP {response.status_code}")
        results = [
            SearchResult(
                title=item.get("title") or item.get("url", ""),
                url=item.get("url", ""),
                snippet=item.get("content") or "",
                kind=kind,
                thumbnail=item.get("thumbnail") or item.get("img_src") or None,
                source=item.get("engine", ""),
            )
            for item in response.json().get("results", [])
        ]
        return unique(results, limit)

    def close(self) -> None:
        self.client.close()


class BraveSearch:
    name = "brave"

    def __init__(
        self,
        api_key: str,
        proxy: str | None = None,
        base_url: str = "https://api.search.brave.com/res/v1",
        timeout: float = 20,
    ):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(
            proxy=proxy,
            timeout=timeout,
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        )

    def search(
        self, query: str, kind: SearchKind = "web", limit: int = 10, language: str | None = None
    ) -> list[SearchResult]:
        params: dict = {"q": query, "count": min(limit, 20)}
        if language:
            params["search_lang"] = language
        path = "videos/search" if kind == "video" else "web/search"
        try:
            response = self.client.get(f"{self.base_url}/{path}", params=params)
        except httpx.HTTPError as e:
            raise SearchError(f"Brave Search is not reachable ({type(e).__name__}: {e})") from None
        if response.status_code in (401, 403):
            raise SearchError("Brave Search rejected the API key (TINYLM_BRAVE_API_KEY)")
        if response.status_code != 200:
            raise SearchError(f"Brave Search answered HTTP {response.status_code}")
        body = response.json()
        items = (
            body.get("results", []) if kind == "video" else body.get("web", {}).get("results", [])
        )
        results = [
            SearchResult(
                title=item.get("title") or item.get("url", ""),
                url=item.get("url", ""),
                snippet=item.get("description") or "",
                kind=kind,
                thumbnail=(item.get("thumbnail") or {}).get("src"),
                source="brave",
            )
            for item in items
        ]
        return unique(results, limit)

    def close(self) -> None:
        self.client.close()


def link_document(result: SearchResult) -> tuple[str, str]:
    """A small document for a search result kept as a link (e.g. a trailer), so answers can
    cite it: (name, text)."""
    label = "Video" if result.kind == "video" else "Link"
    lines = [f"{label}: {result.title}", f"URL: {result.url}"]
    if result.snippet:
        lines.append(result.snippet)
    return result.title, "\n".join(lines)
