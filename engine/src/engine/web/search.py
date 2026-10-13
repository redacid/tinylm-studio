"""Web search for finding pages and videos to add to a collection.

Providers:
- SearXNG (default): a self-hosted metasearch engine. Free, no key; it queries Google, Bing,
  DuckDuckGo, YouTube and others itself. Its JSON output must be enabled (see docs/api.md).
- Brave Search API: an independent index; web and video search.
- Tavily, Exa and Perplexity: search APIs made for LLM applications; web pages only, so video
  search goes through them restricted to video sites.
- Serper: Google's results; web and video search.

Domain filters (include only these sites, never these sites) hold domains (`example.com`, which
covers its subdomains too) and patterns with `*` (`*.ua`, `*.google.*`, `steam*.com`). They are
passed to the providers whose APIs take them (Tavily, Exa, Perplexity) as far as those APIs
understand them. For the others, an include list becomes `site:` queries. Every result is
checked here as well, fetching more pages when filtering leaves too few.
"""

import re
from dataclasses import asdict, dataclass, field, replace
from fnmatch import fnmatchcase
from typing import Literal, Protocol
from urllib.parse import urlsplit

import httpx

SearchKind = Literal["web", "video"]

# Video search through providers without one: web search on these sites.
VIDEO_SITES = ("youtube.com", "vimeo.com")
# An include list becomes one `site:` query per domain, for at most this many domains.
MAX_SITE_QUERIES = 5
DOMAIN = re.compile(r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9-]{2,63}$")
# A pattern: labels of letters, digits, '-' and '*', with at least one letter or digit.
PATTERN = re.compile(r"^(?=.{1,253}$)(?=.*[a-z0-9])[a-z0-9*-]+(\.[a-z0-9*-]+)*$")
# "*.ua", "*.com": every site under a top-level domain.
TOP_LEVEL = re.compile(r"^\*\.([a-z0-9-]+)$")


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


def normalize_domain(text: str) -> str | None:
    """A domain or pattern as kept in the filter, or None when it is neither:
    'https://www.Example.com/path' -> 'example.com', '*.example.com' -> 'example.com' (a domain
    covers its subdomains anyway), '.ua' -> '*.ua', '*.Google.*' -> '*.google.*'."""
    text = text.strip().lower()
    if "://" in text:
        text = urlsplit(text.replace("*", "wildcard-")).hostname or ""
        text = text.replace("wildcard-", "*")
    text = text.split("/")[0].split(":")[0]
    if text.startswith("."):
        text = "*" + text
    if text.startswith("*.") and "*" not in text[2:] and "." in text[2:]:
        text = text[2:]
    text = text.removeprefix("www.")
    try:
        # кіно.укр -> xn--...; label by label, so '*' stays as it is.
        labels = [part if "*" in part else part.encode("idna").decode() for part in text.split(".")]
    except UnicodeError:
        return None
    text = ".".join(labels)
    if "*" in text:
        return re.sub(r"\*+", "*", text) if PATTERN.match(text) else None
    return text if DOMAIN.match(text) else None


def matches(host: str, domain: str) -> bool:
    """A domain covers itself and its subdomains; a pattern's '*' stands for any characters,
    dots included, and a leading "*." may also match nothing."""
    host = host.lower().removeprefix("www.")
    if "*" in domain:
        # "*.google.*" also covers google.co.uk itself, as "*.ua" covers a bare "ua".
        return fnmatchcase(host, domain) or (
            domain.startswith("*.") and fnmatchcase(host, domain[2:])
        )
    return host == domain or host.endswith("." + domain)


def site_operator(domain: str) -> str | None:
    """The `site:` term search engines understand for a domain or pattern, if there is one."""
    if "*" not in domain:
        return domain
    top = TOP_LEVEL.match(domain)
    return f".{top.group(1)}" if top else None


@dataclass(frozen=True)
class DomainFilter:
    include: tuple[str, ...] = ()  # only these sites; any site when empty
    exclude: tuple[str, ...] = ()  # never these sites

    def __bool__(self) -> bool:
        return bool(self.include or self.exclude)

    def native_include(self, top_level: bool = False) -> list[str]:
        """The include list for an API that takes domains (and, with top_level, '.ua' for *.ua).
        Empty when a pattern does not translate: the API must not narrow the search then, and
        the results are filtered here instead."""
        translated = [site_operator(d) if top_level else d for d in self.include]
        if any(d is None or "*" in d for d in translated):
            return []
        return translated

    def native_exclude(self, top_level: bool = False) -> list[str]:
        """The part of the exclude list an API can take; the rest is filtered here."""
        translated = [site_operator(d) if top_level else d for d in self.exclude]
        return [d for d in translated if d is not None and "*" not in d]

    def allows(self, url: str) -> bool:
        host = urlsplit(url).hostname or ""
        if any(matches(host, d) for d in self.exclude):
            return False
        return not self.include or any(matches(host, d) for d in self.include)


ANY_DOMAIN = DomainFilter()


class SearchProvider(Protocol):
    name: str
    # Takes the domain filter in its API, so the query is not rewritten with `site:`.
    native_domains: bool
    videos: bool  # has a video search of its own
    max_pages: int  # pages fetched when filtering here leaves too few results

    def search(
        self,
        query: str,
        kind: SearchKind = "web",
        limit: int = 10,
        language: str | None = None,
        page: int = 1,
        domains: DomainFilter = ANY_DOMAIN,
    ) -> list[SearchResult]: ...

    def close(self) -> None: ...


@dataclass
class SearchOutcome:
    results: list[SearchResult]
    hidden: int = 0  # results dropped by the domain filter
    queries: list[str] = field(default_factory=list)


def search(
    provider: SearchProvider,
    query: str,
    kind: SearchKind = "web",
    limit: int = 10,
    language: str | None = None,
    domains: DomainFilter = ANY_DOMAIN,
) -> SearchOutcome:
    """Searches with the domain filter applied, whatever the provider supports."""
    as_video = kind == "video" and not provider.videos
    if as_video:
        domains = replace(domains, include=domains.include or VIDEO_SITES)
        kind = "web"
    queries = [query]
    if domains.include and not provider.native_domains and kind == "web":
        sites = [site_operator(d) for d in domains.include]
        # Only when every included site can be asked for; otherwise one plain query, filtered.
        if all(sites):
            queries = [f"{query} site:{d}" for d in sites[:MAX_SITE_QUERIES]]
    pages = provider.max_pages if domains else 1

    found: list[list[SearchResult]] = []
    hidden = 0
    for text in queries:
        kept: list[SearchResult] = []
        for page in range(1, pages + 1):
            batch = provider.search(text, kind, limit, language, page, domains)
            allowed = [r for r in batch if domains.allows(r.url)]
            hidden += len(batch) - len(allowed)
            kept += allowed
            if not batch or len(unique(kept, limit)) >= limit:
                break
        found.append(kept)
    # Interleaved, so each site of an include list gets its share of the limit.
    merged = [r for group in zip_longest_lists(found) for r in group]
    results = unique(merged, limit)
    if as_video:
        results = [replace(r, kind="video") for r in results]
    return SearchOutcome(results, hidden, queries)


def zip_longest_lists(lists: list[list[SearchResult]]) -> list[list[SearchResult]]:
    longest = max((len(items) for items in lists), default=0)
    return [[items[i] for items in lists if i < len(items)] for i in range(longest)]


def unique(results: list[SearchResult], limit: int) -> list[SearchResult]:
    seen, kept = set(), []
    for result in results:
        if result.url.startswith(("http://", "https://")) and result.url not in seen:
            seen.add(result.url)
            kept.append(result)
    return kept[:limit]


class SearxngSearch:
    name = "searxng"
    native_domains = False
    videos = True
    max_pages = 3

    def __init__(self, base_url: str, timeout: float = 20):
        # No proxy here: SearXNG usually runs locally and has its own outgoing proxy settings.
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout, trust_env=False)

    def search(
        self,
        query: str,
        kind: SearchKind = "web",
        limit: int = 10,
        language: str | None = None,
        page: int = 1,
        domains: DomainFilter = ANY_DOMAIN,
    ) -> list[SearchResult]:
        params = {
            "q": query,
            "format": "json",
            "categories": "videos" if kind == "video" else "general",
            "pageno": page,
        }
        if language:
            params["language"] = language
        try:
            response = self.client.get(f"{self.base_url}/search", params=params)
        except httpx.HTTPError as e:
            raise SearchError(
                f"SearXNG is not reachable at {self.base_url} ({type(e).__name__}). "
                "Start it (make searxng) or change its address in Settings > Search."
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


class HostedSearch:
    """A search API on the internet with an API key; goes through the proxy when one is set."""

    name = ""
    title = ""
    key_setting = ""

    def __init__(self, api_key: str, base_url: str, proxy: str | None, timeout: float, headers):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(proxy=proxy, timeout=timeout, headers=headers)

    def request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self.client.request(method, f"{self.base_url}/{path}", **kwargs)
        except httpx.HTTPError as e:
            raise SearchError(f"{self.title} is not reachable ({type(e).__name__}: {e})") from None
        if response.status_code in (401, 403):
            raise SearchError(
                f"{self.title} rejected the API key: check it in Settings > Search "
                f"(or {self.key_setting})."
            )
        if response.status_code == 429:
            raise SearchError(f"{self.title}: too many requests or the plan's limit is reached.")
        if response.status_code != 200:
            raise SearchError(
                f"{self.title} answered HTTP {response.status_code}: {response.text[:200]}"
            )
        return response.json()

    def close(self) -> None:
        self.client.close()


class BraveSearch(HostedSearch):
    name = "brave"
    title = "Brave Search"
    key_setting = "TINYLM_BRAVE_API_KEY"
    native_domains = False
    videos = True
    max_pages = 3

    def __init__(
        self,
        api_key: str,
        proxy: str | None = None,
        base_url: str = "https://api.search.brave.com/res/v1",
        timeout: float = 20,
    ):
        headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
        super().__init__(api_key, base_url, proxy, timeout, headers)

    def search(
        self,
        query: str,
        kind: SearchKind = "web",
        limit: int = 10,
        language: str | None = None,
        page: int = 1,
        domains: DomainFilter = ANY_DOMAIN,
    ) -> list[SearchResult]:
        params: dict = {"q": query, "count": min(limit, 20), "offset": page - 1}
        if language:
            params["search_lang"] = language
        path = "videos/search" if kind == "video" else "web/search"
        body = self.request("GET", path, params=params)
        items = (
            body.get("results", []) if kind == "video" else body.get("web", {}).get("results", [])
        )
        return [
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


class TavilySearch(HostedSearch):
    name = "tavily"
    title = "Tavily"
    key_setting = "TINYLM_TAVILY_API_KEY"
    native_domains = True
    videos = False
    max_pages = 1  # no pages; the filter is applied by Tavily itself

    def __init__(
        self,
        api_key: str,
        proxy: str | None = None,
        depth: str = "basic",
        topic: str = "general",
        base_url: str = "https://api.tavily.com",
        timeout: float = 30,
    ):
        super().__init__(api_key, base_url, proxy, timeout, {"Authorization": f"Bearer {api_key}"})
        self.depth, self.topic = depth, topic

    def search(
        self,
        query: str,
        kind: SearchKind = "web",
        limit: int = 10,
        language: str | None = None,
        page: int = 1,
        domains: DomainFilter = ANY_DOMAIN,
    ) -> list[SearchResult]:
        body: dict = {
            "query": query,
            "max_results": min(limit, 20),
            "search_depth": self.depth,
            "topic": self.topic,
        }
        if include := domains.native_include():
            body["include_domains"] = include[:300]
        if exclude := domains.native_exclude():
            body["exclude_domains"] = exclude[:150]
        items = self.request("POST", "search", json=body).get("results", [])
        return [
            SearchResult(
                title=item.get("title") or item.get("url", ""),
                url=item.get("url", ""),
                snippet=item.get("content") or "",
                source="tavily",
            )
            for item in items
        ]


class ExaSearch(HostedSearch):
    name = "exa"
    title = "Exa"
    key_setting = "TINYLM_EXA_API_KEY"
    native_domains = True
    videos = False
    max_pages = 1

    def __init__(
        self,
        api_key: str,
        proxy: str | None = None,
        search_type: str = "auto",
        base_url: str = "https://api.exa.ai",
        timeout: float = 30,
    ):
        super().__init__(api_key, base_url, proxy, timeout, {"x-api-key": api_key})
        self.search_type = search_type

    def search(
        self,
        query: str,
        kind: SearchKind = "web",
        limit: int = 10,
        language: str | None = None,
        page: int = 1,
        domains: DomainFilter = ANY_DOMAIN,
    ) -> list[SearchResult]:
        body: dict = {
            "query": query,
            "numResults": min(limit, 100),
            "type": self.search_type,
            "contents": {"highlights": True},
        }
        if include := domains.native_include():
            body["includeDomains"] = include
        if exclude := domains.native_exclude():
            body["excludeDomains"] = exclude
        items = self.request("POST", "search", json=body).get("results", [])
        return [
            SearchResult(
                title=item.get("title") or item.get("url", ""),
                url=item.get("url", ""),
                snippet=" … ".join(item.get("highlights") or []) or item.get("text") or "",
                thumbnail=item.get("image"),
                source="exa",
            )
            for item in items
        ]


class PerplexitySearch(HostedSearch):
    name = "perplexity"
    title = "Perplexity"
    key_setting = "TINYLM_PERPLEXITY_API_KEY"
    native_domains = True
    videos = False
    max_pages = 1

    def __init__(
        self,
        api_key: str,
        proxy: str | None = None,
        base_url: str = "https://api.perplexity.ai",
        timeout: float = 30,
    ):
        super().__init__(api_key, base_url, proxy, timeout, {"Authorization": f"Bearer {api_key}"})

    def search(
        self,
        query: str,
        kind: SearchKind = "web",
        limit: int = 10,
        language: str | None = None,
        page: int = 1,
        domains: DomainFilter = ANY_DOMAIN,
    ) -> list[SearchResult]:
        body: dict = {"query": query, "max_results": min(limit, 20)}
        if language:
            body["search_language_filter"] = [language]
        # Up to 20 domains or top-level domains (".ua"), either all included or all excluded
        # ("-site.com"); the rest of the filter is applied to the results afterwards.
        include = domains.native_include(top_level=True)
        if include and len(include) <= 20:
            body["search_domain_filter"] = include
        elif exclude := domains.native_exclude(top_level=True):
            body["search_domain_filter"] = ["-" + d for d in exclude][:20]
        items = self.request("POST", "search", json=body).get("results", [])
        return [
            SearchResult(
                title=item.get("title") or item.get("url", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet") or "",
                source="perplexity",
            )
            for item in items
        ]


class SerperSearch(HostedSearch):
    name = "serper"
    title = "Serper"
    key_setting = "TINYLM_SERPER_API_KEY"
    native_domains = False
    videos = True
    max_pages = 3

    def __init__(
        self,
        api_key: str,
        proxy: str | None = None,
        country: str | None = None,
        base_url: str = "https://google.serper.dev",
        timeout: float = 20,
    ):
        super().__init__(api_key, base_url, proxy, timeout, {"X-API-KEY": api_key})
        self.country = country

    def search(
        self,
        query: str,
        kind: SearchKind = "web",
        limit: int = 10,
        language: str | None = None,
        page: int = 1,
        domains: DomainFilter = ANY_DOMAIN,
    ) -> list[SearchResult]:
        body: dict = {"q": query, "num": min(limit, 100), "page": page}
        if self.country:
            body["gl"] = self.country
        if language:
            body["hl"] = language
        found = self.request("POST", "videos" if kind == "video" else "search", json=body)
        items = found.get("videos" if kind == "video" else "organic", [])
        return [
            SearchResult(
                title=item.get("title") or item.get("link", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet") or "",
                kind=kind,
                thumbnail=item.get("imageUrl"),
                source=item.get("source") or "google",
            )
            for item in items
        ]


def link_document(result: SearchResult) -> tuple[str, str]:
    """A small document for a search result kept as a link (e.g. a trailer), so answers can
    cite it: (name, text)."""
    label = "Video" if result.kind == "video" else "Link"
    lines = [f"{label}: {result.title}", f"URL: {result.url}"]
    if result.snippet:
        lines.append(result.snippet)
    return result.title, "\n".join(lines)
