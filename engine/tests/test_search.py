"""Web search: SearXNG and Brave providers, the search API and keeping results as links."""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from engine.web.search import (
    VIDEO_SITES,
    BraveSearch,
    DomainFilter,
    ExaSearch,
    PerplexitySearch,
    SearchError,
    SearxngSearch,
    SerperSearch,
    TavilySearch,
    normalize_domain,
    search,
)


def item(title: str, url: str, content: str = "", engine: str = "bing", **extra) -> dict:
    return {"title": title, "url": url, "content": content, "engine": engine, **extra}


SEARXNG_RESULTS = {
    "general": [
        item("Elden Ring review", "https://games.example/elden", "A masterpiece."),
        item("Elden Ring review (dup)", "https://games.example/elden", engine="google"),
        item("Elden Ring wiki", "https://wiki.example/elden", "Bosses and maps."),
        item("Not a web page", "magnet:?xt=urn:btih:1"),
    ],
    "videos": [
        item(
            "Elden Ring trailer",
            "https://www.youtube.com/watch?v=abc",
            "Official trailer.",
            "youtube",
            thumbnail="https://i.ytimg.com/abc.jpg",
        ),
    ],
}


class FakeSearch(BaseHTTPRequestHandler):
    """Answers like SearXNG on /search and like the Brave API on /res/v1/..."""

    requests: list[tuple[str, dict, str | None]] = []

    def do_GET(self):  # noqa: N802
        url = urlsplit(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        FakeSearch.requests.append((url.path, query, self.headers.get("X-Subscription-Token")))
        if url.path == "/search":
            if query.get("format") != "json" or query["q"] == "json-disabled":
                return self.reply(403, {})
            return self.reply(200, {"results": SEARXNG_RESULTS[query["categories"]]})
        if self.headers.get("X-Subscription-Token") != "good-key":
            return self.reply(401, {})
        if url.path == "/res/v1/web/search":
            items = [
                {"title": "Brave result", "url": "https://brave.example/a", "description": "Hi"}
            ]
            return self.reply(200, {"web": {"results": items}})
        if url.path == "/res/v1/videos/search":
            items = [{"title": "Trailer", "url": "https://v.example/t", "thumbnail": {"src": "x"}}]
            return self.reply(200, {"results": items})
        self.reply(404, {})

    def reply(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def search_server():
    FakeSearch.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeSearch)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_searxng_finds_pages_and_videos(search_server):
    searxng = SearxngSearch(search_server)
    pages = searxng.search("elden ring", language="uk")
    assert [p.url for p in pages] == ["https://games.example/elden", "https://wiki.example/elden"]
    assert pages[0].to_dict()["site"] == "games.example"
    assert FakeSearch.requests[0][1]["language"] == "uk"

    videos = searxng.search("elden ring trailer", kind="video")
    assert videos[0].kind == "video" and videos[0].thumbnail.endswith("abc.jpg")
    assert len(searxng.search("elden ring", limit=1)) == 1


def test_searxng_errors_say_what_to_fix(search_server):
    with pytest.raises(SearchError, match="not reachable"):
        SearxngSearch("http://127.0.0.1:9").search("x")

    # A default SearXNG install has JSON output switched off and answers 403.
    with pytest.raises(SearchError, match="settings.yml"):
        SearxngSearch(search_server).search("json-disabled")


def test_brave_search(search_server):
    brave = BraveSearch("good-key", base_url=f"{search_server}/res/v1")
    assert [r.url for r in brave.search("elden ring")] == ["https://brave.example/a"]
    assert brave.search("trailer", kind="video")[0].thumbnail == "x"
    with pytest.raises(SearchError, match="API key"):
        BraveSearch("bad", base_url=f"{search_server}/res/v1").search("x")


def test_search_api_and_links(client, settings, search_server):
    settings.searxng_url = search_server
    found = client.post("/api/search", json={"query": "elden ring", "kind": "video"})
    assert found.status_code == 200
    assert found.json()["provider"] == "searxng"
    trailer = found.json()["data"][0]
    assert trailer["url"] == "https://www.youtube.com/watch?v=abc"

    client.post("/api/collections", json={"name": "games"})
    added = client.post("/api/collections/games/links", json={"links": [trailer]})
    assert added.json()["data"][0]["chunks"] == 1
    documents = client.get("/api/collections/games/documents").json()["data"]
    assert documents[0]["source"] == trailer["url"]
    hits = client.post("/api/collections/games/search", json={"query": "Elden Ring trailer"})
    assert "URL: https://www.youtube.com/watch?v=abc" in hits.json()["data"][0]["text"]

    assert (
        client.post("/api/collections/missing/links", json={"links": [trailer]}).status_code == 404
    )
    bad = client.post(
        "/api/collections/games/links", json={"links": [{"title": "x", "url": "ftp://x"}]}
    )
    assert bad.status_code == 422

    settings.searxng_url = "http://127.0.0.1:9"
    down = client.post("/api/search", json={"query": "elden ring"})
    assert down.status_code == 503 and "not reachable" in down.json()["detail"]
    settings.search_provider = "brave"
    assert "BRAVE_API_KEY" in client.post("/api/search", json={"query": "x"}).json()["detail"]


# Domain filters and the other providers


def site_results(query: str, page: int) -> list[dict]:
    """Ten results per page from a mix of sites; `site:x` keeps only x, like real engines."""
    sites = ["games.example", "www.wiki.example", "spam.example", "news.wiki.example"]
    wanted = re.findall(r"site:(\S+)", query)
    results = [
        item(f"{site} {page}-{i}", f"https://{site}/{page}/{i}")
        for i, site in enumerate(sites * 3)
        if not wanted or any(site == w or site.endswith("." + w) for w in wanted)
    ]
    return results[:10]


class FakeProviders(BaseHTTPRequestHandler):
    """SearXNG with pages, and the Tavily, Exa, Perplexity and Serper APIs."""

    requests: list[tuple[str, dict]] = []

    def do_GET(self):  # noqa: N802
        url = urlsplit(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        FakeProviders.requests.append((url.path, query))
        page = int(query.get("pageno", 1))
        self.reply({"results": site_results(query["q"], page) if page <= 3 else []})

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeProviders.requests.append((self.path, body))
        if self.headers.get("Authorization") == "Bearer bad":
            return self.reply({}, 401)
        url = lambda site, i: f"https://{site}/{i}"  # noqa: E731
        if self.path == "/search" and "query" in body and "numResults" in body:  # Exa
            items = [{"title": "Exa", "url": url("wiki.example", 1), "highlights": ["a", "b"]}]
        elif self.path == "/search" and "search_depth" in body:  # Tavily
            items = [{"title": "Tavily", "url": url("www.youtube.com", 1), "content": "T"}]
        elif self.path == "/search" and "query" in body:  # Perplexity
            items = [
                {"title": "P", "url": url("wiki.example", 1), "snippet": "S"},
                {"title": "Spam", "url": url("spam.example", 2), "snippet": "S"},
            ]
        else:  # Serper
            key = "videos" if self.path == "/videos" else "organic"
            items = [{"title": r["title"], "link": r["url"]} for r in site_results(body["q"], 1)]
            return self.reply({key: items})
        self.reply({"results": items})

    def reply(self, body: dict, status: int = 200) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def providers():
    FakeProviders.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeProviders)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_domains_are_normalized_and_matched():
    assert normalize_domain(" https://www.Wiki.Example/page?x=1 ") == "wiki.example"
    assert normalize_domain("*.youtube.com") == "youtube.com"
    assert normalize_domain("кіно.укр").endswith(".xn--j1amh")  # stored as punycode
    assert normalize_domain("not a domain") is None and normalize_domain("localhost") is None

    domains = DomainFilter(include=("wiki.example",), exclude=("news.wiki.example",))
    assert domains.allows("https://wiki.example/a") and domains.allows("https://www.wiki.example/")
    assert not domains.allows("https://news.wiki.example/a")  # excluded wins
    assert not domains.allows("https://otherwiki.example/a")  # not a subdomain
    assert DomainFilter().allows("https://anything.example/") and not DomainFilter()


def test_domain_patterns():
    assert [normalize_domain(d) for d in ("*.ua", ".UA", "*.Google.*", "steam*.com")] == [
        "*.ua",
        "*.ua",
        "*.google.*",
        "steam*.com",
    ]
    assert normalize_domain("*.com.ua") == "com.ua"  # a plain domain covers its subdomains
    assert normalize_domain("*.укр") == "*.xn--j1amh"
    assert all(normalize_domain(d) is None for d in ("*", "*.*", "ua"))

    ua = DomainFilter(include=("*.ua",))
    assert ua.allows("https://www.ign.com.ua/a") and ua.allows("https://rozetka.ua/")
    assert not ua.allows("https://ign.com/a") and not ua.allows("https://ua.example.com/")
    google = DomainFilter(exclude=("*.google.*",))
    assert not google.allows("https://www.google.co.uk/") and not google.allows(
        "https://maps.google.com/"
    )
    assert google.allows("https://googleblog.com/")
    steam = DomainFilter(include=("steam*.com",))
    assert steam.allows("https://steampowered.com/") and steam.allows("https://steamcommunity.com/")
    assert not steam.allows("https://store.steampowered.com/")  # '*' does not start before 'steam'


def test_patterns_in_site_queries_and_native_filters(providers):
    searxng = SearxngSearch(providers)
    outcome = search(searxng, "q", domains=DomainFilter(include=("*.ua", "games.example")))
    assert outcome.queries == ["q site:.ua", "q site:games.example"]
    # A pattern search engines cannot take: one plain query, filtered here.
    outcome = search(searxng, "q", domains=DomainFilter(include=("*games*",)))
    assert outcome.queries == ["q"] and {r.site for r in outcome.results} == {"games.example"}

    domains = DomainFilter(include=("*.ua", "wiki.example"), exclude=("spam.*", "x.example"))
    search(ExaSearch("k", base_url=providers), "q", domains=domains)
    body = FakeProviders.requests[-1][1]
    assert "includeDomains" not in body and body["excludeDomains"] == ["x.example"]
    search(PerplexitySearch("k", base_url=providers), "q", domains=domains)
    assert FakeProviders.requests[-1][1]["search_domain_filter"] == [".ua", "wiki.example"]
    search(PerplexitySearch("k", base_url=providers), "q", domains=DomainFilter(exclude=("*.ru",)))
    assert FakeProviders.requests[-1][1]["search_domain_filter"] == ["-.ru"]


def test_searxng_include_uses_site_queries_and_checks_results(providers):
    searxng = SearxngSearch(providers)
    domains = DomainFilter(include=("games.example", "wiki.example"))
    outcome = search(searxng, "elden ring", limit=6, domains=domains)
    assert outcome.queries == ["elden ring site:games.example", "elden ring site:wiki.example"]
    sites = [r.site for r in outcome.results]
    assert len(sites) == 6 and set(sites) <= {
        "games.example",
        "www.wiki.example",
        "news.wiki.example",
    }
    assert sites[0] == "games.example" and "wiki.example" in sites[1]  # interleaved


def test_excluded_sites_are_dropped_and_more_pages_fetched(providers):
    searxng = SearxngSearch(providers)
    outcome = search(
        searxng, "elden ring", limit=12, domains=DomainFilter(exclude=("wiki.example",))
    )
    # Each page of 10 has 5 results from wiki.example, so three pages give the 12 asked for.
    assert len(outcome.results) == 12 and outcome.hidden == 15
    assert {r.site for r in outcome.results} == {"games.example", "spam.example"}
    assert [q["pageno"] for _, q in FakeProviders.requests] == ["1", "2", "3"]

    FakeProviders.requests = []
    plain = search(searxng, "elden ring", limit=5)
    assert len(plain.results) == 5 and plain.hidden == 0 and len(FakeProviders.requests) == 1


def test_providers_with_domain_filters_of_their_own(providers):
    domains = DomainFilter(include=("wiki.example",), exclude=("spam.example",))
    exa = search(ExaSearch("k", base_url=providers), "q", limit=5, domains=domains)
    body = FakeProviders.requests[-1][1]
    assert body["includeDomains"] == ["wiki.example"] and body["excludeDomains"] == ["spam.example"]
    assert exa.queries == ["q"] and exa.results[0].snippet == "a … b"

    # Perplexity takes either an allow list or a deny list ("-site"); the rest is checked here.
    perplexity = PerplexitySearch("k", base_url=providers)
    found = search(perplexity, "q", domains=domains, language="uk")
    body = FakeProviders.requests[-1][1]
    assert body["search_domain_filter"] == ["wiki.example"] and body["search_language_filter"] == [
        "uk"
    ]
    assert [r.site for r in found.results] == ["wiki.example"] and found.hidden == 1
    search(perplexity, "q", domains=DomainFilter(exclude=("spam.example",)))
    assert FakeProviders.requests[-1][1]["search_domain_filter"] == ["-spam.example"]

    # No video search of its own: web search on video sites.
    tavily = TavilySearch("k", depth="advanced", base_url=providers)
    videos = search(tavily, "trailer", kind="video")
    body = FakeProviders.requests[-1][1]
    assert body["include_domains"] == list(VIDEO_SITES) and body["search_depth"] == "advanced"
    assert videos.results[0].kind == "video"
    with pytest.raises(SearchError, match="rejected the API key"):
        search(TavilySearch("bad", base_url=providers), "q")

    serper = SerperSearch("k", country="ua", base_url=providers)
    found = search(
        serper, "q", limit=3, language="uk", domains=DomainFilter(include=("games.example",))
    )
    path, body = FakeProviders.requests[-1]
    assert path == "/search" and body == {
        "q": "q site:games.example",
        "num": 3,
        "page": 1,
        "gl": "ua",
        "hl": "uk",
    }
    assert {r.site for r in found.results} == {"games.example"}
    assert search(serper, "q", kind="video").results[0].kind == "video"


def test_search_settings_api(client, settings, providers):
    settings.searxng_url = providers
    current = client.get("/api/search/settings").json()
    assert current["provider"] == "searxng" and current["include_domains"] == []
    assert current["brave_api_key_set"] is False and "brave_api_key" not in current

    saved = client.put(
        "/api/search/settings",
        json={
            "exclude_domains": ["https://www.wiki.example/x", "", "wiki.example"],
            "tavily_api_key": "tvly-secret",
        },
    ).json()
    assert saved["exclude_domains"] == ["wiki.example"] and saved["tavily_api_key_set"]
    stored = (settings.workspace_dir / "search-settings.json").read_text()
    assert "tvly-secret" in stored and "tvly-secret" not in json.dumps(
        client.get("/api/search/settings").json()
    )

    found = client.post("/api/search", json={"query": "elden ring", "limit": 4}).json()
    assert found["exclude_domains"] == ["wiki.example"] and found["hidden"] > 0
    assert all("wiki" not in r["site"] for r in found["data"])
    # The request's own filter wins for that search.
    only = client.post(
        "/api/search",
        json={
            "query": "elden ring",
            "limit": 2,
            "include_domains": ["games.example"],
            "exclude_domains": [],
        },
    ).json()
    assert {r["site"] for r in only["data"]} == {"games.example"} and only["queries"] == [
        "elden ring site:games.example"
    ]

    bad = client.put(
        "/api/search/settings", json={"include_domains": ["ok.example", "not a domain"]}
    )
    assert bad.status_code == 422 and "not a domain" in bad.json()["detail"]
    assert "pattern: *" in client.put("/api/search/settings", json={"exclude_domains": ["*"]}).text
    patterns = client.put("/api/search/settings", json={"exclude_domains": [".RU", "*.google.*"]})
    assert patterns.json()["exclude_domains"] == ["*.ru", "*.google.*"]
    assert client.put("/api/search/settings", json={"color": "red"}).status_code == 422
    assert client.put("/api/search/settings", json={"provider": "bing"}).status_code == 422

    # The key is saved; switching to a provider without one says where to add it.
    client.put("/api/search/settings", json={"provider": "exa"})
    missing = client.post("/api/search", json={"query": "x"})
    assert missing.status_code == 503 and "Settings > Search" in missing.json()["detail"]
    cleared = client.put(
        "/api/search/settings", json={"tavily_api_key": "", "provider": "searxng"}
    ).json()
    assert cleared["tavily_api_key_set"] is False
