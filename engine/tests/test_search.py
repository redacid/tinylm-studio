"""Web search: SearXNG and Brave providers, the search API and keeping results as links."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from engine.web.search import BraveSearch, SearchError, SearxngSearch


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
