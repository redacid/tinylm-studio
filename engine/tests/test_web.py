"""Website crawling: link following, politeness, proxies and the browser (CDP) fetcher."""

import json
import os
import shutil
import socket
import socketserver
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from engine.web.crawl_settings import CrawlSettings
from engine.web.crawler import Crawler, CrawlJob, CrawlOptions
from engine.web.fetch import CdpFetcher, HttpFetcher
from engine.web.politeness import RateLimiter

ARTICLE = "<p>" + "Geralt hunts monsters across the Continent for coin. " * 12 + "</p>"


def page(title: str, body: str) -> str:
    return f"<html><head><title>{title}</title></head><body><article>{body}</article></body></html>"


SITE = {
    "/": page(
        "Home",
        '<a href="/a">A</a> <a href="/b#top">B</a> <a href="/private/x">P</a> '
        '<a href="/shot.png">img</a> <a href="https://elsewhere.example/">out</a>',
    ),
    "/a": page("Article A", ARTICLE.replace("Geralt", "Ciri") + '<a href="/">home</a>'),
    "/b": page("Article B", ARTICLE + '<a href="/c">next</a>'),
    "/c": page("Article C", ARTICLE.replace("Geralt", "Yennefer") + '<a href="/d">deeper</a>'),
    "/d": page("Article D", ARTICLE.replace("Geralt", "Dandelion")),
    "/private/x": page("Private", ARTICLE),
    "/guides/": page(
        "Guides",
        '<a href="/guides/one">1</a> <a href="/guides/deep/two">2</a> <a href="/a">A</a>',
    ),
    "/guides/one": page("Guide One", ARTICLE.replace("Geralt", "Vesemir")),
    "/guides/deep/two": page("Guide Two", ARTICLE.replace("Geralt", "Lambert")),
    "/robots.txt": "User-agent: *\nDisallow: /private/\n",
    "/js": "<html><head><title>Rendered</title></head><body><div id=app></div><script>"
    "document.getElementById('app').innerHTML = '<article><p>' + "
    "'Rendered by JavaScript: Triss Merigold is a sorceress. '.repeat(12) + '</p></article>'"
    "</script></body></html>",
}


# Redirects: a path to the address it moved to; "@" stands for the other host name of the site's
# server (localhost and 127.0.0.1 are different sites).
REDIRECTS = {"/old": "/guides/", "/moved": "@/", "/a-moved": "@/a", "/filer/1/": "/media/2025.pdf"}
SITE["/meta"] = (
    '<html><head><meta http-equiv="Refresh" content="0; URL=\'/guides/\'"></head></html>'
)
SITE["/links"] = page(
    "Links", '<a href="/a-moved">A</a> <a href="/b">B</a> <a href="/filer/1/">PDF</a>'
)


class Site(BaseHTTPRequestHandler):
    requests: list[str] = []
    throttled: set[str] = set()

    def do_GET(self):  # noqa: N802
        Site.requests.append(self.path)
        if self.path == "/slow" and "/slow" not in Site.throttled:
            Site.throttled.add("/slow")
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.end_headers()
            return
        if self.path in REDIRECTS:
            host, port = self.server.server_address[:2]
            other = f"http://localhost:{port}" if host == "127.0.0.1" else ""
            self.send_response(301 if self.path == "/old" else 302)
            self.send_header("Location", REDIRECTS[self.path].replace("@", other))
            self.end_headers()
            return
        path = "/a" if self.path == "/slow" else self.path
        if path not in SITE:
            self.send_error(404)
            return
        body = SITE[path].encode()
        self.send_response(200)
        kind = "text/plain" if path.endswith(".txt") else "text/html; charset=utf-8"
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class ForwardProxy(BaseHTTPRequestHandler):
    """Plain HTTP proxy: clients send the absolute URL as the request path."""

    seen: list[str] = []

    def do_GET(self):  # noqa: N802
        ForwardProxy.seen.append(self.path)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(self.path) as response:
                status, headers, body = response.status, response.headers, response.read()
        except urllib.error.HTTPError as e:
            status, headers, body = e.code, e.headers, e.read()
        self.send_response(status)
        self.send_header("Content-Type", headers.get("Content-Type", "text/html"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class Socks5(socketserver.BaseRequestHandler):
    """Minimal SOCKS5 server: no authentication, CONNECT to an IPv4 address or domain name."""

    targets: list[tuple[str, int]] = []

    def handle(self):
        sock = self.request
        sock.recv(sock.recv(2)[1])  # greeting: version, number of methods, methods
        sock.sendall(b"\x05\x00")
        header = sock.recv(4)
        if header[3] == 1:
            host = socket.inet_ntoa(sock.recv(4))
        else:
            host = sock.recv(sock.recv(1)[0]).decode()
        port = int.from_bytes(sock.recv(2), "big")
        Socks5.targets.append((host, port))
        upstream = socket.create_connection((host, port))
        sock.sendall(b"\x05\x00\x00\x01" + bytes(4) + bytes(2))

        def pipe(src, dst):
            try:
                while data := src.recv(65536):
                    dst.sendall(data)
            except OSError:
                pass
            finally:
                dst.close()

        threading.Thread(target=pipe, args=(upstream, sock), daemon=True).start()
        pipe(sock, upstream)


def serve(server) -> str:
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    return f"{host}:{port}"


@pytest.fixture(autouse=True)
def no_system_proxy(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def site():
    Site.requests, Site.throttled = [], set()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Site)
    yield f"http://{serve(server)}"
    server.shutdown()


@pytest.fixture
def games(store):
    store.create_collection("games")
    return store


def crawl(
    store, urls, proxy=None, cdp_url="http://127.0.0.1:9", robots=True, **options
) -> CrawlJob:
    options.setdefault("per_minute", 6000)
    job = CrawlJob("games", CrawlOptions(urls, **options))
    current = CrawlSettings(user_agent="test", cdp_url=cdp_url, respect_robots=robots)
    Crawler(store, proxy, lambda: current).run(job)
    return job


def titles(store) -> list[str]:
    return sorted(d["name"] for d in store.documents("games"))


def test_crawl_follows_links_within_the_site(games, site):
    job = crawl(games, [f"{site}/"], max_depth=2)
    assert job.state == "done"
    # Home is too short to index; D is three hops away; private/ is disallowed by robots.txt.
    assert titles(games) == ["Article A", "Article B", "Article C"]
    assert not {"/private/x", "/shot.png", "/d"} & set(Site.requests)
    assert job.skipped >= 2
    assert games.documents("games")[0]["source"].startswith(site)


def test_only_under_the_start_address(games, site):
    job = crawl(games, [f"{site}/guides/"], max_depth=2, only_under_start=True)
    assert job.state == "done"
    assert titles(games) == ["Guide One", "Guide Two"]
    assert "/a" not in Site.requests


def test_a_start_address_that_redirects(games, site):
    """Crawled where it leads: /old to /guides/ (its folder counts as the start), /meta by a
    <meta http-equiv="refresh">, /moved to another host name."""
    for start in ("/old", "/meta"):
        job = crawl(games, [f"{site}{start}"], max_depth=2, only_under_start=True)
        assert titles(games) == ["Guide One", "Guide Two"], start
        assert any("redirects to" in line for line in job.log)
    job = crawl(games, [f"{site}/moved"], max_depth=1)
    assert "Article A" in titles(games)
    assert games.documents("games")[-1]["source"].startswith("http://localhost:")


def test_redirects_are_held_to_the_rules_of_links(games, site):
    """A page that redirects to another site, or to an excluded address, is skipped without
    downloading what the redirect leads to."""
    job = crawl(games, [f"{site}/links"], max_depth=1, exclude=["*.pdf"])
    assert titles(games) == ["Article B"]
    assert "/media/2025.pdf" not in Site.requests and "/filer/1/" in Site.requests
    assert any("/a-moved: it redirects to http://localhost:" in line for line in job.log)
    assert any("/filer/1/: it redirects to " in line for line in job.log)
    assert job.skipped >= 2


def test_exclude_patterns_skip_links(games, site):
    job = crawl(games, [f"{site}/guides/"], max_depth=2, exclude=["/DEEP/", "*/a"])
    assert titles(games) == ["Guide One"]
    assert not {"/a", "/guides/deep/two"} & set(Site.requests)
    assert any("are skipped" in line for line in job.log)


def test_exclude_pattern_matching():
    from engine.web.crawler import excluded

    url = "https://example.com/news/2024/page?sort=asc"
    assert excluded(url, ["?sort="])  # a part of the address; ? is not special
    assert excluded(url, ["*/NEWS/*"])  # * is any characters, case is ignored
    assert not excluded(url, ["*/news"])  # a pattern with * matches the whole address
    assert not excluded(url, ["/forum/"])


def test_whole_site_from_a_folder_by_default(games, site):
    crawl(games, [f"{site}/guides/"], max_depth=1)
    assert titles(games) == ["Article A", "Guide One", "Guide Two"]


def test_max_pages_and_robots_switch(games, site):
    job = crawl(games, [f"{site}/private/x"], max_pages=1, respect_robots=False)
    assert job.fetched == 1
    assert titles(games) == ["Private"]


def test_robots_txt_off_in_the_settings(games, site):
    job = crawl(games, [f"{site}/private/x"], max_depth=0, robots=False)
    assert titles(games) == ["Private"] and "/robots.txt" not in Site.requests
    assert "robots.txt is ignored (Settings > Crawler)" in job.log


def test_backs_off_when_the_site_says_too_many_requests(games, site):
    job = crawl(games, [f"{site}/slow"], max_depth=0)
    assert Site.requests.count("/slow") == 2
    assert any("slow down" in line for line in job.log)
    assert titles(games) == ["Article A"]


def test_cancel_stops_the_crawl(games, site):
    job = CrawlJob("games", CrawlOptions([f"{site}/"], per_minute=1))
    thread = threading.Thread(target=Crawler(games).run, args=(job,))
    thread.start()
    time.sleep(0.3)  # robots.txt and the first page; the next request waits a minute
    job.cancelled.set()
    thread.join(timeout=5)
    assert job.state == "cancelled"


def test_rate_limiter_spaces_requests_per_site():
    now = [0.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(per_minute=6, clock=lambda: now[0], sleep=sleep)
    limiter.wait("a.com")
    limiter.wait("b.com")  # another site is not delayed
    assert slept == []
    limiter.wait("a.com")
    assert 10 <= slept[0] <= 12  # 60 / 6 seconds plus up to 20% jitter
    limiter.set_min_delay("b.com", 30)  # e.g. robots.txt Crawl-delay
    limiter.wait("b.com")
    limiter.wait("b.com")
    assert slept[-1] >= 30 - 0.01


def test_http_proxy_is_used(games, site):
    ForwardProxy.seen = []
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), ForwardProxy)
    address = serve(proxy)
    crawl(games, [f"{site}/a"], proxy=f"http://{address}", max_depth=0)
    proxy.shutdown()
    assert f"{site}/a" in ForwardProxy.seen
    assert titles(games) == ["Article A"]


def test_socks_proxy_is_used(site):
    Socks5.targets = []
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Socks5)
    server.daemon_threads = True
    address = serve(server)
    fetcher = HttpFetcher(proxy=f"socks5://{address}")
    page = fetcher.fetch(f"{site}/a")
    server.shutdown()
    assert page.status == 200 and b"Article A" in page.body
    assert Socks5.targets and Socks5.targets[0][1] == int(site.rsplit(":", 1)[1])


def chromium() -> str | None:
    installed = [shutil.which(name) for name in ("chromium", "chromium-browser", "google-chrome")]
    return os.environ.get("TINYLM_TEST_CHROMIUM") or next(filter(None, installed), None)


@pytest.fixture
def browser(tmp_path):
    binary = chromium()
    if not binary:
        pytest.skip("no Chromium found (set TINYLM_TEST_CHROMIUM)")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    process = subprocess.Popen(
        [binary, "--headless=new", "--no-sandbox", f"--remote-debugging-port={port}",
         f"--user-data-dir={tmp_path / 'profile'}", "--no-proxy-server", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    endpoint = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                f"{endpoint}/json/version", timeout=1
            )
            break
        except OSError:
            time.sleep(0.1)
    yield endpoint
    process.terminate()
    process.wait(timeout=10)


def test_browser_renders_javascript_pages(games, site, browser):
    plain = HttpFetcher().fetch(f"{site}/js")
    assert b"Triss" not in plain.body.split(b"<script>")[0]
    page = CdpFetcher(browser, settle=0.2).fetch(f"{site}/js")
    assert "Triss Merigold" in page.body.decode()

    job = crawl(games, [f"{site}/js"], cdp_url=browser, use_browser=True, max_depth=0)
    assert job.indexed == 1, job.log
    assert titles(games) == ["Rendered"]


def test_browser_in_another_container(site, browser, monkeypatch):
    """A browser addressed by a host name, such as its container's: Chrome answers only an IP
    address or localhost, so the name is looked up."""
    from engine.web import fetch

    port = browser.rsplit(":", 1)[1]
    lookup = fetch.socket.getaddrinfo
    names = {"tinylm-chrome": "127.0.0.1"}
    monkeypatch.setattr(
        fetch.socket,
        "getaddrinfo",
        lambda host, *args, **kw: lookup(names.get(host, host), *args, **kw),
    )
    page = CdpFetcher(f"http://tinylm-chrome:{port}", settle=0.2).fetch(f"{site}/js")
    assert "Triss Merigold" in page.body.decode()
    monkeypatch.undo()
    with pytest.raises(fetch.FetchError, match="Cannot find the browser's host"):
        CdpFetcher("http://nowhere.invalid:9222").fetch(f"{site}/js")

    # A name with only an IPv6 address on the container network.
    v6 = [(fetch.socket.AF_INET6, fetch.socket.SOCK_STREAM, 6, "", ("fd00::5", 9222, 0, 0))]
    monkeypatch.setattr(fetch.socket, "getaddrinfo", lambda *args, **kw: v6)
    assert CdpFetcher("http://tinylm-chrome:9222")._address() == "http://[fd00::5]:9222"


def test_crawl_api(client, site):
    assert client.post("/api/collections/games/crawl", json={"urls": [site]}).status_code == 404
    client.post("/api/collections", json={"name": "games"})
    bad = client.post("/api/collections/games/crawl", json={"urls": ["ftp://x"]})
    assert bad.status_code == 422
    job = client.post(
        "/api/collections/games/crawl",
        json={"urls": [f"{site}/b"], "max_depth": 1, "per_minute": 120},
    ).json()
    for _ in range(100):
        status = client.get(f"/api/crawls/{job['id']}").json()
        if status["state"] != "running":
            break
        time.sleep(0.1)
    assert status["state"] == "done" and status["indexed"] == 2
    assert [j["id"] for j in client.get("/api/crawls").json()["data"]] == [job["id"]]


def test_repeated_crawls_download_only_new_pages(games, site):
    first = crawl(games, [f"{site}/"], max_pages=2, max_depth=3)
    assert first.fetched == 2 and first.known == 0
    # Home and A were downloaded; the next crawl walks through them and fetches B and C.
    second = crawl(games, [f"{site}/"], max_pages=2, max_depth=3)
    assert (second.fetched, second.known) == (2, 2)
    third = crawl(games, [f"{site}/"], max_pages=2, max_depth=3)
    assert (third.fetched, third.known) == (1, 4)
    assert titles(games) == ["Article A", "Article B", "Article C", "Article D"]
    pages = [p for p in Site.requests if p != "/robots.txt"]
    assert sorted(pages) == ["/", "/a", "/b", "/c", "/d"]

    # A deleted document is forgotten, so the next crawl downloads its page again.
    article_c = next(d for d in games.documents("games") if d["name"] == "Article C")
    games.delete_document("games", article_c["id"])
    fourth = crawl(games, [f"{site}/"], max_depth=3)
    assert fourth.fetched == 1 and fourth.indexed == 1
    assert Site.requests.count("/c") == 2


def wait_for(client, job_id: str) -> dict:
    for _ in range(100):
        status = client.get(f"/api/crawls/{job_id}").json()
        if status["state"] != "running":
            return status
        time.sleep(0.1)
    raise AssertionError("crawl did not finish")


def test_saved_websites_api(client, site):
    client.post("/api/collections", json={"name": "games"})
    body = {"urls": [f"{site}/"], "max_pages": 2, "max_depth": 3, "per_minute": 120}
    saved = client.post("/api/collections/games/websites", json=body)
    assert saved.status_code == 201
    website = saved.json()
    assert website["name"] == "127.0.0.1" and website["last_crawled_at"] is None

    assert website["only_under_start"] is False
    edited = client.put(
        f"/api/websites/{website['id']}",
        json={**body, "name": "Witcher fans", "only_under_start": True},
    )
    assert edited.json()["name"] == "Witcher fans" and edited.json()["max_pages"] == 2
    assert edited.json()["only_under_start"] is True
    edited = client.put(
        f"/api/websites/{website['id']}",
        json={**body, "exclude": [" /tag/ ", "", "*.pdf", "/tag/"]},
    )
    assert edited.json()["exclude"] == ["/tag/", "*.pdf"]
    assert client.get("/api/collections/games/websites").json()["data"][0]["exclude"] == [
        "/tag/",
        "*.pdf",
    ]
    client.put(f"/api/websites/{website['id']}", json={**body, "name": "Witcher fans"})
    listed = client.get("/api/collections/games/websites").json()["data"]
    assert [w["name"] for w in listed] == ["Witcher fans"]

    job = client.post(f"/api/websites/{website['id']}/crawl").json()
    assert job["website_id"] == website["id"]
    assert wait_for(client, job["id"])["fetched"] == 2
    again = wait_for(client, client.post(f"/api/websites/{website['id']}/crawl").json()["id"])
    assert (again["fetched"], again["known"]) == (2, 2)
    assert client.get("/api/collections/games/websites").json()["data"][0]["last_crawled_at"]

    assert client.delete(f"/api/websites/{website['id']}").status_code == 204
    assert client.post(f"/api/websites/{website['id']}/crawl").status_code == 404
    assert client.get("/api/collections/games/websites").json()["data"] == []


def test_crawl_settings_api(client, settings, site):
    current = client.get("/api/crawl/settings").json()
    assert current["respect_robots"] is True and current["per_minute"] == 20
    assert current["max_pages"] == 50 and current["user_agent"].startswith("tinylm-studio")

    changed = client.put(
        "/api/crawl/settings",
        json={"respect_robots": False, "max_pages": 7, "max_depth": 0, "per_minute": 120},
    ).json()
    assert changed["respect_robots"] is False and changed["max_pages"] == 7
    saved = json.loads((settings.workspace_dir / "crawl-settings.json").read_text())
    assert saved == {"respect_robots": False, "per_minute": 120, "max_pages": 7, "max_depth": 0}
    bad = client.put("/api/crawl/settings", json={"cdp_url": "localhost:9222"})
    assert bad.status_code == 422 and "cdp_url" in bad.json()["detail"]
    assert client.put("/api/crawl/settings", json={"color": "red"}).status_code == 422

    # New websites and crawls take the options they leave out from the settings, and robots.txt
    # is not read.
    client.post("/api/collections", json={"name": "games"})
    website = client.post("/api/collections/games/websites", json={"urls": [site]}).json()
    assert (website["max_pages"], website["max_depth"], website["per_minute"]) == (7, 0, 120)
    job = client.post("/api/collections/games/crawl", json={"urls": [f"{site}/private/x"]})
    assert wait_for(client, job.json()["id"])["indexed"] == 1
    assert "/robots.txt" not in Site.requests

    # Browser (CDP) in the settings: pages added without use_browser load through it.
    client.put("/api/crawl/settings", json={"use_browser": True, "cdp_url": "http://127.0.0.1:9"})
    job = client.post("/api/collections/games/crawl", json={"urls": [f"{site}/b"]}).json()
    assert job["options"]["use_browser"] is True
    done = wait_for(client, job["id"])
    assert "loading pages in the browser at http://127.0.0.1:9" in done["log"]
    assert done["failed"] == 1 and "Cannot reach the browser" in done["log"][-1]


def test_outcome_of_each_start_address(games, site):
    crawl(games, [f"{site}/a"], max_depth=0)
    urls = [f"{site}/a#comments", f"{site}/b", f"{site}/", f"{site}/missing", f"{site}/private/x"]
    job = crawl(games, urls, max_depth=0)
    assert job.outcomes == {
        f"{site}/a": "known",
        f"{site}/b": "added",
        f"{site}/": "skipped",  # too little text
        f"{site}/missing": "failed",
        f"{site}/private/x": "skipped",  # robots.txt
    }
    assert (job.known, job.indexed, job.skipped, job.failed) == (1, 1, 2, 1)
    # Next time the short home page is still reported as skipped, not as already there.
    assert crawl(games, [f"{site}/"], max_depth=0).outcomes == {f"{site}/": "skipped"}


def test_older_database_gets_the_new_website_column(tmp_path):
    import sqlite3

    from engine.rag.embeddings import HashEmbedder
    from engine.rag.store import KnowledgeStore
    from engine.web.sites import WebsiteStore

    path = tmp_path / "knowledge.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE websites (id INTEGER PRIMARY KEY, collection TEXT NOT NULL, name TEXT NOT"
        " NULL, urls TEXT NOT NULL, max_pages INTEGER NOT NULL, max_depth INTEGER NOT NULL,"
        " per_minute REAL NOT NULL, use_browser INTEGER NOT NULL, respect_robots INTEGER NOT"
        " NULL, created_at TEXT NOT NULL, last_crawled_at TEXT);"
        "INSERT INTO websites VALUES (1, 'games', 'old', '[\"https://a.example/\"]', 5, 1, 20,"
        " 0, 1, '2026-01-01', NULL);"
    )
    old.commit()
    old.close()
    store = KnowledgeStore(path, HashEmbedder())
    old_website = WebsiteStore(store).website(1)
    assert old_website.only_under_start is False and old_website.exclude == []
