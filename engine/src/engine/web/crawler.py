"""Crawls websites into a collection: follows links within each start site, politely."""

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Literal
from urllib.parse import urldefrag, urljoin, urlsplit

from engine.rag.convert import UnsupportedFile, convert
from engine.rag.store import KnowledgeStore
from engine.web.fetch import DEFAULT_USER_AGENT, CdpFetcher, FetchError, HttpFetcher, Page
from engine.web.politeness import RateLimiter, Robots, site

log = logging.getLogger(__name__)

# Links to these are never fetched: they hold no text worth indexing.
SKIP_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".bmp", ".avif",
    ".mp4", ".webm", ".mp3", ".ogg", ".wav", ".avi", ".mov",
    ".zip", ".rar", ".7z", ".gz", ".tar", ".exe", ".dmg", ".apk", ".iso",
    ".css", ".js", ".json", ".xml", ".rss", ".woff", ".woff2", ".ttf",
}  # fmt: skip
# Shorter pages are usually listings or navigation: followed for links but not indexed.
MIN_TEXT_CHARS = 300


@dataclass
class CrawlOptions:
    urls: list[str]
    max_pages: int = 50  # pages fetched, across all start sites
    max_depth: int = 2  # link hops from a start URL
    per_minute: float = 20  # requests per minute to each site
    use_browser: bool = False  # fetch through the browser at cdp_url instead of plain HTTP
    respect_robots: bool = True


@dataclass
class CrawlJob:
    collection: str
    options: CrawlOptions
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: Literal["running", "done", "cancelled", "error"] = "running"
    fetched: int = 0
    indexed: int = 0
    duplicates: int = 0
    skipped: int = 0
    failed: int = 0
    current: str | None = None
    log: list[str] = field(default_factory=list)  # the last 50 events, newest last
    notes: int = 0  # events so far, including those dropped from log
    error: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    finished_at: str | None = None

    def __post_init__(self):
        self.cancelled = threading.Event()

    def note(self, message: str) -> None:
        self.log = [*self.log[-49:], message]
        self.notes += 1

    def to_dict(self) -> dict:
        return asdict(self)


def normalize(url: str) -> str:
    return urldefrag(url)[0]


def links(page: Page) -> list[str]:
    import lxml.html

    try:
        tree = lxml.html.fromstring(page.body)
    except (ValueError, lxml.etree.ParserError):
        return []
    found = []
    for href in tree.xpath("//a/@href"):
        url = normalize(urljoin(page.url, href.strip()))
        if urlsplit(url).scheme in ("http", "https"):
            found.append(url)
    return found


def worth_fetching(url: str) -> bool:
    return PurePosixPath(urlsplit(url).path).suffix.lower() not in SKIP_SUFFIXES


class Crawler:
    def __init__(
        self,
        store: KnowledgeStore,
        proxy: str | None = None,
        cdp_url: str = "http://127.0.0.1:9222",
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        self.store = store
        self.proxy = proxy
        self.cdp_url = cdp_url
        self.user_agent = user_agent

    def run(self, job: CrawlJob) -> None:
        options = job.options
        http = HttpFetcher(self.proxy, self.user_agent)
        browser = CdpFetcher(self.cdp_url) if options.use_browser else None
        limiter = RateLimiter(options.per_minute, sleep=job.cancelled.wait)

        def robots_text(url: str) -> str | None:
            limiter.wait(site(url))
            try:
                page = http.fetch(url)
            except FetchError:
                return None
            return page.body.decode(errors="replace") if page.status == 200 else None

        robots = Robots(robots_text, self.user_agent)
        start_sites = {site(u) for u in options.urls}
        queue = deque((normalize(u), 0) for u in options.urls)
        seen = {url for url, _ in queue}
        attempts: dict[str, int] = {}
        try:
            while queue and job.fetched < options.max_pages and not job.cancelled.is_set():
                url, depth = queue.popleft()
                host = site(url)
                if options.respect_robots:
                    if not robots.allowed(url):
                        job.skipped += 1
                        job.note(f"robots.txt disallows {url}")
                        continue
                    limiter.set_min_delay(host, robots.crawl_delay(url))
                limiter.wait(host)
                if job.cancelled.is_set():
                    break
                job.current = url
                try:
                    page = (browser or http).fetch(url)
                except FetchError as e:
                    job.failed += 1
                    job.note(f"failed {url}: {e}")
                    continue
                if page.status in (429, 503) and attempts.get(url, 0) < 2:
                    # The site asks us to slow down: pause it and try this page again later.
                    attempts[url] = attempts.get(url, 0) + 1
                    pause = 60.0 if page.retry_after is None else page.retry_after
                    limiter.back_off(host, pause)
                    queue.append((url, depth))
                    job.note(
                        f"{host} asked to slow down (HTTP {page.status}); pausing {pause:.0f} s"
                    )
                    continue
                job.fetched += 1
                if page.status >= 400:
                    job.failed += 1
                    job.note(f"HTTP {page.status} {url}")
                    continue
                self._index(job, page)
                if page.is_html and depth < options.max_depth:
                    for link in links(page):
                        if link not in seen and site(link) in start_sites and worth_fetching(link):
                            seen.add(link)
                            queue.append((link, depth + 1))
            job.state = "cancelled" if job.cancelled.is_set() else "done"
        except Exception as e:  # keep the job record useful instead of losing the thread
            log.exception("Crawl %s failed", job.id)
            job.state, job.error = "error", str(e)
        finally:
            job.current = None
            job.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
            http.close()
            if browser:
                browser.close()

    def _index(self, job: CrawlJob, page: Page) -> None:
        if page.is_html:
            filename = "page.html"
        elif "pdf" in page.content_type:
            filename = "document.pdf"
        else:
            job.skipped += 1
            return
        try:
            doc = convert(filename, page.body)
        except (UnsupportedFile, ValueError) as e:
            job.failed += 1
            job.note(f"cannot read {page.url}: {e}")
            return
        if len(doc.text) < MIN_TEXT_CHARS:
            job.skipped += 1
            return
        result = self.store.add_document(job.collection, doc.title or page.url, doc.text, page.url)
        if result["duplicate"]:
            job.duplicates += 1
        else:
            job.indexed += 1
            job.note(f"added {doc.title or page.url} ({result['chunks']} chunks)")


class CrawlManager:
    """Runs crawl jobs in background threads and keeps their status for the UI."""

    def __init__(self, crawler: Crawler):
        self.crawler = crawler
        self.jobs: dict[str, CrawlJob] = {}

    def start(self, collection: str, options: CrawlOptions) -> CrawlJob:
        job = CrawlJob(collection, options)
        self.jobs[job.id] = job
        threading.Thread(target=self.crawler.run, args=(job,), daemon=True).start()
        return job

    def wait(self, job: CrawlJob, poll: float = 0.1) -> None:
        while job.state == "running":
            time.sleep(poll)
