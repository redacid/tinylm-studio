"""Crawls websites into a collection: follows links within each start site, politely.

A site is one host (www. aside): links to other sites and to other subdomains are never
followed, nor links matching the crawl's exclude patterns. With only_under_start, links are
also kept inside the folders of the start addresses, like wget --no-parent: from
https://example.com/guides/ the crawl stays under /guides/.

Pages downloaded by earlier crawls into the same collection are not downloaded again: the crawl
walks through them using their remembered links and spends its page budget only on new pages,
so repeated crawls of a site gradually cover all of it.
"""

import logging
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Literal
from urllib.parse import urldefrag, urljoin, urlsplit

from engine.rag.convert import UnsupportedFile, convert
from engine.rag.store import KnowledgeStore
from engine.web.crawl_settings import CrawlSettings
from engine.web.fetch import DEFAULT_USER_AGENT, CdpFetcher, FetchError, HttpFetcher, Page
from engine.web.politeness import RateLimiter, Robots, site
from engine.web.sites import WebsiteStore

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


Outcome = Literal["added", "duplicate", "known", "skipped", "failed"]
OUTCOME_COUNTERS = {
    "added": "indexed",
    "duplicate": "duplicates",
    "known": "known",
    "skipped": "skipped",
    "failed": "failed",
}


@dataclass
class CrawlOptions:
    urls: list[str]
    max_pages: int = 50  # new pages downloaded, across all start sites
    max_depth: int = 2  # link hops from a start URL
    per_minute: float = 20  # requests per minute to each site
    use_browser: bool = False  # fetch through the browser at cdp_url instead of plain HTTP
    # Read robots.txt; it is not read either way when the crawler settings turn it off.
    respect_robots: bool = True
    # Follow only links under the start addresses' folders, not the whole site.
    only_under_start: bool = False
    # Links never followed: a part of the address ("/tag/", "?sort="), or a pattern with * that
    # covers the whole address ("*/forum/*", "*.pdf"); see excluded().
    exclude: list[str] = field(default_factory=list)


@dataclass
class CrawlJob:
    collection: str
    options: CrawlOptions
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: Literal["running", "done", "cancelled", "error"] = "running"
    website_id: int | None = None  # the saved website this crawl runs, if any
    fetched: int = 0  # new pages downloaded
    known: int = 0  # pages downloaded by earlier crawls, walked through without a request
    indexed: int = 0
    duplicates: int = 0
    skipped: int = 0
    failed: int = 0
    current: str | None = None
    log: list[str] = field(default_factory=list)  # the last 50 events, newest last
    # What happened to each start address (without #fragment): added, duplicate, known (downloaded
    # by an earlier crawl), skipped (robots.txt, too little text) or failed. Unlisted: not reached.
    outcomes: dict[str, Outcome] = field(default_factory=dict)
    notes: int = 0  # events so far, including those dropped from log
    error: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    finished_at: str | None = None

    def __post_init__(self):
        self.cancelled = threading.Event()
        self.starts = {urldefrag(u)[0] for u in self.options.urls}

    def record(self, url: str, outcome: Outcome, counter: str | None = None) -> None:
        """Counts what happened to a page (in counter, by default the outcome's own); for start
        addresses, also remembers it per address."""
        counter = counter or OUTCOME_COUNTERS[outcome]
        setattr(self, counter, getattr(self, counter) + 1)
        if url in self.starts:
            self.outcomes[url] = outcome

    def note(self, message: str) -> None:
        log.info("Crawl %s (%s): %s", self.id, self.collection, message)
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


def folder(url: str) -> str:
    """The folder a start address is in: its path up to the last slash."""
    path = urlsplit(url).path or "/"
    return path[: path.rfind("/") + 1]


def under(url: str, folders: dict[str, set[str]]) -> bool:
    """Whether a URL is in one of its site's start folders."""
    path = urlsplit(url).path or "/"
    return any(path.startswith(f) for f in folders.get(site(url), ()))


def excluded(url: str, patterns: list[str]) -> bool:
    """Whether a link matches one of the exclude patterns; letter case is ignored. Only * is
    special (any characters); "?" and the rest match themselves."""
    url = url.lower()
    for pattern in patterns:
        pattern = pattern.lower()
        if "*" not in pattern:
            if pattern in url:
                return True
        elif re.fullmatch(".*".join(map(re.escape, pattern.split("*"))), url):
            return True
    return False


def worth_fetching(url: str) -> bool:
    return PurePosixPath(urlsplit(url).path).suffix.lower() not in SKIP_SUFFIXES


DEFAULT_SETTINGS = CrawlSettings(user_agent=DEFAULT_USER_AGENT, cdp_url="http://127.0.0.1:9222")


class Crawler:
    def __init__(
        self,
        store: KnowledgeStore,
        proxy: str | None = None,
        settings: Callable[[], CrawlSettings] = lambda: DEFAULT_SETTINGS,
    ):
        self.store = store
        self.proxy = proxy
        self.settings = settings  # read at the start of each crawl, so changes apply to the next
        self.pages = WebsiteStore(store)

    def run(self, job: CrawlJob) -> None:
        options = job.options
        current = self.settings()
        http = HttpFetcher(self.proxy, current.user_agent)
        browser = CdpFetcher(current.cdp_url) if options.use_browser else None
        if browser:
            job.note(f"loading pages in the browser at {current.cdp_url}")
        respect_robots = options.respect_robots and current.respect_robots
        if options.respect_robots and not respect_robots:
            job.note("robots.txt is ignored (Settings > Crawler)")
        limiter = RateLimiter(options.per_minute, sleep=job.cancelled.wait)
        if options.exclude:
            job.note(f"links matching {', '.join(options.exclude)} are skipped")

        def robots_text(url: str) -> str | None:
            limiter.wait(site(url))
            try:
                page = http.fetch(url)
            except FetchError:
                return None
            return page.body.decode(errors="replace") if page.status == 200 else None

        robots = Robots(robots_text, current.user_agent)
        start_sites = {site(u) for u in options.urls}
        folders: dict[str, set[str]] = {}
        for u in options.urls:
            folders.setdefault(site(u), set()).add(folder(u))
        queue = deque((normalize(u), 0) for u in options.urls)
        seen = {url for url, _ in queue}
        attempts: dict[str, int] = {}

        def follow(found: list[str], depth: int) -> None:
            if depth >= options.max_depth:
                return
            for link in found:
                if (
                    link not in seen
                    and site(link) in start_sites
                    and (not options.only_under_start or under(link, folders))
                    and worth_fetching(link)
                    and not excluded(link, options.exclude)
                ):
                    seen.add(link)
                    queue.append((link, depth + 1))

        try:
            while queue and job.fetched < options.max_pages and not job.cancelled.is_set():
                url, depth = queue.popleft()
                known = self.pages.known_page(job.collection, url)
                if known is not None:
                    found, indexed = known
                    # Pages with too little text were remembered too, only for their links.
                    job.record(url, "known" if indexed else "skipped", counter="known")
                    follow(found, depth)
                    continue
                host = site(url)
                if respect_robots:
                    if not robots.allowed(url):
                        job.record(url, "skipped")
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
                    job.record(url, "failed")
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
                    job.record(url, "failed")
                    job.note(f"HTTP {page.status} {url}")
                    continue
                document_id = self._index(job, url, page)
                found = [link for link in links(page) if site(link) == host] if page.is_html else []
                self.pages.remember_page(job.collection, url, found, document_id)
                follow(found, depth)
            job.state = "cancelled" if job.cancelled.is_set() else "done"
        except Exception as e:  # keep the job record useful instead of losing the thread
            log.exception("Crawl %s failed", job.id)
            job.state, job.error = "error", str(e)
        finally:
            job.current = None
            log.info(
                "Crawl %s %s: %d downloaded, %d indexed, %d duplicates, %d skipped, %d failed",
                job.id,
                job.state,
                job.fetched,
                job.indexed,
                job.duplicates,
                job.skipped,
                job.failed,
            )
            if job.website_id is not None:
                self.pages.mark_crawled(job.website_id)
            job.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
            http.close()
            if browser:
                browser.close()

    def _index(self, job: CrawlJob, url: str, page: Page) -> int | None:
        """Adds the page to the collection; returns its document id, or None if not indexed."""
        if page.is_html:
            filename = "page.html"
        elif "pdf" in page.content_type:
            filename = "document.pdf"
        else:
            job.record(url, "skipped")
            return None
        try:
            doc = convert(filename, page.body)
        except (UnsupportedFile, ValueError) as e:
            job.record(url, "failed")
            job.note(f"cannot read {page.url}: {e}")
            return None
        if len(doc.text) < MIN_TEXT_CHARS:
            job.record(url, "skipped")
            return None
        result = self.store.add_document(job.collection, doc.title or page.url, doc.text, page.url)
        if result["duplicate"]:
            job.record(url, "duplicate")
        else:
            job.record(url, "added")
            job.note(f"added {doc.title or page.url} ({result['chunks']} chunks)")
        return result["id"]


class CrawlManager:
    """Runs crawl jobs in background threads and keeps their status for the UI."""

    def __init__(self, crawler: Crawler):
        self.crawler = crawler
        self.jobs: dict[str, CrawlJob] = {}

    def start(
        self, collection: str, options: CrawlOptions, website_id: int | None = None
    ) -> CrawlJob:
        job = CrawlJob(collection, options, website_id=website_id)
        self.jobs[job.id] = job
        log.info(
            "Crawl %s into %s: %s, up to %d new pages, depth %d",
            job.id,
            collection,
            " ".join(options.urls),
            options.max_pages,
            options.max_depth,
        )
        threading.Thread(target=self.crawler.run, args=(job,), daemon=True).start()
        return job

    def wait(self, job: CrawlJob, poll: float = 0.1) -> None:
        while job.state == "running":
            time.sleep(poll)
