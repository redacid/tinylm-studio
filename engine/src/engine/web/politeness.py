"""Keeps the crawler polite: a per-site request rate limit and robots.txt rules."""

import random
import threading
import time
from collections.abc import Callable
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser


def site(url: str) -> str:
    """The site a URL belongs to, for rate limits and same-site checks (www. is ignored)."""
    host = (urlsplit(url).hostname or "").lower()
    return host.removeprefix("www.")


class RateLimiter:
    """Spaces requests to each site at least 60 / per_minute seconds apart (plus up to 20%
    random jitter); a site can push its own slot further with Crawl-delay or Retry-After."""

    def __init__(
        self,
        per_minute: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], object] = time.sleep,
    ):
        self.interval = 60.0 / per_minute
        self.clock = clock
        self.sleep = sleep
        self.next_at: dict[str, float] = {}
        self.delays: dict[str, float] = {}
        self._lock = threading.Lock()

    def set_min_delay(self, host: str, seconds: float) -> None:
        self.delays[host] = seconds

    def wait(self, host: str) -> None:
        with self._lock:
            now = self.clock()
            start = max(now, self.next_at.get(host, now))
            interval = max(self.interval, self.delays.get(host, 0.0))
            self.next_at[host] = start + interval * random.uniform(1.0, 1.2)
        if start > now:
            self.sleep(start - now)

    def back_off(self, host: str, seconds: float) -> None:
        with self._lock:
            self.next_at[host] = max(self.next_at.get(host, 0.0), self.clock() + seconds)


class Robots:
    """robots.txt per site, fetched once. A missing or unreadable file allows everything."""

    def __init__(self, fetch_text: Callable[[str], str | None], user_agent: str):
        self.fetch_text = fetch_text
        self.user_agent = user_agent
        self.parsers: dict[str, RobotFileParser] = {}

    def _parser(self, url: str) -> RobotFileParser:
        parts = urlsplit(url)
        key = f"{parts.scheme}://{parts.netloc}"
        if key not in self.parsers:
            parser = RobotFileParser()
            parser.parse((self.fetch_text(f"{key}/robots.txt") or "").splitlines())
            self.parsers[key] = parser
        return self.parsers[key]

    def allowed(self, url: str) -> bool:
        return self._parser(url).can_fetch(self.user_agent, url)

    def crawl_delay(self, url: str) -> float:
        delay = self._parser(url).crawl_delay(self.user_agent)
        return float(delay) if delay else 0.0
