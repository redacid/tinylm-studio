"""Page fetchers: plain HTTP (optionally through an HTTP or SOCKS proxy) and a real browser
controlled over the Chrome DevTools Protocol (CDP), for sites that need JavaScript."""

import itertools
import json
import time
from dataclasses import dataclass

import httpx

DEFAULT_USER_AGENT = "tinylm-studio/0.1 (local research crawler)"


class FetchError(Exception):
    pass


@dataclass
class Page:
    url: str  # after redirects
    status: int
    content_type: str
    body: bytes
    # Seconds the site asked us to wait (Retry-After on 429/503), if any.
    retry_after: float | None = None

    @property
    def is_html(self) -> bool:
        return "html" in self.content_type


def retry_after_seconds(value: str | None) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:  # an HTTP date; rare, treat as "a while"
        return 120.0


class HttpFetcher:
    """Plain HTTP client. proxy: http://user:pass@host:port, socks5://host:port or socks5h://..."""

    def __init__(self, proxy: str | None = None, user_agent: str = DEFAULT_USER_AGENT, timeout=30):
        self.client = httpx.Client(
            proxy=proxy,
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": user_agent, "Accept-Language": "uk,ru;q=0.9,en;q=0.8"},
        )
        self.user_agent = user_agent

    def fetch(self, url: str) -> Page:
        try:
            response = self.client.get(url)
        except httpx.HTTPError as e:
            raise FetchError(f"{type(e).__name__}: {e}") from None
        return Page(
            url=str(response.url),
            status=response.status_code,
            content_type=response.headers.get("content-type", "").lower(),
            body=response.content,
            retry_after=retry_after_seconds(response.headers.get("retry-after")),
        )

    def close(self) -> None:
        self.client.close()


class CdpFetcher:
    """Opens each page in a new tab of an already running Chrome/Chromium, waits for it to load
    and render, and returns the resulting HTML. Start the browser with, for example:

        chromium --remote-debugging-port=9222 --user-data-dir=/tmp/tinylm-browser

    The browser uses its own network settings; to go through a proxy, start it with
    --proxy-server=socks5://host:port.
    """

    def __init__(self, endpoint: str = "http://127.0.0.1:9222", timeout=30, settle=1.5):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.settle = settle  # extra time for scripts to render content after the load event
        # The DevTools endpoint is local: never route it through a system proxy.
        self.http = httpx.Client(timeout=10, trust_env=False)

    def _new_tab(self) -> dict:
        try:
            # Chrome 111+ requires PUT; older versions only accept GET.
            response = self.http.put(f"{self.endpoint}/json/new?about:blank")
            if response.status_code == 405:
                response = self.http.get(f"{self.endpoint}/json/new?about:blank")
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as e:
            raise FetchError(f"Cannot reach the browser at {self.endpoint}: {e}") from None

    def fetch(self, url: str) -> Page:
        from websockets.sync.client import connect

        tab = self._new_tab()
        try:
            with connect(tab["webSocketDebuggerUrl"], max_size=None, open_timeout=10) as ws:
                session = CdpSession(ws, self.timeout)
                session.call("Page.enable")
                navigation = session.call("Page.navigate", url=url)
                if navigation.get("errorText"):
                    raise FetchError(f"{navigation['errorText']} ({url})")
                session.wait_for("Page.loadEventFired")
                time.sleep(self.settle)
                html = session.evaluate("document.documentElement.outerHTML")
                final_url = session.evaluate("location.href")
        except (OSError, TimeoutError) as e:
            raise FetchError(f"Browser error: {e}") from None
        finally:
            try:
                self.http.get(f"{self.endpoint}/json/close/{tab['id']}")
            except httpx.HTTPError:
                pass
        return Page(final_url, 200, "text/html", html.encode())

    def close(self) -> None:
        self.http.close()


class CdpSession:
    """Minimal request/response and event handling over one DevTools websocket."""

    def __init__(self, ws, timeout: float):
        self.ws = ws
        self.timeout = timeout
        self.ids = itertools.count(1)
        self.events: list[str] = []

    def _receive(self, deadline: float) -> dict:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("the page took too long to load")
        return json.loads(self.ws.recv(timeout=remaining))

    def call(self, method: str, **params) -> dict:
        message_id = next(self.ids)
        self.ws.send(json.dumps({"id": message_id, "method": method, "params": params}))
        deadline = time.monotonic() + self.timeout
        while True:
            message = self._receive(deadline)
            if message.get("id") == message_id:
                if "error" in message:
                    raise FetchError(f"{method}: {message['error'].get('message')}")
                return message.get("result", {})
            if "method" in message:
                self.events.append(message["method"])

    def wait_for(self, event: str) -> None:
        deadline = time.monotonic() + self.timeout
        while event not in self.events:
            message = self._receive(deadline)
            if "method" in message:
                self.events.append(message["method"])

    def evaluate(self, expression: str):
        result = self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        return result["result"].get("value")
