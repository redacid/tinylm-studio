"""Teacher models: larger models that write training examples for a small one.

Three kinds:
- local: a model installed in the studio (e.g. gemma-3-4b-gguf). Free and private, but slow on a
  CPU.
- openai: any OpenAI-compatible chat completions API: OpenRouter (including its free models),
  OpenAI, Groq, a local llama.cpp server, Ollama or LM Studio.
- anthropic: the Anthropic Messages API (Claude).
"""

import json
import logging
import time
from collections.abc import Iterator
from typing import Literal, Protocol

import httpx

from engine.chat import ChatRequest, Finish, Message, StreamEvent, TextDelta, Usage, collect

log = logging.getLogger(__name__)

TeacherKind = Literal["local", "openai", "anthropic"]
DEFAULT_BASE_URLS = {
    "openai": "https://openrouter.ai/api/v1",
    "anthropic": "https://api.anthropic.com",
}
ANTHROPIC_VERSION = "2023-06-01"
# OpenRouter shows these on its activity page; other providers ignore them.
APP_HEADERS = {"HTTP-Referer": "https://github.com/tinylm-studio", "X-Title": "tinylm-studio"}


class TeacherError(Exception):
    pass


class SamplingRefused(Exception):
    """The model refused a sampling parameter before answering anything."""


class RateLimited(TeacherError):
    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


class Teacher(Protocol):
    name: str
    # Who answered the last request, or failed it: {"model": ..., "provider": ...}. OpenRouter
    # names the model that actually answered (it can differ from the one asked for) and the
    # upstream provider, also in its errors; until an answer says otherwise it is the model asked.
    last_served: dict | None

    def complete(
        self, system: str, prompt: str, max_tokens: int = 2048, temperature: float = 0.7
    ) -> str: ...


def retry_after(response: httpx.Response, default: float = 30) -> float:
    try:
        return min(600.0, float(response.headers.get("retry-after", default)))
    except ValueError:
        return default


def served_by(response: httpx.Response, model: str, provider: str | None = None) -> dict:
    """The model and provider an answer or an error names, falling back to the ones asked."""
    try:
        data = response.json()
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    error = data.get("error") if isinstance(data.get("error"), dict) else {}
    metadata = error.get("metadata") if isinstance(error.get("metadata"), dict) else {}
    return {
        "model": data.get("model") or model,
        "provider": data.get("provider") or metadata.get("provider_name") or provider,
    }


def check(response: httpx.Response, provider: str) -> None:
    if response.status_code == 429:
        raise RateLimited(f"{provider}: rate limit reached", retry_after(response))
    if response.status_code in (401, 403):
        raise TeacherError(f"{provider} rejected the API key (HTTP {response.status_code})")
    if response.status_code >= 400:
        try:
            body = response.json()
            error = body.get("error")
            detail = error.get("message") if isinstance(error, dict) else error or body
        except ValueError:
            detail = response.text[:300]
        raise TeacherError(f"{provider} answered HTTP {response.status_code}: {detail}")


def sse_data(response: httpx.Response) -> Iterator[dict]:
    """The JSON of each "data:" line of a server-sent event stream, until [DONE]."""
    for line in response.iter_lines():
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except ValueError:
            continue


def check_stream(response: httpx.Response, provider: str) -> None:
    """check() for a streamed response, whose error body has to be read first."""
    if response.status_code >= 400:
        response.read()
        check(response, provider)


class LocalTeacher:
    """A model installed in the studio, run in this process."""

    def __init__(self, models, model_id: str):
        self.models = models
        self.model_id = model_id
        self.name = f"local:{model_id}"
        self.last_served = {"model": model_id, "provider": "local"}

    def complete(
        self, system: str, prompt: str, max_tokens: int = 2048, temperature: float = 0.7
    ) -> str:
        runtime = self.models.runtime(self.model_id)
        messages = [Message("system", system), Message("user", prompt)]
        request = ChatRequest(self.model_id, messages, max_tokens, temperature)
        return collect(runtime.stream(request)).text


class OpenAICompatibleTeacher:
    def __init__(
        self, base_url: str, api_key: str | None, model: str, name: str = "", timeout: float = 300
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name or model
        self.last_served: dict | None = None
        headers = {**APP_HEADERS}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(timeout=timeout, headers=headers)

    def complete(
        self, system: str, prompt: str, max_tokens: int = 2048, temperature: float = 0.7
    ) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        self.last_served = {"model": self.model, "provider": None}
        try:
            response = self.client.post(f"{self.base_url}/chat/completions", json=body)
        except httpx.HTTPError as e:
            raise TeacherError(f"{self.base_url} is not reachable ({type(e).__name__})") from None
        self.last_served = served_by(response, self.model)
        check(response, self.name)
        try:
            data = response.json()
            if data.get("error"):
                # OpenRouter reports some upstream failures with HTTP 200.
                error = data["error"]
                message = error.get("message") if isinstance(error, dict) else error
                raise TeacherError(f"{self.name}: {message}")
            return data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError):
            raise TeacherError(f"{self.name}: unexpected answer {response.text[:300]}") from None

    def stream_chat(self, request: ChatRequest) -> Iterator[StreamEvent]:
        """Answers a whole conversation as it is written, for the chat."""
        body = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": True,
        }
        if request.stop:
            body["stop"] = request.stop
        reason, usage, pieces = "stop", Usage(), 0
        try:
            with self.client.stream(
                "POST", f"{self.base_url}/chat/completions", json=body
            ) as response:
                check_stream(response, self.name)
                for data in sse_data(response):
                    if data.get("error"):
                        error = data["error"]
                        message = error.get("message") if isinstance(error, dict) else error
                        raise TeacherError(f"{self.name}: {message}")
                    if counted := data.get("usage"):  # OpenRouter sends it in the last chunk
                        usage.input_tokens = counted.get("prompt_tokens") or 0
                        usage.output_tokens = counted.get("completion_tokens") or 0
                    for choice in data.get("choices") or []:
                        if text := (choice.get("delta") or {}).get("content"):
                            pieces += 1
                            yield TextDelta(text)
                        if choice.get("finish_reason") == "length":
                            reason = "length"
        except httpx.HTTPError as e:
            raise TeacherError(f"{self.base_url} is not reachable ({type(e).__name__})") from None
        # Without a count from the provider, each streamed piece is about one token.
        usage.output_tokens = usage.output_tokens or pieces
        yield Finish(reason, usage)

    def models(self) -> list[dict]:
        try:
            response = self.client.get(f"{self.base_url}/models")
        except httpx.HTTPError as e:
            raise TeacherError(f"{self.base_url} is not reachable ({type(e).__name__})") from None
        check(response, self.base_url)
        models = []
        for item in response.json().get("data", []):
            pricing = item.get("pricing") or {}
            free = item["id"].endswith(":free") or (
                pricing.get("prompt") in ("0", 0) and pricing.get("completion") in ("0", 0)
            )
            models.append({"id": item["id"], "name": item.get("name") or item["id"], "free": free})
        return sorted(models, key=lambda m: (not m["free"], m["id"]))


# Anthropic models that refuse sampling parameters: newer Claude models answer HTTP 400 to
# temperature (Opus 4.7 and later reject it outright, Haiku 5.5 and Sonnet 5.5 any value but the
# default). Learned from the first refusal, by base URL and model, so later requests skip it.
NO_SAMPLING: set[tuple[str, str]] = set()


def rejects_sampling(response: httpx.Response) -> bool:
    if response.status_code != 400:
        return False
    text = response.text.lower()
    return any(p in text for p in ("temperature", "top_p", "top_k"))


class AnthropicTeacher:
    def __init__(
        self, base_url: str, api_key: str | None, model: str, name: str = "", timeout: float = 300
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name or model
        self.last_served: dict | None = None
        headers = {"anthropic-version": ANTHROPIC_VERSION}
        if api_key:
            headers["x-api-key"] = api_key
        self.client = httpx.Client(timeout=timeout, headers=headers)

    def complete(
        self, system: str, prompt: str, max_tokens: int = 2048, temperature: float = 0.7
    ) -> str:
        body = {
            "model": self.model,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        self.last_served = {"model": self.model, "provider": "Anthropic"}
        try:
            response = self.client.post(
                f"{self.base_url}/v1/messages", json=self._sampled(body, temperature)
            )
            if self._refused_sampling(response):
                response = self.client.post(f"{self.base_url}/v1/messages", json=body)
        except httpx.HTTPError as e:
            raise TeacherError(f"{self.base_url} is not reachable ({type(e).__name__})") from None
        self.last_served = served_by(response, self.model, "Anthropic")
        check(response, self.name)
        try:
            blocks = response.json()["content"]
            return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        except (ValueError, KeyError, TypeError):
            raise TeacherError(f"{self.name}: unexpected answer {response.text[:300]}") from None

    def stream_chat(self, request: ChatRequest) -> Iterator[StreamEvent]:
        """Answers a whole conversation as it is written, for the chat."""
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        body = {
            "model": self.model,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in request.messages
                if m.role != "system"
            ],
            "max_tokens": request.max_tokens or 1024,
            "stream": True,
        }
        if system:
            body["system"] = system
        if request.stop:
            body["stop_sequences"] = request.stop
        try:
            yield from self._stream(self._sampled(body, request.temperature))
        except SamplingRefused:
            yield from self._stream(body)

    def _sampled(self, body: dict, temperature: float | None) -> dict:
        if temperature is None or (self.base_url, self.model) in NO_SAMPLING:
            return body
        return {**body, "temperature": temperature}

    def _refused_sampling(self, response: httpx.Response) -> bool:
        """True when the model refused the temperature sent: the request is to be sent without."""
        if "temperature" not in json.loads(response.request.content) or not rejects_sampling(
            response
        ):
            return False
        log.info("%s does not take a temperature; asking without", self.model)
        NO_SAMPLING.add((self.base_url, self.model))
        return True

    def _stream(self, body: dict) -> Iterator[StreamEvent]:
        reasons = {"max_tokens": "length", "stop_sequence": "stop_sequence"}
        reason, stop_sequence, usage = "stop", None, Usage()
        try:
            with self.client.stream("POST", f"{self.base_url}/v1/messages", json=body) as response:
                if response.status_code == 400:
                    response.read()
                    if self._refused_sampling(response):
                        raise SamplingRefused
                check_stream(response, self.name)
                for data in sse_data(response):
                    kind = data.get("type")
                    if kind == "error":
                        raise TeacherError(f"{self.name}: {data.get('error', {}).get('message')}")
                    if kind == "message_start":
                        usage.input_tokens = data["message"].get("usage", {}).get("input_tokens", 0)
                    elif kind == "content_block_delta":
                        if text := data.get("delta", {}).get("text"):
                            yield TextDelta(text)
                    elif kind == "message_delta":
                        delta = data.get("delta", {})
                        reason = reasons.get(delta.get("stop_reason"), "stop")
                        stop_sequence = delta.get("stop_sequence")
                        usage.output_tokens = data.get("usage", {}).get("output_tokens", 0)
        except httpx.HTTPError as e:
            raise TeacherError(f"{self.base_url} is not reachable ({type(e).__name__})") from None
        yield Finish(reason, usage, stop_sequence)

    def models(self) -> list[dict]:
        try:
            response = self.client.get(f"{self.base_url}/v1/models", params={"limit": 100})
        except httpx.HTTPError as e:
            raise TeacherError(f"{self.base_url} is not reachable ({type(e).__name__})") from None
        check(response, self.base_url)
        return [
            {"id": m["id"], "name": m.get("display_name") or m["id"], "free": False}
            for m in response.json().get("data", [])
        ]


class TeacherRuntime:
    """A saved teacher (an API model) answering in the chat, like an installed model."""

    def __init__(self, teacher):
        self.teacher = teacher

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        try:
            yield from self.teacher.stream_chat(request)
        except TeacherError as e:
            # Shown in the chat as the answer, as the error of a local model would be.
            yield TextDelta(f"[{e}]")
            yield Finish("stop", Usage())


class Paced:
    """Keeps a teacher under a requests-per-minute limit and waits out rate limits."""

    def __init__(self, teacher: Teacher, per_minute: float | None, retries: int = 5, sleep=None):
        self.teacher = teacher
        self.name = teacher.name
        self.interval = 60 / per_minute if per_minute else 0
        self.retries = retries
        self.sleep = sleep or time.sleep
        self.last = 0.0
        # Called with the number of seconds before waiting out a rate limit.
        self.on_wait = lambda seconds: None

    def complete(
        self, system: str, prompt: str, max_tokens: int = 2048, temperature: float = 0.7
    ) -> str:
        for attempt in range(self.retries + 1):
            wait = self.last + self.interval - time.monotonic()
            if wait > 0:
                self.sleep(wait)
            self.last = time.monotonic()
            try:
                return self.teacher.complete(system, prompt, max_tokens, temperature)
            except RateLimited as e:
                if attempt == self.retries:
                    raise
                self.on_wait(e.retry_after)
                self.sleep(e.retry_after)
        raise AssertionError("unreachable")

    @property
    def last_served(self) -> dict | None:
        return self.teacher.last_served
