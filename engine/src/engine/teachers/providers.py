"""Teacher models: larger models that write training examples for a small one.

Three kinds:
- local: a model installed in the studio (e.g. gemma-3-4b-gguf). Free and private, but slow on a
  CPU.
- openai: any OpenAI-compatible chat completions API: OpenRouter (including its free models),
  OpenAI, Groq, a local llama.cpp server, Ollama or LM Studio.
- anthropic: the Anthropic Messages API (Claude).
"""

import time
from typing import Literal, Protocol

import httpx

from engine.chat import ChatRequest, Message, collect

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


class RateLimited(TeacherError):
    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


class Teacher(Protocol):
    name: str
    # Who answered the last request: {"model": ..., "provider": ...}. OpenRouter names the model
    # that actually answered (it can differ from the one asked for) and the upstream provider.
    last_served: dict | None

    def complete(
        self, system: str, prompt: str, max_tokens: int = 2048, temperature: float = 0.7
    ) -> str: ...


def retry_after(response: httpx.Response, default: float = 30) -> float:
    try:
        return min(600.0, float(response.headers.get("retry-after", default)))
    except ValueError:
        return default


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
        try:
            response = self.client.post(f"{self.base_url}/chat/completions", json=body)
        except httpx.HTTPError as e:
            raise TeacherError(f"{self.base_url} is not reachable ({type(e).__name__})") from None
        check(response, self.name)
        try:
            data = response.json()
            if data.get("error"):
                # OpenRouter reports some upstream failures with HTTP 200.
                error = data["error"]
                message = error.get("message") if isinstance(error, dict) else error
                raise TeacherError(f"{self.name}: {message}")
            content = data["choices"][0]["message"]["content"] or ""
            self.last_served = {
                "model": data.get("model") or self.model,
                "provider": data.get("provider"),
            }
            return content
        except (ValueError, KeyError, IndexError, TypeError):
            raise TeacherError(f"{self.name}: unexpected answer {response.text[:300]}") from None

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
            "temperature": temperature,
        }
        try:
            response = self.client.post(f"{self.base_url}/v1/messages", json=body)
        except httpx.HTTPError as e:
            raise TeacherError(f"{self.base_url} is not reachable ({type(e).__name__})") from None
        check(response, self.name)
        try:
            data = response.json()
            text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
            self.last_served = {"model": data.get("model") or self.model, "provider": "Anthropic"}
            return text
        except (ValueError, KeyError, TypeError):
            raise TeacherError(f"{self.name}: unexpected answer {response.text[:300]}") from None

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


class Paced:
    """Keeps a teacher under a requests-per-minute limit and waits out rate limits."""

    def __init__(self, teacher: Teacher, per_minute: float | None, retries: int = 5, sleep=None):
        self.teacher = teacher
        self.name = teacher.name
        self.interval = 60 / per_minute if per_minute else 0
        self.retries = retries
        self.sleep = sleep or time.sleep
        self.last = 0.0
        # Called after each answer with who served it (see Teacher.last_served).
        self.on_served = lambda served: None
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
                text = self.teacher.complete(system, prompt, max_tokens, temperature)
            except RateLimited as e:
                if attempt == self.retries:
                    raise
                self.on_wait(e.retry_after)
                self.sleep(e.retry_after)
                continue
            if self.teacher.last_served:
                self.on_served(self.teacher.last_served)
            return text
        raise AssertionError("unreachable")
