"""Teacher models: larger models that write training examples for a small one.

Four kinds:
- local: a model installed in the studio (e.g. gemma-3-4b-gguf). Free and private, but slow on a
  CPU.
- openai: any OpenAI-compatible chat completions API: OpenRouter (including its free models),
  OpenAI, Groq, a local llama.cpp server, Ollama or LM Studio.
- anthropic: the Anthropic Messages API (Claude).
- cli: the command-line tool of a subscription (Claude Code, Codex or Gemini CLI), run headless.
  The tool keeps its own login; the studio never sees a token.
"""

import json
import logging
import os
import subprocess
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import httpx

from engine.chat import ChatRequest, Finish, Message, StreamEvent, TextDelta, Usage, collect

log = logging.getLogger(__name__)

TeacherKind = Literal["local", "openai", "anthropic", "cli"]
DEFAULT_BASE_URLS = {
    "openai": "https://openrouter.ai/api/v1",
    "anthropic": "https://api.anthropic.com",
    "cli": "claude",
}
ANTHROPIC_VERSION = "2023-06-01"
# OpenRouter shows these on its activity page; other providers ignore them.
APP_HEADERS = {"HTTP-Referer": "https://github.com/tinylm-studio", "X-Title": "tinylm-studio"}


ThinkingMode = Literal["default", "off", "on"]


@dataclass
class AnswerOptions:
    """A saved teacher's own answer settings, for every request it gets: the chat, generation,
    judging. None and "default" leave it to each request and the model."""

    # Tokens per answer. On models that think, thinking counts toward it.
    max_tokens: int | None = None
    temperature: float | None = None
    # Thinking (reasoning) of models that have it: the model's default, off or on.
    thinking: ThinkingMode = "default"

    def tokens(self, asked: int | None) -> int | None:
        return self.max_tokens or asked

    def heat(self, asked: float | None) -> float | None:
        return asked if self.temperature is None else self.temperature


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


def thought_out(name: str, max_tokens: int | None) -> str:
    """The chat's answer when the model spent the whole answer thinking."""
    return (
        f"[{name} used all {max_tokens or 'its'} tokens before answering, probably thinking: "
        "raise Max tokens or turn Thinking off in the teacher's settings]"
    )


class OpenAICompatibleTeacher:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        name: str = "",
        timeout: float = 300,
        options: AnswerOptions | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name or model
        self.options = options or AnswerOptions()
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
            "max_tokens": self.options.tokens(max_tokens),
            "temperature": self.options.heat(temperature),
            **self._thinking(),
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
            "max_tokens": self.options.tokens(request.max_tokens),
            "temperature": self.options.heat(request.temperature),
            "top_p": request.top_p,
            "stream": True,
            **self._thinking(),
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
        if reason == "length" and not pieces:
            yield TextDelta(thought_out(self.name, body["max_tokens"]))
        yield Finish(reason, usage)

    def _thinking(self) -> dict:
        """OpenRouter's reasoning switch; other APIs get nothing unless it is set."""
        if self.options.thinking == "default":
            return {}
        return {"reasoning": {"enabled": self.options.thinking == "on"}}

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
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        name: str = "",
        timeout: float = 300,
        options: AnswerOptions | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name or model
        self.options = options or AnswerOptions()
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
            "max_tokens": self.options.tokens(max_tokens),
            **self._thinking(),
        }
        self.last_served = {"model": self.model, "provider": "Anthropic"}
        try:
            response = self.client.post(
                f"{self.base_url}/v1/messages",
                json=self._sampled(body, self.options.heat(temperature)),
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
            # The API refuses an empty message, such as an answer that was cut off.
            "messages": [
                {"role": m.role, "content": m.content}
                for m in request.messages
                if m.role != "system" and m.content
            ],
            "max_tokens": self.options.tokens(request.max_tokens) or 1024,
            "stream": True,
            **self._thinking(),
        }
        if system:
            body["system"] = system
        if request.stop:
            body["stop_sequences"] = request.stop
        try:
            yield from self._stream(self._sampled(body, self.options.heat(request.temperature)))
        except SamplingRefused:
            yield from self._stream(body)

    def _thinking(self) -> dict:
        """Adaptive thinking (Claude 4.6 and later) when on; nothing for the model's default."""
        if self.options.thinking == "default":
            return {}
        return {"thinking": {"type": "adaptive" if self.options.thinking == "on" else "disabled"}}

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
        reason, stop_sequence, usage, answered = "stop", None, Usage(), False
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
                        # Only the answer: thinking comes in thinking_delta events.
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta" and (text := delta.get("text")):
                            answered = True
                            yield TextDelta(text)
                    elif kind == "message_delta":
                        delta = data.get("delta", {})
                        reason = reasons.get(delta.get("stop_reason"), "stop")
                        stop_sequence = delta.get("stop_sequence")
                        usage.output_tokens = data.get("usage", {}).get("output_tokens", 0)
        except httpx.HTTPError as e:
            raise TeacherError(f"{self.base_url} is not reachable ({type(e).__name__})") from None
        if reason == "length" and not answered:
            yield TextDelta(thought_out(self.name, body["max_tokens"]))
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


# Command-line tools a cli teacher can run, by the name of the program.
CLI_TOOLS = ("claude", "codex", "gemini")
# Model names the tools take besides full ones; empty leaves the model to the tool.
CLI_MODELS = {
    "claude": [
        {"id": "sonnet", "name": "Claude Sonnet (latest)", "free": False},
        {"id": "opus", "name": "Claude Opus (latest)", "free": False},
        {"id": "haiku", "name": "Claude Haiku (latest)", "free": False},
    ],
}
# With one of these set a tool bills the API instead of the subscription it is logged in with.
API_KEY_VARIABLES = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)
# Words in a tool's error that mean the subscription's usage limit was reached.
LIMIT_WORDS = ("usage limit", "rate limit", "limit reached", "quota", "429")
# How long to wait out a reached limit; subscription limits reset within a few hours.
LIMIT_WAIT = 600.0
LOGIN_WORDS = (
    "log in",
    "login",
    "logged in",
    "not authenticated",
    "unauthorized",
    "401",
    "auth method",
)


def cli_tool(program: str) -> str:
    """Which tool a program is ("claude" for /usr/local/bin/claude), by its name."""
    name = Path(program).name.lower()
    for tool in CLI_TOOLS:
        if name.startswith(tool):
            return tool
    raise TeacherError(f"{program}: not a known tool, use one of {', '.join(CLI_TOOLS)}")


def json_events(text: str) -> list[dict]:
    """The JSON objects a tool printed: one document, or one per line."""
    try:
        data = json.loads(text)
        return [data] if isinstance(data, dict) else [d for d in data if isinstance(d, dict)]
    except ValueError:
        pass
    events = []
    for line in text.splitlines():
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict):
            events.append(data)
    if not events and "{" in text:
        # One indented document after some text, as Gemini CLI prints its errors.
        try:
            data, _ = json.JSONDecoder().raw_decode(text[text.index("{") :])
            events = [data] if isinstance(data, dict) else []
        except ValueError:
            pass
    return events


def transcript(messages: list[Message]) -> tuple[str, str]:
    """A conversation as one prompt, for tools that take a single one, and its system prompt."""
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    turns = [m for m in messages if m.role != "system" and m.content]
    if len(turns) == 1:
        return system, turns[0].content
    lines = [f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in turns]
    lines.append("Write the assistant's next answer only.")
    return system, "\n\n".join(lines)


class CliTeacher:
    """A subscription's command-line tool (Claude Code, Codex, Gemini CLI) run headless, one
    process per request, without its tools. It answers with the account it is logged in with,
    which the user logs in once in the app (Settings › Subscriptions, see cli_tools.py)."""

    def __init__(
        self,
        program: str,
        model: str = "",
        name: str = "",
        timeout: float = 600,
        options: AnswerOptions | None = None,
    ):
        self.program = program or "claude"
        self.tool = cli_tool(self.program)
        self.model = model
        self.name = name or model or self.tool
        self.timeout = timeout
        self.options = options or AnswerOptions()
        self.last_served: dict | None = None
        # The environment the tool runs in, with the engine's logins (CliTools.environment());
        # None: the engine's own without API keys.
        self.environment: dict | None = None

    def complete(
        self, system: str, prompt: str, max_tokens: int = 2048, temperature: float = 0.7
    ) -> str:
        # The tools set no token limit or temperature: their models answer with their own.
        self.last_served = {"model": self.model or "default", "provider": self.tool}
        args, stdin = self._command(system, prompt, stream=False)
        events = json_events(self._run(args, stdin))
        return getattr(self, f"_{self.tool}_answer")(events)

    def stream_chat(self, request: ChatRequest) -> Iterator[StreamEvent]:
        """Answers a whole conversation; Claude Code as it is written, the others at once."""
        system, prompt = transcript(request.messages)
        if self.tool != "claude":
            text = self.complete(system, prompt)
            yield TextDelta(text)
            yield Finish("stop", Usage(output_tokens=len(text.split())))
            return
        yield from self._claude_stream(system, prompt)

    def models(self) -> list[dict]:
        return CLI_MODELS.get(self.tool, [])

    def _command(self, system: str, prompt: str, stream: bool) -> tuple[list[str], str]:
        """The tool's arguments and what it reads from stdin."""
        model = ["--model", self.model] if self.model else []
        if self.tool == "claude":
            output = (
                ["stream-json", "--verbose", "--include-partial-messages"] if stream else ["json"]
            )
            args = [
                self.program, "-p", "--output-format", *output, "--tools", "",
                "--strict-mcp-config", "--no-session-persistence", *model,
            ]  # fmt: skip
            if system:
                args += ["--system-prompt", system]
            return args, prompt
        # Codex and Gemini CLI keep their own system prompt: ours goes before the request.
        stdin = f"{system}\n\n{prompt}" if system else prompt
        if self.tool == "codex":
            args = [
                self.program, "exec", "--json", "--skip-git-repo-check", "--ephemeral",
                "--sandbox", "read-only", *model, "-",
            ]  # fmt: skip
            return args, stdin
        return [self.program, "--output-format", "json", "--approval-mode", "plan", *model,
                "--prompt", ""], stdin  # fmt: skip

    def _environment(self) -> dict:
        if self.environment is not None:
            return self.environment
        return {k: v for k, v in os.environ.items() if k not in API_KEY_VARIABLES}

    def _run(self, args: list[str], stdin: str) -> str:
        # An empty folder: the tool finds no project files or instructions of its own there.
        with tempfile.TemporaryDirectory(prefix="tinylm-teacher-") as folder:
            try:
                done = subprocess.run(
                    args, input=stdin, capture_output=True, text=True, cwd=folder,
                    env=self._environment(), timeout=self.timeout,
                )  # fmt: skip
            except FileNotFoundError:
                raise TeacherError(f"{self.program} is not installed or not on PATH") from None
            except subprocess.TimeoutExpired:
                raise TeacherError(f"{self.name} did not answer in {self.timeout:.0f} s") from None
        if done.returncode != 0:
            # Some tools print their error as JSON on stdout, others as text on stderr.
            events = json_events(done.stdout) + json_events(done.stderr)
            self._failed(self._error(events) or done.stderr or done.stdout)
        return done.stdout

    def _error(self, events: list[dict]) -> str:
        for event in reversed(events):
            error = event.get("error")
            if isinstance(error, dict) and error.get("message"):
                return error["message"]
            if event.get("type") == "error" and event.get("message"):
                return event["message"]
            if event.get("is_error") and event.get("result"):
                return event["result"]
        return ""

    def _failed(self, message: str):
        message = " ".join(message.split())[-300:] or "no output"
        lowered = message.lower()
        if any(w in lowered for w in LIMIT_WORDS):
            raise RateLimited(f"{self.name}: {message}", LIMIT_WAIT)
        if any(w in lowered for w in LOGIN_WORDS):
            raise TeacherError(
                f"{self.name} is not logged in: log in under Settings › Subscriptions"
            )
        raise TeacherError(f"{self.name}: {message}")

    def _claude_answer(self, events: list[dict]) -> str:
        for event in reversed(events):
            if event.get("type") == "result":
                if event.get("is_error"):
                    self._failed(event.get("result") or event.get("subtype") or "")
                return event.get("result") or ""
        self._failed(self._error(events) or "no answer")

    def _codex_answer(self, events: list[dict]) -> str:
        # Codex reports reconnecting as errors too: only a failed turn without an answer counts.
        answers = [
            e["item"].get("text") or ""
            for e in events
            if e.get("type") == "item.completed"
            and isinstance(e.get("item"), dict)
            and e["item"].get("type") == "agent_message"
        ]
        if not answers:
            self._failed(self._error(events) or "no answer")
        return answers[-1]

    def _gemini_answer(self, events: list[dict]) -> str:
        if error := self._error(events):
            self._failed(error)
        for event in reversed(events):
            if "response" in event:
                return event.get("response") or ""
        self._failed("no answer")

    def _claude_stream(self, system: str, prompt: str) -> Iterator[StreamEvent]:
        self.last_served = {"model": self.model or "default", "provider": self.tool}
        args, stdin = self._command(system, prompt, stream=True)
        usage, answered, reason = Usage(), False, "stop"
        with tempfile.TemporaryDirectory(prefix="tinylm-teacher-") as folder:
            try:
                process = subprocess.Popen(
                    args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, cwd=folder, env=self._environment(),
                )  # fmt: skip
            except FileNotFoundError:
                raise TeacherError(f"{self.program} is not installed or not on PATH") from None
            try:
                process.stdin.write(stdin)
                process.stdin.close()
                for line in process.stdout:
                    try:
                        data = json.loads(line)
                    except ValueError:
                        continue
                    kind = data.get("type")
                    if kind == "stream_event":
                        event = data.get("event") or {}
                        delta = event.get("delta") or {}
                        if event.get("type") == "content_block_delta" and delta.get("type") == (
                            "text_delta"
                        ):
                            answered = True
                            yield TextDelta(delta.get("text") or "")
                    elif kind == "result":
                        if data.get("is_error"):
                            self._failed(data.get("result") or data.get("subtype") or "")
                        counted = data.get("usage") or {}
                        usage.input_tokens = counted.get("input_tokens") or 0
                        usage.output_tokens = counted.get("output_tokens") or 0
                        if data.get("stop_reason") == "max_tokens":
                            reason = "length"
                        if not answered and data.get("result"):
                            yield TextDelta(data["result"])
                            answered = True
                if process.wait() != 0 and not answered:
                    self._failed(process.stderr.read())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        yield Finish(reason, usage)


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
