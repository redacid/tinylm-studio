"""Per-model settings: how a model is loaded and the answer settings it uses by default.

Loading settings (context size, threads, GPU layers, batch size) take effect when the model is
loaded, so changing them unloads it. Answer settings fill in what a request leaves out: a chat
request that sends its own temperature keeps it.
"""

import json
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from pydantic import BaseModel, Field

from engine.chat import ChatRequest

LOAD_KEYS = ("context_size", "threads", "gpu_layers", "batch_size")
# Used when neither the request nor the model's settings say otherwise.
ANSWER_DEFAULTS = {"max_tokens": 512, "temperature": 0.7, "top_p": 1.0, "thinking": False}


class ModelSettings(BaseModel):
    """Every field is optional: None means the engine's default."""

    # Loading (GGUF through llama.cpp, except threads which PyTorch models use too).
    context_size: int | None = Field(None, ge=256, le=262144)
    threads: int | None = Field(None, ge=1, le=256)
    gpu_layers: int | None = Field(None, ge=-1, le=1000)
    batch_size: int | None = Field(None, ge=1, le=8192)
    # Answers.
    max_tokens: int | None = Field(None, ge=1, le=32768)
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, gt=0, le=1)
    top_k: int | None = Field(None, ge=0, le=1000)
    min_p: float | None = Field(None, ge=0, le=1)
    repetition_penalty: float | None = Field(None, gt=0, le=2)
    # Lets models with a reasoning mode (Qwen3) think before answering.
    thinking: bool | None = None
    # Data: requests that name no collection answer from this one, with sources; "" turns off
    # a collection that came bundled with the model.
    collection: str | None = Field(None, max_length=64)
    rag_top_k: int | None = Field(None, ge=1, le=20)

    def values(self) -> dict:
        return self.model_dump(exclude_none=True)


class ModelSettingsStore:
    """Settings of all models in one JSON file, keyed by model id."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict[str, dict]:
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def get(self, model_id: str) -> dict:
        return self._read().get(model_id, {})

    def set(self, model_id: str, settings: ModelSettings) -> dict:
        with self._lock:
            data = self._read()
            if values := settings.values():
                data[model_id] = values
            else:
                data.pop(model_id, None)
            self.path.write_text(json.dumps(data, indent=2))
        return values

    def forget(self, model_id: str) -> None:
        with self._lock:
            data = self._read()
            if data.pop(model_id, None) is not None:
                self.path.write_text(json.dumps(data, indent=2))


def load_options(settings: dict) -> dict:
    return {k: settings[k] for k in LOAD_KEYS if k in settings}


def with_defaults(request: ChatRequest, settings: dict) -> ChatRequest:
    """Fills in what the request left out: from the model's settings, then the defaults.
    top_k, min_p and the repetition penalty stay None when unset: the runtime's own default."""
    changes = {}
    for key in ("max_tokens", "temperature", "top_p", "top_k", "min_p", "repetition_penalty"):
        if getattr(request, key) is None:
            changes[key] = settings.get(key, ANSWER_DEFAULTS.get(key))
    if request.thinking is None:
        changes["thinking"] = settings.get("thinking", ANSWER_DEFAULTS["thinking"])
    return replace(request, **changes)


class ConfiguredRuntime:
    """A loaded runtime that answers with the model's current answer settings."""

    def __init__(self, runtime, settings: Callable[[], dict]):
        self.runtime = runtime
        self._settings = settings

    def stream(self, request: ChatRequest):
        return self.runtime.stream(with_defaults(request, self._settings()))
