"""Engine settings the app can change: the Hugging Face token, the proxy, the hardware and answer
defaults, the GGUF converter and the log level.

The defaults come from TINYLM_* environment variables (and HF_TOKEN; see engine.config); what is
changed in the app's Settings > Engine options is saved in <workspace>/engine-settings.json and
wins over them. The token is never sent back to clients (only a hint like "hf_…a1b2"), and the
proxy's password is masked.
"""

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field, field_validator

from engine.config import Settings

FIELDS = (
    "hf_token",
    "proxy",
    "threads",
    "repetition_penalty",
    "rag_top_k",
    "embedding_device",
    "llama_cpp_dir",
    "log_level",
)
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
DEVICES = ("auto", "cuda", "cpu", "mps")
# Stands for the proxy's password in what clients see and send back.
MASK = "****"


class EngineOptions(BaseModel):
    # "" for the three below: none, even when the environment sets one.
    hf_token: str | None = None
    # http://user:pass@host:port, socks5://host:port or socks5h://...
    proxy: str | None = None
    # A llama.cpp checkout on the engine's machine, for exporting to GGUF.
    llama_cpp_dir: str | None = None
    # None: physical cores, at most 8.
    threads: int | None = Field(None, ge=1, le=256)
    repetition_penalty: float = Field(1.1, ge=0.5, le=2)
    rag_top_k: int = Field(4, ge=1, le=50)
    embedding_device: Literal["auto", "cuda", "cpu", "mps"] = "auto"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("hf_token", "proxy", "llama_cpp_dir", mode="before")
    @classmethod
    def stripped(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("proxy")
    @classmethod
    def proxy_address(cls, value: str | None) -> str | None:
        if value and not value.startswith(("http://", "https://", "socks5://", "socks5h://")):
            raise ValueError("not an http://, https://, socks5:// or socks5h:// address")
        return value

    @field_validator("llama_cpp_dir")
    @classmethod
    def folder(cls, value: str | None) -> str | None:
        if value and not Path(value).expanduser().is_dir():
            raise ValueError(f"no folder {value} on the engine's machine")
        return value

    @field_validator("embedding_device", mode="before")
    @classmethod
    def lower(cls, value):
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("log_level", mode="before")
    @classmethod
    def upper(cls, value):
        return value.strip().upper() if isinstance(value, str) else value


def env_defaults(settings: Settings) -> dict:
    """The environment's values; a log level or device it spells wrong falls back to the
    default instead of breaking the settings."""
    level = settings.log_level.strip().upper()
    device = settings.embedding_device.strip().lower()
    return {
        "hf_token": settings.hf_token,
        "proxy": settings.proxy,
        "llama_cpp_dir": str(settings.llama_cpp_dir) if settings.llama_cpp_dir else None,
        "threads": settings.threads,
        "repetition_penalty": settings.repetition_penalty,
        "rag_top_k": settings.rag_top_k,
        "embedding_device": device if device in DEVICES else "auto",
        "log_level": level if level in LOG_LEVELS else "INFO",
    }


def overlay(options: EngineOptions) -> dict:
    """The Settings fields for these options."""
    folder = options.llama_cpp_dir
    return {
        "hf_token": options.hf_token,
        "proxy": options.proxy or None,
        "llama_cpp_dir": Path(folder).expanduser() if folder else None,
        "threads": options.threads,
        "repetition_penalty": options.repetition_penalty,
        "rag_top_k": options.rag_top_k,
        "embedding_device": options.embedding_device,
        "log_level": options.log_level,
    }


def token_hint(token: str | None) -> str | None:
    """Enough of a token to recognize it: "hf_…a1b2"."""
    if not token:
        return None
    return f"{token[:3]}…{token[-4:]}" if len(token) > 12 else MASK


def mask_proxy(proxy: str | None) -> str | None:
    if not proxy:
        return proxy
    parts = urlsplit(proxy)
    if parts.password is None:
        return proxy
    host = parts.netloc.rpartition("@")[2]
    return urlunsplit(parts._replace(netloc=f"{parts.username}:{MASK}@{host}"))


def unmask_proxy(proxy: str | None, current: str | None) -> str | None:
    """The proxy sent back with the masked password, with the current password in its place."""
    if not proxy or not current:
        return proxy
    parts = urlsplit(proxy)
    if parts.password != MASK:
        return proxy
    password = urlsplit(current).password or ""
    host = parts.netloc.rpartition("@")[2]
    return urlunsplit(parts._replace(netloc=f"{parts.username}:{password}@{host}"))


def saved_login_token() -> str | None:
    """A token saved by `hf auth login` on the engine's machine."""
    from huggingface_hub.constants import HF_TOKEN_PATH

    try:
        return Path(HF_TOKEN_PATH).read_text().strip() or None
    except OSError:
        return None


class EngineSettingsStore:
    def __init__(
        self,
        path: Path,
        settings: Settings,
        on_change: Callable[[EngineOptions], None] | None = None,
    ):
        self.path = path
        # The environment's values, before anything saved is applied.
        self.env = settings.model_copy()
        self.on_change = on_change
        self._lock = threading.Lock()

    def _saved(self) -> dict:
        try:
            saved = json.loads(self.path.read_text())
        except (FileNotFoundError, ValueError):
            return {}
        return {k: v for k, v in saved.items() if k in FIELDS} if isinstance(saved, dict) else {}

    def get(self) -> EngineOptions:
        # The environment's values are taken as they are: a wrong one there is reported where
        # it is used, not here.
        return EngineOptions.model_construct(**{**env_defaults(self.env), **self._valid_saved()})

    def _valid_saved(self) -> dict:
        """The saved values that are still valid (a folder may have gone since)."""
        valid = {}
        for name, value in self._saved().items():
            try:
                valid[name] = checked(name, value)
            except ValueError:
                continue
        return valid

    def update(self, changes: dict) -> EngineOptions:
        """Saves the given fields; a field left out keeps its value, null goes back to the
        environment's (or the built-in) default. Raises ValidationError."""
        with self._lock:
            saved = self._valid_saved()
            if changes.get("proxy"):
                current = {**env_defaults(self.env), **saved}.get("proxy")
                changes = {**changes, "proxy": unmask_proxy(changes["proxy"], current)}
            for name, value in changes.items():
                if value is None:
                    saved.pop(name, None)
                else:
                    saved[name] = checked(name, value)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(saved, indent=2))
            tmp.chmod(0o600)  # holds the token and the proxy's password
            tmp.replace(self.path)
        options = self.get()
        if self.on_change:
            self.on_change(options)
        return options

    def public(self) -> dict:
        """The settings without the token, and the environment's values the saved ones replace."""
        saved = self._valid_saved()
        current = self.get()
        env = EngineOptions.model_construct(**env_defaults(self.env))
        login = saved_login_token()
        if saved.get("hf_token"):
            source, token = "app", current.hf_token
        elif current.hf_token == "":
            source, token = None, None
        elif env.hf_token:
            source, token = "environment", env.hf_token
        elif login:
            source, token = "login", login
        else:
            source, token = None, None
        return {
            "hf_token_set": bool(token),
            "hf_token_hint": token_hint(token),
            "hf_token_source": source,
            **values(current),
            "saved": sorted(saved),
            "defaults": {
                "hf_token_hint": token_hint(env.hf_token or login),
                **values(env),
            },
        }


def checked(name: str, value):
    """The value, validated and normalized like the field it is for."""
    return getattr(EngineOptions.model_validate({name: value}), name)


def values(options: EngineOptions) -> dict:
    return {
        "proxy": mask_proxy(options.proxy),
        "llama_cpp_dir": options.llama_cpp_dir,
        "threads": options.threads,
        "repetition_penalty": options.repetition_penalty,
        "rag_top_k": options.rag_top_k,
        "embedding_device": options.embedding_device,
        "log_level": options.log_level,
    }


def engine_settings(settings: Settings, on_change=None) -> EngineSettingsStore:
    return EngineSettingsStore(settings.workspace_dir / "engine-settings.json", settings, on_change)


def with_saved(settings: Settings) -> Settings:
    """The settings with what was saved in the app's Settings > Engine options applied."""
    options = engine_settings(settings).get()
    return settings.model_copy(update=overlay(options))
