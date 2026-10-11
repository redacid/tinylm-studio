"""Engine settings, read from environment variables prefixed with TINYLM_."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def default_data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / "tinylm-studio"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TINYLM_")

    host: str = "127.0.0.1"
    port: int = 8765
    data_dir: Path = default_data_dir()
    # When set, clients must send it as "Authorization: Bearer <key>" or "x-api-key: <key>".
    api_key: str | None = None
    # Offer the built-in "echo" model (no weights needed); for tests and UI development.
    enable_echo: bool = False
    # Browser origins allowed to call the API; the desktop app sets this for its renderer.
    cors_origins: list[str] = []
    hf_token: str | None = None
    # CPU threads for inference; default is physical cores, at most 8.
    threads: int | None = None
    # Penalizes repeated tokens; small models loop without it.
    repetition_penalty: float = 1.1

    @property
    def workspace_dir(self) -> Path:
        return self.data_dir / "workspace"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"
