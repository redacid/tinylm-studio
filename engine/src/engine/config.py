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

    @property
    def workspace_dir(self) -> Path:
        return self.data_dir / "workspace"
