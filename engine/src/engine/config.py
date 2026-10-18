"""Engine settings, read from environment variables prefixed with TINYLM_."""

import os
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from engine.rag.embeddings import DEFAULT_EMBEDDING_MODEL
from engine.web.fetch import DEFAULT_USER_AGENT


def default_data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / "tinylm-studio"


class Settings(BaseSettings):
    # An empty variable (HF_TOKEN= in .env, TINYLM_THREADS: "" in docker-compose.yaml) means
    # the default, not an empty value.
    model_config = SettingsConfigDict(
        env_prefix="TINYLM_", populate_by_name=True, env_ignore_empty=True
    )

    host: str = "127.0.0.1"
    port: int = 8765
    data_dir: Path = default_data_dir()
    # When set, clients must send it as "Authorization: Bearer <key>" or "x-api-key: <key>".
    api_key: str | None = None
    # Offer the built-in "echo" model (no weights needed); for tests and UI development.
    enable_echo: bool = False
    # Browser origins allowed to call the API; the desktop app sets this for its renderer.
    cors_origins: list[str] = []
    # Hugging Face token; the standard HF_TOKEN works too.
    hf_token: str | None = Field(None, validation_alias=AliasChoices("TINYLM_HF_TOKEN", "HF_TOKEN"))
    # CPU threads for inference; default is physical cores, at most 8.
    threads: int | None = None
    # Penalizes repeated tokens; small models loop without it.
    repetition_penalty: float = 1.1
    # Sentence-transformers model used to index and search uploaded data;
    # "hash" is a crude built-in one that needs no download, for tests and UI development.
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    # Where it runs: auto (an NVIDIA GPU when there is one), cuda, cpu or mps.
    embedding_device: str = "auto"
    # How many chunks are added to the prompt when a chat request names a collection.
    rag_top_k: int = 4
    # Website crawling. proxy: http://user:pass@host:port, socks5://host:port or socks5h://...
    proxy: str | None = None
    # Chrome DevTools endpoint of a running browser, used for sites that need JavaScript.
    cdp_url: str = "http://127.0.0.1:9222"
    # These are defaults: what is saved in the app's Settings > Crawler wins over them.
    # Default requests per minute to each site; keep it low to stay polite and avoid bans.
    crawl_per_minute: float = 20
    # False: robots.txt is not read for any crawl.
    crawl_respect_robots: bool = True
    # New pages per crawl and link hops from a start address, for new websites and crawls.
    crawl_max_pages: int = 50
    crawl_max_depth: int = 2
    # Load pages in the browser at cdp_url by default (for sites that need JavaScript).
    crawl_use_browser: bool = False
    user_agent: str = DEFAULT_USER_AGENT
    # Web search for finding pages and videos: a self-hosted SearXNG (free, no key), or Brave,
    # Tavily, Exa, Perplexity or Serper (each needs its API key; see engine.web.search).
    # These are defaults: what is saved in the app's Settings > Search wins over them.
    search_provider: Literal["searxng", "brave", "tavily", "exa", "perplexity", "serper"] = (
        "searxng"
    )
    searxng_url: str = "http://127.0.0.1:8888"
    # Only results from these sites, and never from these (JSON lists of domains or patterns
    # such as "*.ua").
    search_include_domains: list[str] = []
    search_exclude_domains: list[str] = []
    tavily_api_key: str | None = None
    exa_api_key: str | None = None
    perplexity_api_key: str | None = None
    serper_api_key: str | None = None
    # A llama.cpp checkout whose convert_hf_to_gguf.py exports fine-tuned models to GGUF; by
    # default the converter of the installed llama-cpp-python is downloaded once from PyPI.
    llama_cpp_dir: Path | None = None
    brave_api_key: str | None = None
    # Where the subscription tools of cli teachers (Claude Code, Codex CLI, Gemini CLI) keep the
    # logins made in the app; default <data dir>/cli-logins.
    cli_home: Path | None = None
    # DEBUG, INFO, WARNING or ERROR: what goes to the console and <data dir>/logs/engine.log.
    log_level: str = "INFO"

    @property
    def workspace_dir(self) -> Path:
        return self.data_dir / "workspace"

    @property
    def knowledge_db(self) -> Path:
        return self.workspace_dir / "knowledge.db"

    @property
    def cli_logins_dir(self) -> Path:
        return self.cli_home or self.data_dir / "cli-logins"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"
