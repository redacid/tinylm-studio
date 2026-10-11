"""FastAPI application factory."""

from fastapi import FastAPI

from engine import __version__
from engine.api import system
from engine.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.workspace_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="tinylm-studio engine", version=__version__)
    app.state.settings = settings
    app.include_router(system.router)
    return app
