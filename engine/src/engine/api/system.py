"""Health and system information endpoints. /health answers without the API key, for load
balancers and the desktop app's startup check; /api/system needs it (it shows paths and
hardware)."""

import sys
from typing import Literal

from fastapi import Depends, HTTPException, Query, Request
from pydantic import BaseModel, ValidationError

from engine import __version__, commit, hardware, logs
from engine.api.deps import require_api_key
from engine.api.schemas import (
    UNAUTHORIZED,
    EngineLog,
    Health,
    PublicEngineSettings,
    Router,
    SystemInfo,
    errors,
)
from engine.web.search_settings import validation_message

router = Router(tags=["system"])


@router.get("/health", response_model=Health)
def health() -> dict:
    """Answers without the API key."""
    return {"status": "ok", "version": __version__, "commit": commit()}


@router.get(
    "/api/logs",
    dependencies=[Depends(require_api_key)],
    response_model=EngineLog,
    responses=UNAUTHORIZED,
)
def engine_log(request: Request, lines: int = Query(500, ge=1, le=5000)) -> dict:
    """The end of the engine's log, oldest first; a traceback is part of its entry."""
    path = logs.log_file(request.app.state.settings.data_dir)
    return {"file": str(path), "lines": logs.tail(path, lines)}


@router.get(
    "/api/system",
    dependencies=[Depends(require_api_key)],
    response_model=SystemInfo,
    responses=UNAUTHORIZED,
)
def system_info() -> dict:
    """The engine's version, Python and hardware."""
    return {
        "version": __version__,
        "commit": commit(),
        "python": sys.executable,
        "hardware": hardware.detect().to_dict(),
    }


class EngineSettingsChanges(BaseModel, extra="forbid"):
    """The settings to change; the others keep their value. null: back to the environment's
    value (or the built-in default). hf_token, proxy and llama_cpp_dir: "" is none, even when the
    environment sets one."""

    hf_token: str | None = None
    # A password sent back as "****" keeps the saved one.
    proxy: str | None = None
    llama_cpp_dir: str | None = None
    threads: int | None = None
    repetition_penalty: float | None = None
    rag_top_k: int | None = None
    embedding_device: Literal["auto", "cuda", "cpu", "mps"] | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] | None = None


@router.get(
    "/api/settings",
    dependencies=[Depends(require_api_key)],
    response_model=PublicEngineSettings,
    responses=UNAUTHORIZED,
)
def engine_settings(request: Request) -> dict:
    """The engine settings the app can change (Settings > Engine options), with the
    environment's values they replace."""
    return request.app.state.engine_settings.public()


@router.put(
    "/api/settings",
    dependencies=[Depends(require_api_key)],
    response_model=PublicEngineSettings,
    responses={**UNAUTHORIZED, **errors(422)},
)
def save_engine_settings(body: EngineSettingsChanges, request: Request) -> dict:
    """Saves the given fields and puts them into effect at once: threads and the repetition
    penalty from the next model loaded, the embedding device from the next search or upload."""
    store = request.app.state.engine_settings
    try:
        store.update(body.model_dump(exclude_unset=True))
    except ValidationError as e:
        raise HTTPException(422, validation_message(e)) from None
    return store.public()
