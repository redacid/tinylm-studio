"""Health and system information endpoints. /health answers without the API key, for load
balancers and the desktop app's startup check; /api/system needs it (it shows paths and
hardware)."""

import sys

from fastapi import APIRouter, Depends, Query, Request

from engine import __version__, commit, hardware, logs
from engine.api.deps import require_api_key

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "commit": commit()}


@router.get("/api/logs", dependencies=[Depends(require_api_key)])
def engine_log(request: Request, lines: int = Query(500, ge=1, le=5000)) -> dict:
    """The end of the engine's log, oldest first; a traceback is part of its entry."""
    path = logs.log_file(request.app.state.settings.data_dir)
    return {"file": str(path), "lines": logs.tail(path, lines)}


@router.get("/api/system", dependencies=[Depends(require_api_key)])
def system_info() -> dict:
    return {
        "version": __version__,
        "commit": commit(),
        "python": sys.executable,
        "hardware": hardware.detect().to_dict(),
    }
