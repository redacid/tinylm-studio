"""Health and system information endpoints. /health answers without the API key, for load
balancers and the desktop app's startup check; /api/system needs it (it shows paths and
hardware)."""

import sys

from fastapi import APIRouter, Depends

from engine import __version__, hardware
from engine.api.deps import require_api_key

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@router.get("/api/system", dependencies=[Depends(require_api_key)])
def system_info() -> dict:
    return {
        "version": __version__,
        "python": sys.executable,
        "hardware": hardware.detect().to_dict(),
    }
