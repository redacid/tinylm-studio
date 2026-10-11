"""Health and system information endpoints."""

import sys

from fastapi import APIRouter

from engine import __version__, hardware

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@router.get("/api/system")
def system_info() -> dict:
    return {
        "version": __version__,
        "python": sys.executable,
        "hardware": hardware.detect().to_dict(),
    }
