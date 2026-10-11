"""Detect compute devices available to the engine."""

import os
import platform
from dataclasses import asdict, dataclass

import psutil


@dataclass
class HardwareInfo:
    cpu: str
    cpu_cores: int
    memory_gb: float
    accelerators: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _detect_accelerators() -> list[str]:
    """Return available accelerators. PyTorch is optional at this stage."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return []
    found = []
    if torch.cuda.is_available():
        found.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        found.append("mps")
    return found


def detect() -> HardwareInfo:
    return HardwareInfo(
        cpu=platform.processor() or platform.machine(),
        cpu_cores=os.cpu_count() or 1,
        memory_gb=round(psutil.virtual_memory().total / 2**30, 1),
        accelerators=_detect_accelerators(),
    )
