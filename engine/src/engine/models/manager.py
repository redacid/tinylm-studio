"""Tracks installed models, downloads catalog models and keeps one model loaded at a time.

A model is any folder under <data_dir>/models containing a config.json, so users can also
drop in their own Hugging Face-format models by copying a folder there.
"""

import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from engine.inference.base import Runtime
from engine.models import catalog as model_catalog

log = logging.getLogger(__name__)

ECHO_MODEL = "echo"


class ModelNotFound(Exception):
    pass


@dataclass
class Download:
    model_id: str
    state: Literal["downloading", "done", "error"] = "downloading"
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ModelManager:
    def __init__(
        self,
        models_dir: Path,
        enable_echo: bool = False,
        hf_token: str | None = None,
        runtime_options: dict | None = None,
    ):
        self.models_dir = models_dir
        # Passed to TransformersRuntime, e.g. threads and repetition_penalty.
        self.runtime_options = runtime_options or {}
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.enable_echo = enable_echo
        self.hf_token = hf_token
        self.downloads: dict[str, Download] = {}
        self._loaded: tuple[str, Runtime] | None = None
        self._lock = threading.Lock()

    def installed(self) -> list[str]:
        # Hidden folders are unfinished downloads.
        ids = sorted(
            p.parent.name
            for p in self.models_dir.glob("*/config.json")
            if not p.parent.name.startswith(".")
        )
        return ([ECHO_MODEL] if self.enable_echo else []) + ids

    def catalog(self) -> list[dict]:
        installed = set(self.installed())
        return [
            {
                **spec.to_dict(),
                "installed": spec.id in installed,
                "download": d.to_dict() if (d := self.downloads.get(spec.id)) else None,
            }
            for spec in model_catalog.CATALOG
        ]

    def download(self, model_id: str) -> Download:
        spec = model_catalog.find(model_id)
        if spec is None:
            raise ModelNotFound(model_id)
        current = self.downloads.get(model_id)
        if current and current.state == "downloading":
            return current
        job = self.downloads[model_id] = Download(model_id)
        threading.Thread(target=self._download, args=(spec, job), daemon=True).start()
        return job

    def _download(self, spec: model_catalog.ModelSpec, job: Download) -> None:
        from huggingface_hub import snapshot_download

        target = self.models_dir / spec.id
        try:
            snapshot_download(
                spec.repo,
                local_dir=self.models_dir / f".{spec.id}.partial",
                token=self.hf_token,
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
            )
            (self.models_dir / f".{spec.id}.partial").rename(target)
            job.state = "done"
        except Exception as error:  # surfaced to the UI as text
            log.exception("Download of %s failed", spec.id)
            job.state, job.error = "error", str(error)

    def runtime(self, model_id: str) -> Runtime:
        """Return a runtime for the model, unloading any other model first."""
        if model_id == ECHO_MODEL and self.enable_echo:
            from engine.inference.echo import EchoRuntime

            return EchoRuntime()
        path = self.models_dir / model_id
        if not (path / "config.json").is_file():
            raise ModelNotFound(model_id)
        with self._lock:
            if self._loaded and self._loaded[0] == model_id:
                return self._loaded[1]
            self._loaded = None  # release memory of the previous model before loading
            from engine.inference.transformers_runtime import TransformersRuntime

            runtime = TransformersRuntime(path, **self.runtime_options)
            self._loaded = (model_id, runtime)
            return runtime
