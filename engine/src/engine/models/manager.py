"""Tracks installed models, downloads catalog models and keeps one model loaded at a time.

A model is any folder under <data_dir>/models that contains either a config.json (Hugging Face
format, run with PyTorch) or a .gguf file (run with llama.cpp), so users can also drop in their
own models by copying a folder there. Fine-tuned models are LoRA adapters: a folder with
adapter_config.json and a tinylm.json naming the installed base model they apply to.
"""

import json
import logging
import shutil
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
        """Models that can answer; a fine-tuned model needs its base model installed too."""
        ids = [
            p.name
            for p in self._folders()
            if (fmt := model_format(p)) in ("hf", "gguf")
            or (fmt == "adapter" and model_format(self.models_dir / adapter_base(p)) == "hf")
        ]
        return ([ECHO_MODEL] if self.enable_echo else []) + ids

    def _folders(self) -> list[Path]:
        # Hidden folders are unfinished downloads and training runs.
        return sorted(
            p for p in self.models_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
        )

    def has_token(self) -> bool:
        """Whether downloads send a Hugging Face token: from the settings, HF_TOKEN or a saved
        `hf auth login`."""
        from huggingface_hub import get_token

        return bool(self.hf_token or get_token())

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
        partial = self.models_dir / f".{spec.id}.partial"
        try:
            snapshot_download(
                spec.repo,
                local_dir=partial,
                token=self.hf_token,
                allow_patterns=[spec.file_pattern]
                if spec.format == "gguf"
                else ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
            )
            if model_format(partial) is None:
                raise RuntimeError(f"No model files in {spec.repo} matched the download filter")
            partial.rename(target)
            job.state = "done"
        except Exception as error:  # surfaced to the UI as text
            log.exception("Download of %s failed", spec.id)
            job.state, job.error = "error", download_error(spec.repo, error)

    def runtime(self, model_id: str) -> Runtime:
        """Return a runtime for the model, unloading any other model first."""
        if model_id == ECHO_MODEL and self.enable_echo:
            from engine.inference.echo import EchoRuntime

            return EchoRuntime()
        path = self.models_dir / model_id
        fmt = model_format(path)
        if fmt is None:
            raise ModelNotFound(model_id)
        with self._lock:
            if self._loaded and self._loaded[0] == model_id:
                return self._loaded[1]
            self._loaded = None  # release memory of the previous model before loading
            if fmt == "gguf":
                from engine.inference.llamacpp_runtime import LlamaCppRuntime

                runtime: Runtime = LlamaCppRuntime(path, **self.runtime_options)
            elif fmt == "adapter":
                from engine.inference.transformers_runtime import TransformersRuntime

                base = self.models_dir / adapter_base(path)
                if model_format(base) != "hf":
                    raise ModelNotFound(f"{model_id} needs its base model {base.name}")
                runtime = TransformersRuntime(base, adapter=path, **self.runtime_options)
            else:
                from engine.inference.transformers_runtime import TransformersRuntime

                runtime = TransformersRuntime(path, **self.runtime_options)
            self._loaded = (model_id, runtime)
            return runtime

    def trainable(self) -> list[str]:
        """Installed models that can be fine-tuned: PyTorch (Hugging Face) weights."""
        return [p.name for p in self._folders() if model_format(p) == "hf"]

    def trained(self) -> list[dict]:
        """Fine-tuned models (adapters) with what they were trained from."""
        models = []
        for path in self._folders():
            if model_format(path) == "adapter":
                info = json.loads((path / "tinylm.json").read_text())
                installed = model_format(self.models_dir / info["base"]) == "hf"
                models.append({"id": path.name, **info, "base_installed": installed})
        return models

    def delete(self, model_id: str) -> None:
        path = self.models_dir / model_id
        if model_id.startswith(".") or "/" in model_id or model_format(path) is None:
            raise ModelNotFound(model_id)
        with self._lock:
            if self._loaded and self._loaded[0] == model_id:
                self._loaded = None
            shutil.rmtree(path)


def download_error(repo: str, error: Exception) -> str:
    """Explains the usual reasons for a refused download of a gated model."""
    from huggingface_hub.errors import GatedRepoError

    if not isinstance(error, GatedRepoError):
        return str(error)
    page = f"https://huggingface.co/{repo}"
    response = getattr(error, "response", None)
    if response is not None and response.status_code == 401:
        return (
            f"{repo} is gated and the engine has no Hugging Face token. Put HF_TOKEN=... into "
            f".env and start the app with make, or set TINYLM_HF_TOKEN; then accept the "
            f"license at {page}."
        )
    return (
        f"Hugging Face refused access to {repo}: the token was sent, but its account has no "
        f"access yet. Open {page} while signed in and accept the license (access is granted "
        "at once for Gemma; a request may wait for approval for other models). A fine-grained "
        'token also needs the permission "Read access to contents of all public gated repos '
        'you can access".'
    )


def adapter_base(path: Path) -> str:
    return json.loads((path / "tinylm.json").read_text())["base"]


def model_format(path: Path) -> str | None:
    """Return "gguf", "hf" or "adapter" for a model folder, or None if it holds no model."""
    if not path.is_dir():
        return None
    if any(path.glob("*.gguf")):
        return "gguf"
    if (path / "adapter_config.json").is_file() and (path / "tinylm.json").is_file():
        return "adapter"
    if (path / "config.json").is_file():
        return "hf"
    return None
