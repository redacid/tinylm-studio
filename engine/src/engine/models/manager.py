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
from engine.models import hub
from engine.models.settings import (
    ConfiguredRuntime,
    ModelSettings,
    ModelSettingsStore,
    load_options,
)

log = logging.getLogger(__name__)

ECHO_MODEL = "echo"
# The chat can also answer with a saved teacher (an API model): "tinylm/teacher/<name>".
TEACHER_PREFIX = "tinylm/teacher/"
# Written into a downloaded model's folder: the Hugging Face repo (and file) it came from.
SOURCE_FILE = "tinylm-source.json"
TRANSFORMERS_OPTIONS = ("threads", "repetition_penalty")


class ModelNotFound(Exception):
    pass


class ModelExists(Exception):
    pass


@dataclass
class Download:
    model_id: str
    repo: str = ""
    state: Literal["downloading", "done", "error"] = "downloading"
    error: str | None = None
    # Bytes to download, when Hugging Face told; the folder being filled shows the progress.
    total: int | None = None
    folder: Path | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        del data["folder"]
        data["downloaded"] = (
            folder_size(self.folder)
            if self.state == "downloading" and self.folder
            else (self.total if self.state == "done" else None)
        )
        return data


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
        self.settings = ModelSettingsStore(models_dir / ".model-settings.json")
        self.downloads: dict[str, Download] = {}
        self._loaded: tuple[str, Runtime] | None = None
        self._lock = threading.Lock()
        # The TeacherStore, set by the app: makes "tinylm/teacher/<name>" models answer in the chat.
        self.teachers = None

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
        """Downloads a model of the catalog."""
        spec = model_catalog.find(model_id)
        if spec is None:
            raise ModelNotFound(model_id)
        patterns = [spec.file_pattern] if spec.format == "gguf" else hub.HF_PATTERNS
        return self._start(spec.id, spec.repo, patterns, {"repo": spec.repo})

    def download_from_hub(self, repo: str, file: str | None = None) -> Download:
        """Downloads any model from Hugging Face: one GGUF file (all parts of a split one) or,
        without a file, the Hugging Face-format weights of the repo."""
        if file:
            if not file.endswith(".gguf"):
                raise ValueError("Only .gguf files can be downloaded one by one")
            model_id = hub.model_id(repo, "gguf", file)
            patterns = hub.gguf_patterns(file)
        else:
            model_id, patterns = hub.model_id(repo, "hf"), hub.HF_PATTERNS
        if (self.models_dir / model_id).exists() or model_id == ECHO_MODEL:
            raise ModelExists(model_id)
        return self._start(model_id, repo, patterns, {"repo": repo, "file": file})

    def _start(self, model_id: str, repo: str, patterns: list[str], source: dict) -> Download:
        current = self.downloads.get(model_id)
        if current and current.state == "downloading":
            return current
        partial = self.models_dir / f".{model_id}.partial"
        job = self.downloads[model_id] = Download(model_id, repo, folder=partial)
        log.info("Downloading %s from %s", model_id, repo)
        threading.Thread(target=self._download, args=(job, patterns, source), daemon=True).start()
        return job

    def _download(self, job: Download, patterns: list[str], source: dict) -> None:
        from huggingface_hub import HfApi, snapshot_download

        target = self.models_dir / job.model_id
        partial = job.folder
        try:
            try:
                info = HfApi(token=self.hf_token).model_info(job.repo, files_metadata=True)
                job.total = sum(
                    s.size or 0 for s in info.siblings or [] if hub.wanted(s.rfilename, patterns)
                )
            except Exception:  # only the progress needs it; the download says what is wrong
                job.total = None
            snapshot_download(
                job.repo, local_dir=partial, token=self.hf_token, allow_patterns=patterns
            )
            # GGUF files in a subfolder of the repo (Q4_K_M/model.gguf) go to the top.
            for gguf in list(partial.glob("*/**/*.gguf")):
                if ".cache" not in gguf.parts:
                    gguf.rename(partial / gguf.name)
            if model_format(partial) is None:
                raise RuntimeError(f"No model files in {job.repo} matched the download filter")
            (partial / SOURCE_FILE).write_text(json.dumps(source))
            partial.rename(target)
            job.state = "done"
            log.info("Downloaded %s", job.model_id)
        except Exception as error:  # surfaced to the UI as text
            log.exception("Download of %s failed", job.model_id)
            job.state, job.error = "error", download_error(job.repo, error)

    def can_chat(self, model_id: str) -> bool:
        """An installed model, or a saved teacher the chat can talk to."""
        if model_id.startswith(TEACHER_PREFIX) and self.teachers is not None:
            from engine.teachers.store import TeacherNotFound

            try:
                self.teachers.by_name(model_id.removeprefix(TEACHER_PREFIX))
                return True
            except TeacherNotFound:
                return False
        return model_id in self.installed()

    def chat_models(self) -> list[str]:
        """Model ids the chat answers with: installed models, then saved teachers."""
        names = [t.name for t in self.teachers.saved()] if self.teachers is not None else []
        return self.installed() + [TEACHER_PREFIX + n for n in dict.fromkeys(names)]

    def runtime(self, model_id: str) -> Runtime:
        """Return a runtime for the model, unloading any other model first. A teacher answers
        through its API and leaves the loaded model alone."""
        if model_id.startswith(TEACHER_PREFIX) and self.teachers is not None:
            teacher = self.teachers.chat_runtime(model_id.removeprefix(TEACHER_PREFIX))
            return ConfiguredRuntime(teacher, lambda: {})
        if model_id == ECHO_MODEL and self.enable_echo:
            from engine.inference.echo import EchoRuntime

            return ConfiguredRuntime(EchoRuntime(), lambda: self.settings.get(ECHO_MODEL))
        path = self.models_dir / model_id
        fmt = model_format(path)
        if fmt is None:
            raise ModelNotFound(model_id)
        with self._lock:
            if self._loaded and self._loaded[0] == model_id:
                return self._loaded[1]
            self._loaded = None  # release memory of the previous model before loading
            options = {**self.runtime_options, **load_options(self.settings.get(model_id))}
            if fmt == "gguf":
                from engine.inference.llamacpp_runtime import LlamaCppRuntime

                runtime: Runtime = LlamaCppRuntime(path, **options)
            else:
                from engine.inference.transformers_runtime import TransformersRuntime

                # Context size, GPU layers and batch size are llama.cpp settings.
                options = {k: v for k, v in options.items() if k in TRANSFORMERS_OPTIONS}
                if fmt == "adapter":
                    base = self.models_dir / adapter_base(path)
                    if model_format(base) != "hf":
                        raise ModelNotFound(f"{model_id} needs its base model {base.name}")
                    runtime = TransformersRuntime(base, adapter=path, **options)
                else:
                    runtime = TransformersRuntime(path, **options)
            configured = ConfiguredRuntime(runtime, lambda: self.settings.get(model_id))
            self._loaded = (model_id, configured)
            return configured

    def exists(self, model_id: str) -> bool:
        """Whether a folder of this name holds a model, whether or not it can answer."""
        return model_format(self.models_dir / model_id) is not None

    def loaded(self) -> str | None:
        return self._loaded[0] if self._loaded else None

    def model_settings(self, model_id: str) -> dict:
        self._check_installed(model_id)
        return self.settings.get(model_id)

    def set_model_settings(self, model_id: str, settings: ModelSettings) -> dict:
        self._check_installed(model_id)
        before = load_options(self.settings.get(model_id))
        values = self.settings.set(model_id, settings)
        if load_options(values) != before:
            # Loading settings apply when the model is loaded next.
            with self._lock:
                if self._loaded and self._loaded[0] == model_id:
                    self._loaded = None
        return values

    def _check_installed(self, model_id: str) -> None:
        if model_id.startswith(".") or "/" in model_id or model_id not in self.installed():
            raise ModelNotFound(model_id)

    def details(self) -> list[dict]:
        """Installed models with their format, size on disk, where they came from, whether
        they are loaded, their settings and the fine-tuned models that need them."""
        adapters: dict[str, list[str]] = {}
        for p in self._folders():
            if model_format(p) == "adapter":
                adapters.setdefault(adapter_base(p), []).append(p.name)
        loaded = self.loaded()
        models = []
        for model_id in self.installed():
            if model_id == ECHO_MODEL:
                continue
            path = self.models_dir / model_id
            source = path / SOURCE_FILE
            models.append(
                {
                    "id": model_id,
                    "format": model_format(path),
                    "size": folder_size(path),
                    "source": json.loads(source.read_text()) if source.is_file() else None,
                    "loaded": model_id == loaded,
                    "settings": self.settings.get(model_id),
                    "used_by": adapters.get(model_id, []),
                }
            )
        return models

    def trainable(self) -> list[str]:
        """Installed models that can be fine-tuned: PyTorch (Hugging Face) weights."""
        return [p.name for p in self._folders() if model_format(p) == "hf"]

    def trained(self) -> list[dict]:
        """Fine-tuned models with what they were trained from: LoRA adapters ("format":
        "adapter", they need their base model) and their GGUF exports ("format": "gguf")."""
        models = []
        for path in self._folders():
            fmt = model_format(path)
            if fmt in ("adapter", "gguf") and (path / "tinylm.json").is_file():
                info = json.loads((path / "tinylm.json").read_text())
                installed = fmt == "gguf" or model_format(self.models_dir / info["base"]) == "hf"
                models.append({"id": path.name, "format": fmt, **info, "base_installed": installed})
        return models

    def delete(self, model_id: str) -> None:
        path = self.models_dir / model_id
        if model_id.startswith(".") or "/" in model_id or model_format(path) is None:
            raise ModelNotFound(model_id)
        with self._lock:
            if self._loaded and self._loaded[0] == model_id:
                self._loaded = None
            shutil.rmtree(path)
        self.settings.forget(model_id)
        self.downloads.pop(model_id, None)


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


def folder_size(path: Path) -> int:
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError:  # a file moved while counting, e.g. a download finishing
        return 0


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
