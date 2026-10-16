"""Moving a model between engines: a model trained on a RunPod pod to this computer, or one
from this computer to a pod.

A model goes out as an uncompressed tar of its folder (weights do not compress), streamed as it
is read and with its exact size known up front, plus its settings in tinylm-settings.json. It
comes in as such a tar or as a single .gguf file, uploaded in parts: RunPod's proxy refuses
large request bodies and requests that run longer than 100 seconds. Unpacking runs in the
background for the same reason.
"""

import json
import logging
import re
import shutil
import tarfile
import threading
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from engine.models import hub
from engine.models.manager import ModelManager, adapter_base, model_format
from engine.models.settings import ModelSettings

log = logging.getLogger(__name__)

SETTINGS_FILE = "tinylm-settings.json"
BLOCK = 512
PIECE = 1024 * 1024
MODEL_ID = re.compile(r"^[\w.-]{1,128}$")


class UploadError(ValueError):
    pass


def files(folder: Path) -> list[Path]:
    """The files of a model folder that go into its archive: not Hugging Face's .cache."""
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and ".cache" not in p.relative_to(folder).parts
    )


def _header(name: str, size: int) -> bytes:
    info = tarfile.TarInfo(name)
    info.size, info.mode = size, 0o644
    return info.tobuf(format=tarfile.PAX_FORMAT)


def _padding(size: int) -> int:
    return -size % BLOCK


def archive(models: ModelManager, model_id: str) -> tuple[int, Iterator[bytes]]:
    """The model's folder as a tar: its size in bytes and its content, read as it streams."""
    folder = models.models_dir / model_id
    settings = json.dumps(models.settings.get(model_id), indent=2).encode()
    members = [(f"{model_id}/{p.relative_to(folder).as_posix()}", p) for p in files(folder)]
    size = sum(
        len(_header(name, p.stat().st_size)) + p.stat().st_size + _padding(p.stat().st_size)
        for name, p in members
    )
    settings_name = f"{model_id}/{SETTINGS_FILE}"
    size += len(_header(settings_name, len(settings))) + len(settings) + _padding(len(settings))
    size += 2 * BLOCK  # the end of the archive

    def stream() -> Iterator[bytes]:
        for name, path in members:
            file_size = path.stat().st_size
            yield _header(name, file_size)
            with path.open("rb") as f:
                while piece := f.read(PIECE):
                    yield piece
            yield b"\0" * _padding(file_size)
        yield _header(settings_name, len(settings)) + settings + b"\0" * _padding(len(settings))
        yield b"\0" * (2 * BLOCK)

    return size, stream()


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Upload:
    name: str  # the file's name: <model>.tar from an export, or any .gguf file
    size: int
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: Literal["receiving", "installing", "done", "error"] = "receiving"
    received: int = 0
    model: str | None = None  # the installed model's id
    # For a fine-tuned model (a LoRA adapter): its base model, when this engine does not have it.
    missing_base: str | None = None
    error: str | None = None
    started_at: str = field(default_factory=now)

    def to_dict(self) -> dict:
        return asdict(self)


class ModelUploads:
    def __init__(
        self,
        models: ModelManager,
        folder: Path,
        collections: Callable[[], set[str]] = set,
        on_installed: Callable[[str], None] = lambda model_id: None,
    ):
        self.models = models
        self.folder = folder
        self.collections = collections  # names of this engine's collections
        # Called with the model's id, e.g. to import the collection a GGUF export carries.
        self.on_installed = on_installed
        self.uploads: dict[str, Upload] = {}
        self._lock = threading.Lock()

    def _path(self, upload: Upload) -> Path:
        return self.folder / f"{upload.id}.part"

    def start(self, name: str, size: int) -> Upload:
        if not name.endswith((".tar", ".gguf")):
            raise UploadError("Use a model archive (.tar, saved by Save to file) or a .gguf file")
        if size <= 0:
            raise UploadError("The file is empty")
        self.folder.mkdir(parents=True, exist_ok=True)
        upload = Upload(Path(name).name, size)
        self._path(upload).write_bytes(b"")
        self.uploads[upload.id] = upload
        log.info("Receiving model file %s (%d bytes)", upload.name, size)
        return upload

    def get(self, upload_id: str) -> Upload:
        upload = self.uploads.get(upload_id)
        if upload is None:
            raise KeyError(upload_id)
        return upload

    def append(self, upload: Upload, offset: int, pieces: Iterator[bytes]) -> Upload:
        """Adds a part at offset, which must be where the file ends: a resent part is refused."""
        with self._lock:
            if upload.state != "receiving":
                raise UploadError(f"The upload is {upload.state}")
            if offset != upload.received:
                raise UploadError(f"Expected the part at {upload.received}, got {offset}")
            with self._path(upload).open("ab") as f:
                for piece in pieces:
                    if upload.received + len(piece) > upload.size:
                        raise UploadError("More data than the file's size")
                    f.write(piece)
                    upload.received += len(piece)
        return upload

    def finish(self, upload: Upload) -> Upload:
        if upload.received != upload.size:
            raise UploadError(f"Received {upload.received} of {upload.size} bytes")
        upload.state = "installing"
        threading.Thread(target=self._install, args=(upload,), daemon=True).start()
        return upload

    def _install(self, upload: Upload) -> None:
        path = self._path(upload)
        partial: Path | None = None
        try:
            if upload.name.endswith(".gguf"):
                model_id = hub.model_id("upload", "gguf", upload.name)
                partial = self._partial(model_id)
                shutil.move(path, partial / upload.name)
                settings = {}
            else:
                model_id, settings = self._unpack(path)
                partial = self.models.models_dir / f".{model_id}.partial"
            if model_format(partial) is None:
                raise UploadError("The file holds no model")
            partial.rename(self.models.models_dir / model_id)
            partial = None
            if settings.get("collection") not in self.collections():
                settings.pop("collection", None)  # a collection this engine does not have
            if settings:
                self.models.settings.set(model_id, ModelSettings(**settings))
            folder = self.models.models_dir / model_id
            if model_format(folder) == "adapter":
                base = adapter_base(folder)
                if model_format(self.models.models_dir / base) != "hf":
                    upload.missing_base = base
            upload.model, upload.state = model_id, "done"
            log.info("Installed model %s from %s", model_id, upload.name)
            self.on_installed(model_id)
        except Exception as e:
            log.exception("Installing the model of %s failed", upload.name)
            upload.state, upload.error = "error", str(e)
            if partial is not None:
                shutil.rmtree(partial, ignore_errors=True)
        finally:
            path.unlink(missing_ok=True)

    def _partial(self, model_id: str) -> Path:
        if (self.models.models_dir / model_id).exists():
            raise UploadError(f"A model named {model_id} is already installed; delete it first")
        partial = self.models.models_dir / f".{model_id}.partial"
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        return partial

    def _unpack(self, path: Path) -> tuple[str, dict]:
        """Unpacks a model archive into a hidden folder; returns the model id and settings."""
        with tarfile.open(path) as tar:
            members = tar.getmembers()
            tops = {m.name.split("/", 1)[0] for m in members}
            if len(tops) != 1:
                raise UploadError("Not a model archive: expected one folder")
            model_id = model_name(tops.pop())
            partial = self._partial(model_id)
            settings = {}
            for m in members:
                if m.isdir():
                    continue
                if not m.isfile():
                    raise UploadError(f"Not a model archive: {m.name} is not a file")
                relative = m.name.split("/", 1)[1] if "/" in m.name else ""
                if relative == SETTINGS_FILE:
                    settings = json.loads(tar.extractfile(m).read() or b"{}")
                    continue
                target = (partial / relative).resolve()
                if not relative or not target.is_relative_to(partial.resolve()):
                    raise UploadError(f"Not a model archive: {m.name} is outside its folder")
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out, PIECE)
        return model_id, settings


def model_name(name: str) -> str:
    if not MODEL_ID.match(name) or name.startswith("."):
        raise UploadError(f"Bad model name {name!r}: use letters, digits, '.', '_' or '-'")
    return name
