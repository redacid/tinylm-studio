"""GGUF exports: one at a time, each in its own worker process (engine.export.worker).

An export lives in <workspace>/exports/<id>/: job.json (settings and state), events.jsonl
(progress from the worker) and worker.log. Exports that were going when the engine stopped are
marked interrupted; their temporary files are removed.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from engine.export.worker import QUANTIZATIONS
from engine.models.bundles import BUNDLE_FILE, write_bundle
from engine.models.manager import model_format
from engine.rag.store import CollectionNotFound, KnowledgeStore, now
from engine.training.jobs import MODEL_NAME
from engine.training.worker import read_jsonl

State = Literal["running", "done", "error", "interrupted"]
STEPS = ("tools", "merge", "convert", "quantize", "save")


class ExportError(ValueError):
    pass


class ExportNotFound(Exception):
    pass


@dataclass
class Export:
    id: str
    model: str  # the fine-tuned model (adapter) exported
    name: str  # the new GGUF model
    quantization: str
    state: State
    created_at: str
    finished_at: str | None = None
    error: str | None = None
    pid: int | None = None
    step: str | None = None
    notes: list[str] = field(default_factory=list)  # downloads and warnings
    size: int | None = None  # bytes of the GGUF file

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("pid")
        data["steps"] = list(STEPS)
        return data


class ExportManager:
    def __init__(
        self,
        folder: Path,
        models_dir: Path,
        tools_dir: Path,
        llama_cpp_dir: Path | None = None,
        threads: int | None = None,
        knowledge: KnowledgeStore | None = None,
        on_done=None,
    ):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.models_dir = models_dir
        self.tools_dir = tools_dir
        self.llama_cpp_dir = llama_cpp_dir
        self.threads = threads
        # For exports that take a collection along, and what to do with a finished model.
        self.knowledge = knowledge
        self.on_done = on_done
        self._lock = threading.Lock()
        for export in self.exports():
            if export.state == "running" and not self._alive(export):
                self._finish(export.id, "interrupted", "The engine stopped during the export.")

    # Reading

    def exports(self) -> list[Export]:
        found = [self._load(p.parent.name) for p in self.folder.glob("*/job.json")]
        return sorted(found, key=lambda e: e.created_at)

    def get(self, export_id: str) -> Export:
        if not (self.folder / export_id / "job.json").is_file() or "/" in export_id:
            raise ExportNotFound(export_id)
        return self._load(export_id)

    def _load(self, export_id: str) -> Export:
        folder = self.folder / export_id
        data = json.loads((folder / "job.json").read_text())
        settings = data.pop("settings")
        export = Export(
            **data,
            model=settings["model"],
            name=settings["name"],
            quantization=settings["quantization"],
        )
        for event in read_jsonl(folder / "events.jsonl"):
            if event["type"] == "step":
                export.step = event["step"]
            elif event["type"] in ("note", "warning"):
                export.notes.append(event["message"])
            elif event["type"] == "done":
                export.size = event.get("size")
            elif event["type"] == "error" and export.error is None:
                export.error = event["message"]
        return export

    def _save(self, export_id: str, **changes) -> None:
        path = self.folder / export_id / "job.json"
        data = json.loads(path.read_text())
        data.update(changes)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)

    # Running

    def start(
        self,
        model: str,
        quantization: str = "Q8_0",
        name: str | None = None,
        collection: str | None = None,
    ) -> Export:
        if quantization not in QUANTIZATIONS:
            raise ExportError(f"Use one of {', '.join(QUANTIZATIONS)}.")
        if model_format(self.models_dir / model) != "adapter" or "/" in model:
            raise ExportError(f"{model} is not a fine-tuned model; only those can be exported.")
        info = json.loads((self.models_dir / model / "tinylm.json").read_text())
        if model_format(self.models_dir / info["base"]) != "hf":
            raise ExportError(f"The base model {info['base']} is not installed.")
        name = (name or f"{model}-{quantization.lower()}").strip()
        if not MODEL_NAME.match(name) or name.startswith("."):
            raise ExportError("Use up to 64 letters, digits, '.', '_' or '-' for the name.")
        if (self.models_dir / name).exists():
            raise ExportError(f"A model named {name} already exists.")
        with self._lock:
            if any(e.state == "running" for e in self.exports()):
                raise ExportError("Another export is running; one runs at a time.")
            export_id = uuid.uuid4().hex[:12]
            folder = self.folder / export_id
            folder.mkdir()
            if collection:
                try:
                    write_bundle(self.knowledge, collection, folder / BUNDLE_FILE)
                except CollectionNotFound:
                    shutil.rmtree(folder)
                    raise ExportError(f"The collection {collection} does not exist.") from None
            settings = {
                "model": model,
                "name": name,
                "quantization": quantization,
                "collection": collection or None,
                "models_dir": str(self.models_dir),
                "tools_dir": str(self.tools_dir),
                "llama_cpp_dir": str(self.llama_cpp_dir) if self.llama_cpp_dir else None,
            }
            if self.threads:
                settings["threads"] = self.threads
            job = {"id": export_id, "settings": settings, "state": "running", "created_at": now()}
            (folder / "job.json").write_text(json.dumps(job, indent=2))
            with (folder / "worker.log").open("w") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "engine.export.worker", str(folder)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
            self._save(export_id, pid=process.pid)
        threading.Thread(target=self._watch, args=(export_id, process), daemon=True).start()
        return self.get(export_id)

    def _watch(self, export_id: str, process: subprocess.Popen) -> None:
        code = process.wait()
        export = self._load(export_id)
        if code == 0 and (self.models_dir / export.name).exists():
            self._finish(export_id, "done")
            if self.on_done:
                self.on_done(export.name)
        else:
            log_tail = (self.folder / export_id / "worker.log").read_text(errors="replace")[-2000:]
            self._finish(export_id, "error", export.error or log_tail or f"exit code {code}")

    def _finish(self, export_id: str, state: State, error: str | None = None) -> None:
        shutil.rmtree(self.folder / export_id / "tmp", ignore_errors=True)
        (self.folder / export_id / BUNDLE_FILE).unlink(missing_ok=True)
        export = self._load(export_id)
        shutil.rmtree(self.models_dir / f".{export.name}.partial", ignore_errors=True)
        self._save(export_id, state=state, error=error, finished_at=now())

    def _alive(self, export: Export) -> bool:
        if not export.pid:
            return False
        try:
            os.kill(export.pid, 0)
        except OSError:
            return False
        cmdline = Path(f"/proc/{export.pid}/cmdline")
        return not cmdline.exists() or export.id.encode() in cmdline.read_bytes()
