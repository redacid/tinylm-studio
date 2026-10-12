"""Training jobs: one LoRA run at a time, each in its own worker process.

A job lives in <workspace>/training/<id>/: job.json (settings and state), the exported examples,
events.jsonl (progress from the worker), checkpoints/ and worker.log. Everything is read back
from these files, so jobs survive an engine restart; a run that was cut off can be resumed from
its last checkpoint.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from engine.datasets.store import DatasetStore
from engine.models.manager import model_format
from engine.rag.store import now
from engine.training.worker import last_checkpoint

MODEL_NAME = re.compile(r"^[\w.-]{1,64}$")
State = Literal["running", "stopping", "stopped", "done", "error", "interrupted"]
# Seconds a worker gets to save a checkpoint after being asked to stop, before it is killed.
STOP_GRACE = 300


class TrainingError(ValueError):
    pass


class JobNotFound(Exception):
    pass


class Busy(Exception):
    pass


@dataclass
class TrainingSettings:
    """What to train: the base model, the dataset and the new model's name, plus LoRA and
    optimizer settings. Defaults suit a ≤1B model on a CPU."""

    name: str
    base: str
    dataset: str
    epochs: float = 3
    learning_rate: float = 2e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_length: int = 1024
    # One example at a time, four per optimizer step: no padding, and a 0.6B model needs ~6 GB
    # of RAM instead of ~14 GB with batches of four (the output layer's logits dominate).
    batch_size: int = 1
    gradient_accumulation: int = 4
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    threads: int | None = None
    # Steps between checkpoints and evaluations; by default about ten per run.
    save_steps: int | None = None
    seed: int = 42


@dataclass
class Job:
    id: str
    settings: TrainingSettings
    state: State
    created_at: str
    finished_at: str | None = None
    error: str | None = None
    pid: int | None = None
    # From the worker's events:
    step: int = 0
    total_steps: int | None = None
    epoch: float = 0
    loss: float | None = None
    eval_loss: float | None = None
    device: str | None = None
    train_examples: int | None = None
    eval_examples: int | None = None
    dropped: int = 0
    seconds_per_step: float | None = None
    checkpoint_step: int | None = None
    history: list[dict] = field(default_factory=list)

    @property
    def resumable(self) -> bool:
        return (
            self.state in ("stopped", "error", "interrupted") and self.checkpoint_step is not None
        )

    def to_dict(self, history: bool = True) -> dict:
        data = asdict(self)
        data["resumable"] = self.resumable
        remaining = (self.total_steps or 0) - self.step
        running = self.state == "running" and self.seconds_per_step
        data["eta_seconds"] = round(remaining * self.seconds_per_step) if running else None
        if not history:
            data.pop("history")
        return data


class TrainingManager:
    def __init__(self, folder: Path, models_dir: Path, datasets: DatasetStore):
        self.folder = folder
        self.models_dir = models_dir
        self.datasets = datasets
        self.folder.mkdir(parents=True, exist_ok=True)
        self._processes: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        # Runs that were going when the engine stopped have lost their worker.
        for job in self.jobs():
            if job.state in ("running", "stopping") and not self._alive(job):
                self._save(job.id, state="interrupted", finished_at=now())

    # Reading jobs

    def jobs(self) -> list[Job]:
        jobs = [self._load(p.parent.name) for p in self.folder.glob("*/job.json")]
        return sorted(jobs, key=lambda j: j.created_at)

    def job(self, job_id: str) -> Job:
        if (
            not re.fullmatch(r"[0-9a-f]{12}", job_id)
            or not (self._dir(job_id) / "job.json").exists()
        ):
            raise JobNotFound(job_id)
        return self._load(job_id)

    def _dir(self, job_id: str) -> Path:
        return self.folder / job_id

    def _load(self, job_id: str) -> Job:
        folder = self._dir(job_id)
        data = json.loads((folder / "job.json").read_text())
        job = Job(**{**data, "settings": TrainingSettings(**data["settings"])})
        apply_events(job, folder / "events.jsonl")
        return job

    def _save(self, job_id: str, **changes) -> None:
        path = self._dir(job_id) / "job.json"
        data = json.loads(path.read_text())
        data.update(changes)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)

    def events(self, job_id: str, start: int = 0) -> tuple[list[dict], int]:
        """Worker events after byte offset start, and the offset to continue from."""
        path = self._dir(job_id) / "events.jsonl"
        if not path.exists():
            return [], start
        with path.open("rb") as f:
            f.seek(start)
            data = f.read()
        complete = data[: data.rfind(b"\n") + 1]
        lines = complete.decode("utf-8").splitlines()
        return [json.loads(line) for line in lines if line], start + len(complete)

    def log(self, job_id: str, lines: int = 40) -> str:
        path = self._dir(job_id) / "worker.log"
        if not path.exists():
            return ""
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])

    # Running jobs

    def start(self, settings: TrainingSettings) -> Job:
        with self._lock:
            self._require_idle()
            self._check(settings)
            train, held_out = self.datasets.split(settings.dataset)
            if not train:
                raise TrainingError(f"Dataset '{settings.dataset}' has no training examples.")
            job_id = uuid.uuid4().hex[:12]
            folder = self._dir(job_id)
            folder.mkdir()
            for name, examples in (("train", train), ("eval", held_out)):
                with (folder / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                    for messages in examples:
                        f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            job = Job(job_id, settings, "running", now())
            (folder / "job.json").write_text(json.dumps(asdict(job), indent=2))
            self._launch(job, resume=False)
            return self._load(job_id)

    def resume(self, job_id: str) -> Job:
        with self._lock:
            job = self.job(job_id)
            if not job.resumable:
                raise TrainingError("This run has no checkpoint to continue from.")
            self._require_idle()
            if (self.models_dir / job.settings.name).exists():
                raise TrainingError(f"A model named {job.settings.name} already exists.")
            (self._dir(job_id) / "stop").unlink(missing_ok=True)
            self._save(job_id, state="running", error=None, finished_at=None)
            self._launch(job, resume=True)
            return self._load(job_id)

    def stop(self, job_id: str) -> Job:
        """Asks the worker to save a checkpoint and stop at the next step."""
        job = self.job(job_id)
        if job.state != "running":
            return job
        (self._dir(job_id) / "stop").touch()
        self._save(job_id, state="stopping")
        process = self._processes.get(job_id)
        if process:
            threading.Thread(target=self._kill_later, args=(process,), daemon=True).start()
        return self._load(job_id)

    def delete(self, job_id: str) -> None:
        job = self.job(job_id)
        if job.state in ("running", "stopping"):
            raise TrainingError("Stop the run first.")
        shutil.rmtree(self._dir(job_id))

    def shutdown(self) -> None:
        for process in list(self._processes.values()):
            if process.poll() is None:
                process.terminate()

    def _require_idle(self) -> None:
        if any(j.state in ("running", "stopping") for j in self.jobs()):
            raise Busy("Another training run is in progress; one runs at a time.")

    def _check(self, s: TrainingSettings) -> None:
        if not MODEL_NAME.match(s.name) or s.name.startswith("."):
            raise TrainingError("Name the model with up to 64 letters, digits, '.', '_' or '-'.")
        if (self.models_dir / s.name).exists():
            raise TrainingError(f"A model named {s.name} already exists.")
        fmt = model_format(self.models_dir / s.base)
        if fmt is None:
            raise TrainingError(f"Base model {s.base} is not installed.")
        if fmt != "hf":
            raise TrainingError(
                f"{s.base} is a {fmt.upper()} model; fine-tuning needs PyTorch (Hugging Face) "
                "weights, such as qwen3-0.6b."
            )
        if not (0 < s.epochs <= 100 and 0 < s.learning_rate < 1 and 1 <= s.lora_rank <= 256):
            raise TrainingError("Epochs, learning rate or LoRA rank are out of range.")
        if not (32 <= s.max_length <= 32768 and 1 <= s.batch_size <= 256):
            raise TrainingError("Max length or batch size are out of range.")
        if s.gradient_accumulation < 1:
            raise TrainingError("Gradient accumulation must be at least 1.")

    def _launch(self, job: Job, resume: bool) -> None:
        folder = self._dir(job.id)
        config = {
            **asdict(job.settings),
            "job": job.id,
            "base_path": str(self.models_dir / job.settings.base),
            "output_path": str(self.models_dir / job.settings.name),
            "resume": resume,
        }
        (folder / "config.json").write_text(json.dumps(config, indent=2))
        log = (folder / "worker.log").open("ab")
        env = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
        process = subprocess.Popen(
            [sys.executable, "-m", "engine.training.worker", str(folder)],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
        )
        log.close()
        self._processes[job.id] = process
        self._save(job.id, pid=process.pid)
        threading.Thread(target=self._watch, args=(job.id, process), daemon=True).start()

    def _watch(self, job_id: str, process: subprocess.Popen) -> None:
        code = process.wait()
        self._processes.pop(job_id, None)
        job = self._load(job_id)
        events, _ = self.events(job_id)
        last = events[-1]["type"] if events else None
        if last == "done":
            state, error = "done", None
        elif last == "stopped" or job.state == "stopping":
            state, error = "stopped", None
        else:
            error = next((e["message"] for e in reversed(events) if e["type"] == "error"), None)
            state = "error"
            error = error or f"The training process ended unexpectedly (exit code {code})."
        self._save(job_id, state=state, error=error, finished_at=now(), pid=None)

    def _kill_later(self, process: subprocess.Popen) -> None:
        try:
            process.wait(STOP_GRACE)
        except subprocess.TimeoutExpired:
            process.send_signal(signal.SIGKILL)

    def _alive(self, job: Job) -> bool:
        if not job.pid:
            return False
        try:
            os.kill(job.pid, 0)
        except OSError:
            return False
        cmdline = Path(f"/proc/{job.pid}/cmdline")
        return not cmdline.exists() or job.id.encode() in cmdline.read_bytes()


def apply_events(job: Job, path: Path) -> None:
    """Fills a job's progress from the worker's events."""
    if not path.exists():
        return
    steps: dict[int, dict] = {}
    times: list[tuple[int, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        event = json.loads(line)
        kind = event["type"]
        if kind == "start":
            job.total_steps = event["total_steps"]
            job.device = event["device"]
            job.train_examples = event["train_examples"]
            job.eval_examples = event["eval_examples"]
            job.dropped = event["dropped"]
            times = []  # a resumed run: time it afresh
        elif kind == "step":
            job.step, job.epoch, job.loss = event["step"], event["epoch"], event["loss"]
            steps.setdefault(event["step"], {"step": event["step"]})["loss"] = event["loss"]
            times.append((event["step"], event["time"]))
        elif kind == "eval":
            job.eval_loss = event["eval_loss"]
            steps.setdefault(event["step"], {"step": event["step"]})["eval_loss"] = event[
                "eval_loss"
            ]
        elif kind == "checkpoint":
            job.checkpoint_step = event["step"]
    # A resumed run repeats the steps after its checkpoint; the later values win.
    job.history = [steps[s] for s in sorted(steps)]
    recent = times[-20:]
    if len(recent) >= 2 and recent[-1][0] > recent[0][0]:
        job.seconds_per_step = (recent[-1][1] - recent[0][1]) / (recent[-1][0] - recent[0][0])
    if job.checkpoint_step is not None and last_checkpoint(path.parent / "checkpoints") is None:
        job.checkpoint_step = None
