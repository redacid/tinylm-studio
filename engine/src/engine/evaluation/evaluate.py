"""Evaluation: how well models answer the held-out examples of a dataset.

Every model answers the same questions (greedy decoding, so runs are repeatable). Each answer is
compared with the dataset's reference answer in two ways:
- similarity: chrF, a character n-gram F-score from 0 to 100. Cheap and language-independent
  (works for Ukrainian and Russian word forms), but it rewards wording, not correctness.
- judge (optional): a teacher model scores every answer from 1 to 5 against the reference, with
  a short reason. All models' answers to one question are judged together under shuffled labels,
  so the judge compares them on the same scale and does not know which model wrote which.

Runs are saved as JSON files after every answer, so a report survives restarts.
"""

import json
import logging
import random
import re
import threading
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from engine.chat import ChatRequest, Message, collect
from engine.datasets.generate import parse_json
from engine.datasets.store import DatasetStore
from engine.rag.store import now
from engine.teachers.providers import TeacherError

log = logging.getLogger(__name__)

State = Literal["running", "done", "cancelled", "error", "interrupted"]
MAX_MODELS = 4
JUDGE_SYSTEM = (
    "You grade answers of assistant models against a reference answer. Be strict and fair. "
    "Reply with JSON only."
)


class EvaluationError(ValueError):
    pass


class EvaluationNotFound(Exception):
    pass


@dataclass
class EvaluationOptions:
    dataset: str
    models: list[str]
    judge: str | None = None  # a teacher "ref"; None scores by similarity only
    limit: int = 50  # held-out examples to use, at most
    max_tokens: int = 512


@dataclass
class Evaluation:
    options: EvaluationOptions
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: State = "running"
    created_at: str = field(default_factory=now)
    finished_at: str | None = None
    error: str | None = None
    phase: Literal["answering", "judging", ""] = "answering"
    current: str | None = None  # the model answering
    done: int = 0  # answers written plus examples judged
    total: int = 0
    # One per example: messages (the prompt), reference and answers by model id, each with
    # text, seconds, similarity and, when judged, score and reason.
    examples: list[dict] = field(default_factory=list)
    summary: dict[str, dict] = field(default_factory=dict)

    def to_dict(self, examples: bool = True) -> dict:
        data = asdict(self)
        if not examples:
            data.pop("examples")
        return data


def chrf(candidate: str, reference: str, max_n: int = 6, beta: float = 2.0) -> float:
    """chrF (Popović, 2015): F-score of character n-grams (n = 1..6, spaces removed), with recall
    weighted twice as much as precision. 0 to 100."""
    hyp, ref = re.sub(r"\s+", "", candidate.casefold()), re.sub(r"\s+", "", reference.casefold())
    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        h = Counter(hyp[i : i + n] for i in range(len(hyp) - n + 1))
        r = Counter(ref[i : i + n] for i in range(len(ref) - n + 1))
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precisions.append(overlap / sum(h.values()))
        recalls.append(overlap / sum(r.values()))
    if not precisions:
        return 0.0
    p, r = sum(precisions) / len(precisions), sum(recalls) / len(recalls)
    if p + r == 0:
        return 0.0
    return round(100 * (1 + beta**2) * p * r / (beta**2 * p + r), 1)


def judge_prompt(messages: list[dict], reference: str, answers: list[tuple[str, str]]) -> str:
    """answers: (label, text) pairs in the order shown to the judge."""
    lines = ["Conversation:"]
    for m in messages:
        lines.append(f"[{m['role']}] {m['content']}")
    lines += ["", "Reference answer:", reference, "", "Answers to grade:"]
    for label, text in answers:
        lines += [f"--- {label} ---", text.strip() or "(empty)"]
    lines += [
        "",
        "Score each answer from 1 to 5:",
        "5 - correct and complete, as good as the reference;",
        "4 - correct, with small omissions or style differences;",
        "3 - partly correct or missing important parts;",
        "2 - mostly wrong, or in a different language than the question;",
        "1 - wrong, invented facts, or no answer.",
        "Facts that contradict the reference or are not in it count as invented. Judge the "
        "content and the language; the wording may differ from the reference.",
        'Reply with JSON: {"scores": {"A": {"score": 4, "reason": "one short sentence"}}}',
    ]
    return "\n".join(lines)


def summarize(evaluation: Evaluation) -> dict[str, dict]:
    summary = {}
    for model in evaluation.options.models:
        answers = [e["answers"][model] for e in evaluation.examples if model in e["answers"]]
        scores = [a["score"] for a in answers if a.get("score") is not None]
        summary[model] = {
            "answers": len(answers),
            "similarity": mean([a["similarity"] for a in answers]),
            "judged": len(scores),
            "score": mean(scores, 2),
            "seconds": mean([a["seconds"] for a in answers], 2),
            "chars": mean([len(a["text"]) for a in answers], 0),
        }
    return summary


def mean(values: list[float], digits: int = 1) -> float | None:
    return round(sum(values) / len(values), digits) if values else None


class EvaluationManager:
    """Runs one evaluation at a time in a background thread; reports are JSON files."""

    def __init__(self, folder: Path, datasets: DatasetStore, models, teachers):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.datasets = datasets
        self.models = models
        self.teachers = teachers
        self.cancel = threading.Event()
        self._running: Evaluation | None = None
        self._lock = threading.Lock()
        for evaluation in self.all():
            if evaluation.state == "running":
                evaluation.state, evaluation.finished_at = "interrupted", now()
                self._save(evaluation)

    # Reading

    def all(self) -> list[Evaluation]:
        runs = [self._load(p) for p in self.folder.glob("*.json")]
        return sorted((r for r in runs if r), key=lambda r: r.created_at)

    def get(self, evaluation_id: str) -> Evaluation:
        """From its file: the running thread saves after every step, so reading the file never
        sees a half-updated report."""
        path = self.folder / f"{evaluation_id}.json"
        evaluation = self._load(path) if re.fullmatch(r"[0-9a-f]{12}", evaluation_id) else None
        if evaluation is None:
            raise EvaluationNotFound(evaluation_id)
        return evaluation

    def _load(self, path: Path) -> Evaluation | None:
        try:
            if not path.exists():
                return None
            data = json.loads(path.read_text())
            data["options"] = EvaluationOptions(**data["options"])
            return Evaluation(**data)
        except (ValueError, TypeError, KeyError):
            log.warning("Skipping unreadable evaluation %s", path)
            return None

    def _save(self, evaluation: Evaluation) -> None:
        path = self.folder / f"{evaluation.id}.json"
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(evaluation.to_dict(), ensure_ascii=False))
        partial.replace(path)

    def delete(self, evaluation_id: str) -> None:
        evaluation = self.get(evaluation_id)
        if evaluation.state == "running":
            raise EvaluationError("Stop the evaluation before deleting it.")
        (self.folder / f"{evaluation_id}.json").unlink()

    # Running

    def start(self, options: EvaluationOptions) -> Evaluation:
        if not options.models:
            raise EvaluationError("Choose at least one model.")
        if len(set(options.models)) > MAX_MODELS:
            raise EvaluationError(f"Compare at most {MAX_MODELS} models at a time.")
        options.models = list(dict.fromkeys(options.models))
        installed = set(self.models.installed())
        missing = [m for m in options.models if m not in installed]
        if missing:
            raise EvaluationError(f"Model {missing[0]} is not installed.")
        if options.judge:
            self.teachers.teacher(options.judge)  # TeacherNotFound for an unknown judge
        _, held_out = self.datasets.split(options.dataset)  # DatasetNotFound
        if not held_out:
            raise EvaluationError(
                "The dataset has no held-out examples; set its evaluation share above 0%."
            )
        with self._lock:
            if self._running:
                raise EvaluationError("Another evaluation is running; one runs at a time.")
            examples = held_out[: options.limit]
            evaluation = Evaluation(options)
            evaluation.examples = [
                {"messages": m[:-1], "reference": m[-1]["content"], "answers": {}} for m in examples
            ]
            evaluation.total = len(examples) * len(options.models) + (
                len(examples) if options.judge else 0
            )
            self._running = evaluation
            self.cancel.clear()
            self._save(evaluation)
        log.info(
            "Evaluation %s of %s on %d examples of %s%s",
            evaluation.id,
            ", ".join(options.models),
            len(examples),
            options.dataset,
            f", judged by {options.judge}" if options.judge else "",
        )
        threading.Thread(target=self._run, args=(evaluation,), daemon=True).start()
        return evaluation

    def stop(self, evaluation_id: str) -> Evaluation:
        evaluation = self.get(evaluation_id)
        running = self._running
        if running and running.id == evaluation_id:
            self.cancel.set()
        return evaluation

    def _run(self, evaluation: Evaluation) -> None:
        try:
            self._answer(evaluation)
            if evaluation.options.judge and not self.cancel.is_set():
                self._judge(evaluation)
            evaluation.state = "cancelled" if self.cancel.is_set() else "done"
        except Exception as e:  # shown in the UI
            log.exception("Evaluation %s failed", evaluation.id)
            evaluation.state, evaluation.error = "error", str(e)
        finally:
            evaluation.current, evaluation.phase = None, ""
            evaluation.finished_at = now()
            evaluation.summary = summarize(evaluation)
            self._save(evaluation)
            self._running = None
            log.info("Evaluation %s %s", evaluation.id, evaluation.state)

    def _answer(self, evaluation: Evaluation) -> None:
        options = evaluation.options
        # Model by model, so each is loaded once.
        for model in options.models:
            evaluation.current = model
            self._save(evaluation)
            runtime = self.models.runtime(model)
            for example in evaluation.examples:
                if self.cancel.is_set():
                    return
                messages = [Message(m["role"], m["content"]) for m in example["messages"]]
                started = time.monotonic()
                text = collect(
                    runtime.stream(ChatRequest(model, messages, options.max_tokens, 0.0))
                ).text.strip()
                example["answers"][model] = {
                    "text": text,
                    "seconds": round(time.monotonic() - started, 2),
                    "similarity": chrf(text, example["reference"]),
                    "score": None,
                    "reason": None,
                }
                evaluation.done += 1
                evaluation.summary = summarize(evaluation)
                self._save(evaluation)

    def _judge(self, evaluation: Evaluation) -> None:
        evaluation.phase, evaluation.current = "judging", None
        self._save(evaluation)
        judge = self.teachers.teacher(evaluation.options.judge)
        failures = 0
        shuffle = random.Random(evaluation.id)
        for example in evaluation.examples:
            if self.cancel.is_set():
                return
            models = list(example["answers"])
            shuffle.shuffle(models)
            labels = {chr(ord("A") + i): model for i, model in enumerate(models)}
            prompt = judge_prompt(
                example["messages"],
                example["reference"],
                [(label, example["answers"][m]["text"]) for label, m in labels.items()],
            )
            try:
                reply = judge.complete(JUDGE_SYSTEM, prompt, max_tokens=1024, temperature=0.0)
                scores = parse_json(reply).get("scores", {})
                for label, model in labels.items():
                    item = scores.get(label) or {}
                    score = item.get("score") if isinstance(item, dict) else item
                    if isinstance(score, int | float) and 1 <= score <= 5:
                        example["answers"][model]["score"] = score
                        example["answers"][model]["reason"] = (
                            item.get("reason") if isinstance(item, dict) else None
                        )
                failures = 0
            except (TeacherError, ValueError, AttributeError) as e:
                failures += 1
                example["judge_error"] = str(e)
                if isinstance(e, TeacherError) and failures >= 3:
                    raise
            evaluation.done += 1
            evaluation.summary = summarize(evaluation)
            self._save(evaluation)
