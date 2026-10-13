"""Generating training examples from a collection with a teacher model.

For each chunk of source text the teacher writes a few question/answer pairs in the wanted
languages and answer style (taken from the instructions and from examples already in the
dataset). Pairs are then filtered: malformed, too short or too long, in the wrong language, and,
when verification is on, not supported by the text (checked by the teacher in a second call).
The rest are added to the dataset, marked as generated, with the document they came from.

Each chunk is used once per dataset, so running the generator again continues with new chunks.
"""

import json
import logging
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import Literal

from engine.datasets.store import DatasetStore
from engine.rag.store import KnowledgeStore, now
from engine.teachers.providers import TeacherError

log = logging.getLogger(__name__)

LANGUAGE_NAMES = {"uk": "Ukrainian", "ru": "Russian", "en": "English"}
# Chunks shorter than this rarely hold enough facts for a good question.
MIN_CHUNK_CHARS = 300
# The job gives up after this many chunks in a row fail (a wrong key, a model that is gone).
MAX_FAILURES_IN_A_ROW = 3
GENERATED_SCHEMA = """
CREATE TABLE IF NOT EXISTS generated_chunks (
    dataset TEXT NOT NULL REFERENCES datasets(name) ON DELETE CASCADE ON UPDATE CASCADE,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    PRIMARY KEY (dataset, chunk_id)
);
"""

SYSTEM = (
    "You write training examples for a small assistant model. You get a passage of source text "
    "and write question and answer pairs based only on it. Reply with JSON only."
)
CHECK_SYSTEM = (
    "You check training examples against their source text. Be strict. Reply with JSON only."
)


@dataclass
class GenerationOptions:
    collection: str
    teacher: str  # "local:<model id>" or a saved teacher's id
    chunks: int = 20  # source chunks to use in this run
    per_chunk: int = 3  # question/answer pairs per chunk
    # Languages of the pairs (uk, ru, en); empty means the language of each source text.
    languages: list[str] = field(default_factory=list)
    # The wanted answer style, e.g. "A short tip, then numbered steps".
    instructions: str = ""
    # Added as the system message of every generated example.
    system_prompt: str = ""
    # Also write a question the text does not answer, with an honest "no information" answer.
    unanswerable: bool = True
    # Let the teacher check each pair against the text and drop unsupported ones.
    verify: bool = True
    min_answer_chars: int = 20
    max_answer_chars: int = 2000


@dataclass
class GenerationJob:
    dataset: str
    options: GenerationOptions
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: Literal["running", "done", "cancelled", "error"] = "running"
    chunks_total: int = 0
    chunks_done: int = 0
    generated: int = 0  # pairs the teacher wrote
    added: int = 0
    duplicates: int = 0
    rejected: dict[str, int] = field(default_factory=dict)  # reason -> count
    failed_chunks: int = 0
    current: str | None = None
    waiting: float | None = None  # seconds of a rate-limit pause in progress
    # Who answered the teacher's requests: [{"model", "provider", "requests"}], as the API
    # reported it (OpenRouter may route to another model or provider than the one asked for).
    served: list[dict] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: str = field(default_factory=now)
    cancelled: threading.Event = field(default_factory=threading.Event, repr=False)

    def note(self, line: str) -> None:
        self.log.append(line)
        del self.log[:-200]

    def count_served(self, served: dict) -> None:
        for row in self.served:
            if (row["model"], row["provider"]) == (served.get("model"), served.get("provider")):
                row["requests"] += 1
                return
        self.served.append(
            {"model": served.get("model"), "provider": served.get("provider"), "requests": 1}
        )

    def reject(self, reason: str, count: int = 1) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + count

    def to_dict(self) -> dict:
        data = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "cancelled"}
        data["options"] = asdict(self.options)
        data["rejected"] = dict(self.rejected)
        data["log"] = self.log[-50:]
        data["served"] = [dict(row) for row in self.served]
        return data


def detect_language(text: str) -> str | None:
    """uk, ru or en for clear cases; None when the text gives no hint."""
    cyrillic = len(re.findall(r"[а-яёіїєґ]", text, re.IGNORECASE))
    latin = len(re.findall(r"[a-z]", text, re.IGNORECASE))
    if latin > cyrillic * 2:
        return "en"
    if cyrillic <= latin:
        return None
    ukrainian = len(re.findall(r"[іїєґ]", text, re.IGNORECASE))
    russian = len(re.findall(r"[ыэъё]", text, re.IGNORECASE))
    if ukrainian > russian:
        return "uk"
    if russian > ukrainian:
        return "ru"
    return None


def parse_json(text: str):
    """The first JSON object or array in a reply, tolerating code fences and chatter around it."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    for opener in ("{", "["):
        start = text.find(opener)
        while start != -1:
            try:
                value, _ = json.JSONDecoder().raw_decode(text[start:])
                return value
            except json.JSONDecodeError:
                start = text.find(opener, start + 1)
    raise ValueError("no JSON in the reply")


def generation_prompt(
    options: GenerationOptions, document: str, text: str, samples: list[list[dict]]
) -> str:
    if options.languages:
        names = [LANGUAGE_NAMES.get(code, code) for code in options.languages]
        language = (
            f"Write each pair in {names[0]}."
            if len(names) == 1
            else f"Write each pair in one of these languages, spread evenly: {', '.join(names)}."
        )
    else:
        language = "Write the pairs in the language of the text."
    lines = [
        f"Write {options.per_chunk} question and answer pairs about the text below.",
        "",
        "Rules:",
        "- Questions are what a real user of the assistant would ask. Vary them: facts, advice, "
        "step-by-step how-to, comparisons when the text allows it.",
        "- Each answer uses only facts from the text and stands on its own: never mention "
        '"the text", "the article" or "the author".',
        f"- {language}",
    ]
    if options.unanswerable:
        lines.append(
            "- Add one more question on the same topic that the text does not answer. Its "
            "answer says briefly that there is no such information and invents nothing. Mark it "
            'with "unanswerable": true.'
        )
    if options.instructions.strip():
        lines += ["", "Answer style:", options.instructions.strip()]
    if samples:
        lines += ["", "Examples of the wanted style (do not copy their content):"]
        for messages in samples:
            question = next(m["content"] for m in messages if m["role"] == "user")
            lines += [f"Q: {question}", f"A: {messages[-1]['content']}", ""]
    lines += [
        "",
        'Reply with JSON: {"examples": [{"question": "...", "answer": "...", '
        '"unanswerable": false}]}',
        "",
        f"Text ({document}):",
        '"""',
        text,
        '"""',
    ]
    return "\n".join(lines)


def check_prompt(text: str, pairs: list[dict]) -> str:
    lines = [
        "Source text:",
        '"""',
        text,
        '"""',
        "",
        "Training examples:",
    ]
    for i, pair in enumerate(pairs, 1):
        mark = " (meant to be unanswerable)" if pair.get("unanswerable") else ""
        lines += [f"{i}.{mark} Q: {pair['question']}", f"   A: {pair['answer']}"]
    lines += [
        "",
        "An example is good when every fact in its answer is stated in the source text and the "
        "answer really answers the question. An example meant to be unanswerable is good when "
        "the text does not answer the question and the answer invents nothing.",
        'Reply with JSON listing the numbers of the good examples: {"good": [1, 2]}',
    ]
    return "\n".join(lines)


class Generator:
    def __init__(self, knowledge: KnowledgeStore, datasets: DatasetStore, teachers):
        self.knowledge = knowledge
        self.datasets = datasets
        self.teachers = teachers
        self.db.executescript(GENERATED_SCHEMA)

    @property
    def db(self):
        return self.knowledge.db

    def unused_chunks(self, dataset: str, collection: str, limit: int) -> list[dict]:
        """Random chunks of the collection that this dataset has not been generated from."""
        rows = self.db.execute(
            "SELECT k.id, k.text, d.name, d.source FROM chunks k "
            "JOIN documents d ON d.id = k.document_id "
            "WHERE d.collection = ? AND length(k.text) >= ? AND k.id NOT IN "
            "(SELECT chunk_id FROM generated_chunks WHERE dataset = ?) "
            "ORDER BY random() LIMIT ?",
            (collection, MIN_CHUNK_CHARS, dataset, limit),
        )
        return [dict(r) for r in rows]

    def count_unused(self, dataset: str, collection: str) -> int:
        return self.db.execute(
            "SELECT COUNT(*) FROM chunks k JOIN documents d ON d.id = k.document_id "
            "WHERE d.collection = ? AND length(k.text) >= ? AND k.id NOT IN "
            "(SELECT chunk_id FROM generated_chunks WHERE dataset = ?)",
            (collection, MIN_CHUNK_CHARS, dataset),
        ).fetchone()[0]

    def _used(self, dataset: str, chunk_id: int) -> None:
        with self.knowledge._lock:
            self.db.execute(
                "INSERT OR IGNORE INTO generated_chunks VALUES (?, ?, ?)",
                (dataset, chunk_id, now()),
            )
            self.db.commit()

    def samples(self, dataset: str, limit: int = 3) -> list[list[dict]]:
        """A few examples written or imported by the user, to show the teacher the style."""
        rows = self.db.execute(
            "SELECT messages FROM examples WHERE dataset = ? AND origin != 'generated' "
            "ORDER BY id LIMIT ?",
            (dataset, limit),
        )
        return [json.loads(r["messages"]) for r in rows]

    def run(self, job: GenerationJob) -> None:
        options = job.options
        try:
            teacher = self.teachers.teacher(options.teacher)
            teacher.on_wait = lambda seconds: self._wait(job, seconds)
            teacher.on_served = job.count_served
            chunks = self.unused_chunks(job.dataset, options.collection, options.chunks)
            job.chunks_total = len(chunks)
            if not chunks:
                job.note("No unused chunks left in the collection for this dataset.")
            samples = self.samples(job.dataset)
            failures = 0
            for chunk in chunks:
                if job.cancelled.is_set():
                    job.state = "cancelled"
                    return
                job.current = chunk["name"]
                try:
                    self._chunk(job, teacher, chunk, samples)
                    failures = 0
                except (TeacherError, ValueError) as e:
                    failures += 1
                    job.failed_chunks += 1
                    job.note(f"failed   {chunk['name']}: {e}")
                    if isinstance(e, TeacherError) and failures >= MAX_FAILURES_IN_A_ROW:
                        raise
                finally:
                    job.chunks_done += 1
                    job.waiting = None
            job.state = "done"
        except Exception as e:  # shown in the UI
            log.exception("Generation %s failed", job.id)
            job.state, job.error = "error", str(e)
        finally:
            job.current = None

    def _wait(self, job: GenerationJob, seconds: float) -> None:
        job.waiting = seconds
        job.note(f"rate limit: waiting {seconds:.0f} s")

    def _chunk(self, job: GenerationJob, teacher, chunk: dict, samples) -> None:
        options = job.options
        prompt = generation_prompt(options, chunk["name"], chunk["text"], samples)
        reply = teacher.complete(SYSTEM, prompt, max_tokens=4096, temperature=0.7)
        try:
            data = parse_json(reply)
        except ValueError:
            job.reject("not JSON", 1)
            raise ValueError("the teacher's reply was not JSON") from None
        items = data.get("examples", []) if isinstance(data, dict) else data
        pairs = []
        for item in items if isinstance(items, list) else []:
            job.generated += 1
            reason = self._problem(item, options)
            if reason:
                job.reject(reason)
            else:
                pairs.append(
                    {
                        "question": item["question"].strip(),
                        "answer": item["answer"].strip(),
                        "unanswerable": bool(item.get("unanswerable")),
                    }
                )
        if pairs and options.verify:
            pairs = self._verified(job, teacher, chunk["text"], pairs)
        added = duplicates = 0
        for pair in pairs:
            messages = [{"role": "user", "content": pair["question"]}]
            if options.system_prompt.strip():
                messages.insert(0, {"role": "system", "content": options.system_prompt.strip()})
            messages.append({"role": "assistant", "content": pair["answer"]})
            result = self.datasets.add(job.dataset, messages, "generated", chunk["source"])
            if result["duplicate"]:
                duplicates += 1
            else:
                added += 1
        job.added += added
        job.duplicates += duplicates
        self._used(job.dataset, chunk["id"])
        job.note(f"added    {added} from {chunk['name']}")

    def _problem(self, item, options: GenerationOptions) -> str | None:
        if not isinstance(item, dict):
            return "malformed"
        question, answer = item.get("question"), item.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            return "malformed"
        if not question.strip() or len(question) > 500:
            return "length"
        if not options.min_answer_chars <= len(answer.strip()) <= options.max_answer_chars:
            return "length"
        if options.languages:
            language = detect_language(f"{question} {answer}")
            if language is not None and language not in options.languages:
                return "language"
        return None

    def _verified(self, job: GenerationJob, teacher, text: str, pairs: list[dict]) -> list[dict]:
        reply = teacher.complete(CHECK_SYSTEM, check_prompt(text, pairs), 512, 0.0)
        try:
            data = parse_json(reply)
            good = data.get("good", []) if isinstance(data, dict) else data
            keep = {int(n) for n in good}
        except (ValueError, TypeError):
            # A broken check reply should not throw away the whole chunk.
            job.note("the check reply was not JSON; keeping the pairs unchecked")
            return pairs
        kept = [pair for i, pair in enumerate(pairs, 1) if i in keep]
        if len(kept) < len(pairs):
            job.reject("not supported by the text", len(pairs) - len(kept))
        return kept


class GenerationManager:
    """Runs one generation at a time in a background thread and keeps jobs for the UI."""

    def __init__(self, generator: Generator):
        self.generator = generator
        self.jobs: dict[str, GenerationJob] = {}

    def start(self, dataset: str, options: GenerationOptions) -> GenerationJob:
        job = GenerationJob(dataset, options)
        self.jobs[job.id] = job
        threading.Thread(target=self.generator.run, args=(job,), daemon=True).start()
        return job

    def running(self) -> GenerationJob | None:
        return next((j for j in self.jobs.values() if j.state == "running"), None)
