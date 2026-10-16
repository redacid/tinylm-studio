"""Per-request timings, reported to clients as a `timings` field (named like llama.cpp server's)."""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from engine.chat import Finish, StreamEvent, TextDelta

log = logging.getLogger("engine.chat")


def ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


@dataclass
class Timings:
    load_ms: float = 0.0  # loading the model, when this request had to load it
    retrieval_ms: float = 0.0  # searching uploaded data
    prompt_n: int = 0  # prompt tokens
    prompt_ms: float = 0.0  # reading the prompt, until the first answer token
    predicted_n: int = 0  # answer tokens
    predicted_ms: float = 0.0  # generating the rest of the answer
    label: str = ""  # what the answer was for, in the log: "chat qwen3-1.7b-gguf, collection games"

    @contextmanager
    def measure(self, field: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            setattr(self, field, ms(time.perf_counter() - started))

    def track(self, events: Iterator[StreamEvent]) -> Iterator[StreamEvent]:
        """Passes events through, filling in prompt and generation figures before Finish."""
        started = time.perf_counter()
        first: float | None = None
        for event in events:
            if isinstance(event, TextDelta) and first is None:
                first = time.perf_counter()
            elif isinstance(event, Finish):
                now = time.perf_counter()
                self.prompt_ms = ms((first or now) - started)
                self.predicted_ms = ms(now - (first or now))
                self.prompt_n = event.usage.input_tokens
                self.predicted_n = event.usage.output_tokens
                t = self.to_dict()
                log.info(
                    "%s: %d prompt tokens in %.1f s, %d answer tokens at %.1f tok/s "
                    "(%.1f s in all%s)",
                    self.label or "answer",
                    self.prompt_n,
                    self.prompt_ms / 1000,
                    self.predicted_n,
                    t["predicted_per_second"],
                    t["total_ms"] / 1000,
                    f", {self.load_ms / 1000:.1f} s loading the model"
                    if self.load_ms > 100
                    else "",
                )
            yield event

    def to_dict(self) -> dict:
        def rate(n: int, duration_ms: float) -> float:
            return round(n * 1000 / duration_ms, 1) if duration_ms else 0.0

        return {
            "load_ms": self.load_ms,
            "retrieval_ms": self.retrieval_ms,
            "prompt_n": self.prompt_n,
            "prompt_ms": self.prompt_ms,
            "prompt_per_second": rate(self.prompt_n, self.prompt_ms),
            "predicted_n": self.predicted_n,
            "predicted_ms": self.predicted_ms,
            "predicted_per_second": rate(self.predicted_n, self.predicted_ms),
            "total_ms": round(
                self.load_ms + self.retrieval_ms + self.prompt_ms + self.predicted_ms, 1
            ),
        }
