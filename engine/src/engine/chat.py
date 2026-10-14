"""Protocol-neutral chat types shared by the OpenAI and Anthropic API layers."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant"]
FinishReason = Literal["stop", "length", "stop_sequence"]


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class ChatRequest:
    model: str
    messages: list[Message]
    # None: the model's settings, then the defaults (models/settings.py fills them in
    # before a runtime sees the request).
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] = field(default_factory=list)
    repetition_penalty: float | None = None  # None at the runtime: TINYLM_REPETITION_PENALTY
    top_k: int | None = None  # None at the runtime: the runtime's own default
    min_p: float | None = None
    thinking: bool | None = None  # reasoning mode of models that have one (Qwen3)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class TextDelta:
    text: str


@dataclass
class Finish:
    reason: FinishReason
    usage: Usage
    stop_sequence: str | None = None


StreamEvent = TextDelta | Finish


@dataclass
class ChatResult:
    text: str
    finish: Finish


def collect(events: Iterator[StreamEvent]) -> ChatResult:
    """Drain a stream into a complete result."""
    parts: list[str] = []
    finish = Finish(reason="stop", usage=Usage())
    for event in events:
        if isinstance(event, TextDelta):
            parts.append(event.text)
        else:
            finish = event
    return ChatResult(text="".join(parts), finish=finish)


class StopSequenceFilter:
    """Cuts streamed text at the first stop sequence, holding back possible partial matches."""

    def __init__(self, stop: list[str]):
        self.stop = [s for s in stop if s]
        self.buffer = ""
        self.matched: str | None = None

    def feed(self, text: str) -> str:
        if self.matched is not None:
            return ""
        self.buffer += text
        for seq in self.stop:
            index = self.buffer.find(seq)
            if index != -1:
                self.matched = seq
                out, self.buffer = self.buffer[:index], ""
                return out
        hold = max((len(s) - 1 for s in self.stop), default=0)
        cut = len(self.buffer) - hold
        if cut <= 0:
            return ""
        out, self.buffer = self.buffer[:cut], self.buffer[cut:]
        return out

    def flush(self) -> str:
        out, self.buffer = ("" if self.matched else self.buffer), ""
        return out
