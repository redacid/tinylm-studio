"""Runtime interface: anything that turns a ChatRequest into a stream of events."""

from collections.abc import Iterator
from typing import Protocol

from engine.chat import ChatRequest, StreamEvent


class Runtime(Protocol):
    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]: ...
