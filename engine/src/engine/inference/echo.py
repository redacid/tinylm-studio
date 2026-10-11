"""Deterministic runtime that echoes the last user message. Used in tests and UI development."""

from collections.abc import Iterator

from engine.chat import ChatRequest, Finish, StopSequenceFilter, StreamEvent, TextDelta, Usage


class EchoRuntime:
    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        last_user = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        words = f"Echo: {last_user}".split(" ")
        input_tokens = sum(len(m.content.split()) for m in request.messages)

        stop = StopSequenceFilter(request.stop)
        emitted = 0
        for index, word in enumerate(words):
            if emitted >= request.max_tokens:
                yield Finish("length", Usage(input_tokens, emitted))
                return
            text = stop.feed(word if index == 0 else " " + word)
            emitted += 1
            if text:
                yield TextDelta(text)
            if stop.matched:
                yield Finish("stop_sequence", Usage(input_tokens, emitted), stop.matched)
                return
        if tail := stop.flush():
            yield TextDelta(tail)
        yield Finish("stop", Usage(input_tokens, emitted))
