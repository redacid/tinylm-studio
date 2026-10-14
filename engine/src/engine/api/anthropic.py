"""Anthropic-compatible endpoint: /v1/messages (GET /v1/models is shared with the OpenAI router)."""

import json
import uuid
from collections.abc import Iterator
from typing import Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from engine.api.deps import grounded, manager, require_api_key
from engine.chat import ChatRequest, Finish, Message, TextDelta, collect
from engine.models.manager import ModelManager
from engine.rag.store import CollectionNotFound
from engine.timings import Timings

router = APIRouter(dependencies=[Depends(require_api_key)], tags=["anthropic"])

STOP_REASONS = {"stop": "end_turn", "stop_sequence": "stop_sequence", "length": "max_tokens"}


class TextBlock(BaseModel):
    type: str
    text: str = ""


class InputMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[TextBlock]


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: list[InputMessage]
    system: str | list[TextBlock] | None = None
    # Left out: the model's settings (Models tab), then 0.7 and 1.
    temperature: float | None = Field(None, ge=0, le=1)
    top_p: float | None = Field(None, gt=0, le=1)
    top_k: int | None = Field(None, ge=0)
    stop_sequences: list[str] = []
    stream: bool = False
    # tinylm extension: answer from this collection of uploaded data, with cited sources.
    collection: str | None = None
    # tinylm extensions: how many chunks of the collection to add (TINYLM_RAG_TOP_K by default)
    # and the repetition penalty (TINYLM_REPETITION_PENALTY by default).
    rag_top_k: int | None = Field(None, ge=1, le=20)
    repetition_penalty: float | None = Field(None, gt=0, le=2)
    # tinylm extensions, left out: the model's settings.
    min_p: float | None = Field(None, ge=0, le=1)
    thinking: bool | None = None

    def to_internal(self) -> ChatRequest:
        messages: list[Message] = []
        if self.system:
            messages.append(Message("system", text_of(self.system)))
        messages += [Message(m.role, text_of(m.content)) for m in self.messages]
        return ChatRequest(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop_sequences,
            repetition_penalty=self.repetition_penalty,
            top_k=self.top_k,
            min_p=self.min_p,
            thinking=self.thinking,
        )


def text_of(content: str | list[TextBlock]) -> str:
    if isinstance(content, str):
        return content
    return "".join(block.text for block in content if block.type == "text")


def error(status: int, message: str, kind: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"type": "error", "error": {"type": kind, "message": message}}
    )


def list_models(models: ModelManager) -> dict:
    data = [
        {"type": "model", "id": m, "display_name": m, "created_at": "1970-01-01T00:00:00Z"}
        for m in models.installed()
    ]
    return {
        "data": data,
        "has_more": False,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
    }


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps({'type': event, **data}, ensure_ascii=False)}\n\n"


@router.post("/v1/messages")
def create_message(body: MessagesRequest, http: Request, models: ModelManager = Depends(manager)):
    if body.model not in models.installed():
        return error(404, f"model: {body.model}", "not_found_error")
    timings = Timings()
    try:
        with timings.measure("retrieval_ms"):
            request, sources, collection = grounded(
                http, body.to_internal(), body.collection, body.rag_top_k
            )
    except CollectionNotFound as missing:
        return error(404, f"collection: {missing}", "not_found_error")
    with timings.measure("load_ms"):
        runtime = models.runtime(body.model)
    message_id = f"msg_{uuid.uuid4().hex}"

    def envelope(content: list, stop_reason=None, stop_sequence=None, usage=None) -> dict:
        return {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": body.model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": stop_sequence,
            "usage": usage or {"input_tokens": 0, "output_tokens": 0},
            # Extension field, only present when a collection was used.
            **({"sources": sources, "collection": collection} if collection else {}),
        }

    if not body.stream:
        result = collect(timings.track(runtime.stream(request)))
        usage = result.finish.usage
        message = envelope(
            [{"type": "text", "text": result.text}],
            STOP_REASONS[result.finish.reason],
            result.finish.stop_sequence,
            {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
        )
        return {**message, "timings": timings.to_dict()}

    def events() -> Iterator[str]:
        stream = timings.track(runtime.stream(request))
        yield sse("message_start", {"message": envelope([])})
        yield sse(
            "content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}
        )
        yield sse("ping", {})
        for event in stream:
            if isinstance(event, TextDelta):
                yield sse(
                    "content_block_delta",
                    {"index": 0, "delta": {"type": "text_delta", "text": event.text}},
                )
            elif isinstance(event, Finish):
                yield sse("content_block_stop", {"index": 0})
                yield sse(
                    "message_delta",
                    {
                        "delta": {
                            "stop_reason": STOP_REASONS[event.reason],
                            "stop_sequence": event.stop_sequence,
                        },
                        "usage": {
                            "input_tokens": event.usage.input_tokens,
                            "output_tokens": event.usage.output_tokens,
                        },
                        "timings": timings.to_dict(),
                    },
                )
        yield sse("message_stop", {})

    return StreamingResponse(events(), media_type="text/event-stream")
