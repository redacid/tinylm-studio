"""Anthropic-compatible endpoint: /v1/messages (GET /v1/models is shared with the OpenAI router)."""

import json
import uuid
from collections.abc import Iterator
from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from engine.api.deps import manager, require_api_key
from engine.chat import ChatRequest, Finish, Message, TextDelta, collect
from engine.models.manager import ModelManager

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
    temperature: float = 1.0
    top_p: float = 1.0
    stop_sequences: list[str] = []
    stream: bool = False

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
def create_message(body: MessagesRequest, models: ModelManager = Depends(manager)):
    if body.model not in models.installed():
        return error(404, f"model: {body.model}", "not_found_error")
    request = body.to_internal()
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
        }

    if not body.stream:
        result = collect(runtime.stream(request))
        usage = result.finish.usage
        return envelope(
            [{"type": "text", "text": result.text}],
            STOP_REASONS[result.finish.reason],
            result.finish.stop_sequence,
            {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
        )

    def events() -> Iterator[str]:
        stream = runtime.stream(request)
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
                    },
                )
        yield sse("message_stop", {})

    return StreamingResponse(events(), media_type="text/event-stream")
