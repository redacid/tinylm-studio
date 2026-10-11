"""OpenAI-compatible endpoints: /v1/chat/completions and /v1/models."""

import json
import time
import uuid
from collections.abc import Iterator
from typing import Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from engine.api import anthropic as anthropic_api
from engine.api.deps import grounded, is_anthropic, manager, require_api_key
from engine.chat import ChatRequest, Finish, Message, TextDelta, collect
from engine.models.manager import ModelManager
from engine.rag.store import CollectionNotFound

router = APIRouter(dependencies=[Depends(require_api_key)], tags=["openai"])

FINISH_REASONS = {"stop": "stop", "stop_sequence": "stop", "length": "length"}


class ContentPart(BaseModel):
    type: str
    text: str = ""


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[ContentPart] | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 1.0
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    # tinylm extension: answer from this collection of uploaded data, with cited sources.
    collection: str | None = None

    def to_internal(self) -> ChatRequest:
        messages = []
        for m in self.messages:
            if isinstance(m.content, list):
                text = "".join(p.text for p in m.content if p.type == "text")
            else:
                text = m.content or ""
            role = "system" if m.role == "developer" else m.role
            messages.append(Message(role=role, content=text))
        stop = [self.stop] if isinstance(self.stop, str) else (self.stop or [])
        return ChatRequest(
            model=self.model,
            messages=messages,
            max_tokens=self.max_completion_tokens or self.max_tokens or 512,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=stop,
        )


def error(status: int, message: str, kind: str, code: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": {"message": message, "type": kind, "code": code}}
    )


@router.get("/v1/models")
def list_models(request: Request, models: ModelManager = Depends(manager)):
    if is_anthropic(request):
        return anthropic_api.list_models(models)
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 0, "owned_by": "tinylm"}
            for m in models.installed()
        ],
    }


@router.post("/v1/chat/completions")
def chat_completions(
    body: ChatCompletionRequest, http: Request, models: ModelManager = Depends(manager)
):
    if body.model not in models.installed():
        return error(
            404,
            f"The model '{body.model}' does not exist",
            "invalid_request_error",
            "model_not_found",
        )
    try:
        request, sources = grounded(http, body.to_internal(), body.collection)
    except CollectionNotFound:
        return error(
            404, f"The collection '{body.collection}' does not exist", "invalid_request_error"
        )
    # Sources are an extension field, only present when a collection was used.
    extra = {"sources": sources} if body.collection else {}
    runtime = models.runtime(body.model)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if not body.stream:
        result = collect(runtime.stream(request))
        usage = result.finish.usage
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": FINISH_REASONS[result.finish.reason],
                }
            ],
            "usage": {
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "total_tokens": usage.input_tokens + usage.output_tokens,
            },
            **extra,
        }

    include_usage = bool(body.stream_options and body.stream_options.include_usage)

    def chunk(delta: dict, finish_reason: str | None = None, **extra) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": body.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            **extra,
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def events() -> Iterator[str]:
        yield chunk({"role": "assistant", "content": ""}, **extra)
        for event in runtime.stream(request):
            if isinstance(event, TextDelta):
                yield chunk({"content": event.text})
            elif isinstance(event, Finish):
                yield chunk({}, FINISH_REASONS[event.reason])
                if include_usage:
                    usage = event.usage
                    payload = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": body.model,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": usage.input_tokens,
                            "completion_tokens": usage.output_tokens,
                            "total_tokens": usage.input_tokens + usage.output_tokens,
                        },
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
