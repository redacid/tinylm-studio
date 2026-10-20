"""OpenAI-compatible endpoints: /v1/chat/completions and /v1/models."""

import json
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict
from typing import Literal

from fastapi import Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from engine.api import anthropic as anthropic_api
from engine.api.deps import grounded, is_anthropic, manager, require_api_key
from engine.api.schemas import (
    UNAUTHORIZED,
    AnthropicModelList,
    ChatCompletion,
    OpenAIError,
    OpenAIModelList,
    Router,
    event_stream,
)
from engine.chat import ChatRequest, Finish, Message, TextDelta, collect
from engine.models.manager import ModelManager
from engine.rag.store import CollectionNotFound
from engine.timings import Timings

router = Router(dependencies=[Depends(require_api_key)], tags=["openai"], responses=UNAUTHORIZED)

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
    # Left out: the model's settings (Models tab), then 0.7 and 1.
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, gt=0, le=1)
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    # tinylm extension: answer from this collection of uploaded data, with cited sources.
    collection: str | None = None
    # tinylm extensions: how many chunks of the collection to add (TINYLM_RAG_TOP_K by default)
    # and the repetition penalty (TINYLM_REPETITION_PENALTY by default).
    rag_top_k: int | None = Field(None, ge=1, le=20)
    repetition_penalty: float | None = Field(None, gt=0, le=2)
    # tinylm extensions: sampling settings llama.cpp and transformers know, and the reasoning
    # mode of models that have one (Qwen3); left out: the model's settings.
    top_k: int | None = Field(None, ge=0)
    min_p: float | None = Field(None, ge=0, le=1)
    thinking: bool | None = None
    # tinylm extension: also return the request as the model got it, with the added sources.
    debug: bool = False

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
            max_tokens=self.max_completion_tokens or self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=stop,
            repetition_penalty=self.repetition_penalty,
            top_k=self.top_k,
            min_p=self.min_p,
            thinking=self.thinking,
        )


def error(status: int, message: str, kind: str, code: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": {"message": message, "type": kind, "code": code}}
    )


@router.get("/v1/models", response_model=OpenAIModelList | AnthropicModelList)
def list_models(request: Request, models: ModelManager = Depends(manager)):
    """The models to chat with: installed models, then saved teachers
    ("tinylm/teacher/<name>"). In Anthropic's form when the request has anthropic-version."""
    if is_anthropic(request):
        return anthropic_api.list_models(models)
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 0, "owned_by": "tinylm"}
            for m in models.chat_models()
        ],
    }


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletion,
    responses={
        **event_stream("With stream: chat.completion.chunk objects, then [DONE]"),
        404: {"model": OpenAIError, "description": "Unknown model or collection"},
    },
)
def chat_completions(
    body: ChatCompletionRequest, http: Request, models: ModelManager = Depends(manager)
):
    """OpenAI's chat completions, with tinylm's extensions (collection, sources, timings)."""
    if not models.can_chat(body.model):
        return error(
            404,
            f"The model '{body.model}' does not exist",
            "invalid_request_error",
            "model_not_found",
        )
    timings = Timings()
    try:
        with timings.measure("retrieval_ms"):
            request, sources, collection = grounded(
                http, body.to_internal(), body.collection, body.rag_top_k
            )
    except CollectionNotFound as missing:
        return error(404, f"The collection '{missing}' does not exist", "invalid_request_error")
    # Sources are an extension field, only present when a collection was used.
    extra = {"sources": sources, "collection": collection} if collection else {}
    if body.debug:
        extra["prompt"] = asdict(request)
    timings.label = f"{body.model}" + (f", collection {collection}" if collection else "")
    with timings.measure("load_ms"):
        runtime = models.runtime(body.model)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if not body.stream:
        result = collect(timings.track(runtime.stream(request)))
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
            "timings": timings.to_dict(),
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
        for event in timings.track(runtime.stream(request)):
            if isinstance(event, TextDelta):
                yield chunk({"content": event.text})
            elif isinstance(event, Finish):
                yield chunk({}, FINISH_REASONS[event.reason], timings=timings.to_dict())
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
