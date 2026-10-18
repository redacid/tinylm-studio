"""Management API for teachers: the models that write training examples."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from engine.api.deps import require_api_key
from engine.teachers.providers import TeacherError, cli_tool
from engine.teachers.store import TeacherConfig, TeacherNotFound, TeacherStore, client

router = APIRouter(
    prefix="/api/teachers", dependencies=[Depends(require_api_key)], tags=["teachers"]
)


class TeacherRequest(BaseModel):
    name: str = ""  # defaults to the model
    kind: Literal["openai", "anthropic", "cli"] = "openai"
    # Defaults: https://openrouter.ai/api/v1 (openai) and https://api.anthropic.com (anthropic).
    # For cli, the program: claude, codex or gemini, or a path to one.
    base_url: str = ""
    # Empty for cli leaves the model to the tool.
    model: str = ""
    # Left empty when editing: the saved key is kept.
    api_key: str | None = None
    per_minute: float | None = Field(None, gt=0, le=10000)
    # Answer settings for every request; empty (null, "default") leaves it to the request.
    max_tokens: int | None = Field(None, ge=1, le=1_000_000)
    temperature: float | None = Field(None, ge=0, le=2)
    thinking: Literal["default", "off", "on"] = "default"
    # For listing models or testing a teacher being edited, whose key is not sent again.
    id: int | None = None


def teachers(request: Request) -> TeacherStore:
    return request.app.state.teachers


def config(body: TeacherRequest, store: TeacherStore) -> TeacherConfig:
    api_key = body.api_key
    if not api_key and body.id is not None:
        try:
            api_key = store.get(body.id).api_key
        except TeacherNotFound:
            raise HTTPException(404, f"Unknown teacher {body.id}") from None
    name = body.name.strip() or body.model.strip()
    if body.kind == "cli":
        try:
            name = name or cli_tool(body.base_url.strip() or "claude")
        except TeacherError as e:
            raise HTTPException(422, str(e)) from None
    return TeacherConfig(
        name=name,
        kind=body.kind,
        model=body.model.strip(),
        base_url=body.base_url.strip(),
        api_key=(api_key or None) if body.kind != "cli" else None,
        per_minute=body.per_minute,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        thinking=body.thinking,
    )


@router.get("")
def list_teachers(store: TeacherStore = Depends(teachers)) -> dict:
    """Saved teachers (without their keys) and installed models as local teachers; use "ref"
    to pick one for generation."""
    return {"data": store.all()}


@router.post("", status_code=201)
def add_teacher(body: TeacherRequest, store: TeacherStore = Depends(teachers)) -> dict:
    if not body.model.strip() and body.kind != "cli":
        raise HTTPException(422, "Choose a model.")
    return store.add(config(body, store)).to_dict()


@router.put("/{teacher_id}")
def change_teacher(
    teacher_id: int, body: TeacherRequest, store: TeacherStore = Depends(teachers)
) -> dict:
    try:
        return store.update(teacher_id, config(body, store)).to_dict()
    except TeacherNotFound:
        raise HTTPException(404, f"Unknown teacher {teacher_id}") from None


@router.delete("/{teacher_id}", status_code=204)
def delete_teacher(teacher_id: int, store: TeacherStore = Depends(teachers)) -> None:
    try:
        store.delete(teacher_id)
    except TeacherNotFound:
        raise HTTPException(404, f"Unknown teacher {teacher_id}") from None


@router.post("/models")
def provider_models(body: TeacherRequest, store: TeacherStore = Depends(teachers)) -> dict:
    """The provider's models, free ones first (OpenRouter marks them)."""
    try:
        return {"data": client(config(body, store)).models()}
    except TeacherError as e:
        raise HTTPException(502, str(e)) from None


@router.post("/test")
def test_teacher(body: TeacherRequest, store: TeacherStore = Depends(teachers)) -> dict:
    """Sends a tiny request to check the address, key and model."""
    teacher = client(config(body, store))
    try:
        reply = teacher.complete("Reply with one word.", "Say OK.", max_tokens=16, temperature=0)
    except TeacherError as e:
        raise HTTPException(502, str(e)) from None
    # The model and provider that answered, as the API reported them.
    return {"reply": reply.strip()[:200], "served": teacher.last_served}
