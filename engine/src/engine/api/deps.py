"""Shared request dependencies: API key check, model manager and knowledge base."""

import secrets

from fastapi import Request

from engine.chat import ChatRequest
from engine.models.manager import ModelManager
from engine.rag.prompt import retrieve
from engine.rag.store import KnowledgeStore


class AuthError(Exception):
    pass


def require_api_key(request: Request) -> None:
    expected = request.app.state.settings.api_key
    if not expected:
        return
    supplied = request.headers.get("x-api-key") or ""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        supplied = supplied or auth[7:]
    if not secrets.compare_digest(supplied, expected):
        raise AuthError()


def manager(request: Request) -> ModelManager:
    return request.app.state.models


def knowledge(request: Request) -> KnowledgeStore:
    return request.app.state.knowledge


def is_anthropic(request: Request) -> bool:
    """Anthropic SDKs always send anthropic-version; OpenAI SDKs never do."""
    return "anthropic-version" in request.headers


def grounded(
    request: Request, chat: ChatRequest, collection: str | None
) -> tuple[ChatRequest, list[dict]]:
    """Adds retrieved chunks from the collection to the prompt; raises CollectionNotFound."""
    if not collection:
        return chat, []
    top_k = request.app.state.settings.rag_top_k
    return retrieve(request.app.state.knowledge, collection, chat, top_k)
