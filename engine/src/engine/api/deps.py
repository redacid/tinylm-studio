"""Shared request dependencies: API key check, model manager and knowledge base."""

import secrets

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader, HTTPBearer

from engine.chat import ChatRequest
from engine.models.manager import ModelManager
from engine.rag.prompt import retrieve
from engine.rag.store import KnowledgeStore


class AuthError(Exception):
    pass


# Either header carries the key; declared so that the OpenAPI schema shows them (Swagger's
# Authorize button). The check itself is below, so a missing key gets the OpenAI-style error.
bearer = HTTPBearer(auto_error=False, description="The engine's TINYLM_API_KEY")
api_key_header = APIKeyHeader(
    name="x-api-key", auto_error=False, description="The engine's TINYLM_API_KEY"
)


def require_api_key(
    request: Request, _bearer=Depends(bearer), _header=Depends(api_key_header)
) -> None:
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
    request: Request, chat: ChatRequest, collection: str | None, top_k: int | None = None
) -> tuple[ChatRequest, list[dict], str | None]:
    """Adds retrieved chunks from the collection to the prompt and returns the collection used.
    A request that names no collection uses the model's own (its settings), if it has one; ""
    asks for none. Raises CollectionNotFound."""
    model_settings = request.app.state.models.settings.get(chat.model)
    if collection is None:
        collection = model_settings.get("collection")
    if not collection:
        return chat, [], None
    top_k = top_k or model_settings.get("rag_top_k") or request.app.state.settings.rag_top_k
    grounded_chat, sources = retrieve(request.app.state.knowledge, collection, chat, top_k)
    return grounded_chat, sources, collection
