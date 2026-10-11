"""Shared request dependencies: API key check and access to the model manager."""

import secrets

from fastapi import Request

from engine.models.manager import ModelManager


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


def is_anthropic(request: Request) -> bool:
    """Anthropic SDKs always send anthropic-version; OpenAI SDKs never do."""
    return "anthropic-version" in request.headers
