"""Logging the subscription tools of cli teachers in from the app: Claude Code, Codex CLI and
Gemini CLI. The logins are the engine's own, kept in its data folder."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from engine.api.deps import require_api_key
from engine.teachers.cli_tools import CliTools, ToolError

router = APIRouter(
    prefix="/api/cli-tools", dependencies=[Depends(require_api_key)], tags=["teachers"]
)


class CodeRequest(BaseModel):
    code: str


def tools(request: Request) -> CliTools:
    return request.app.state.cli_tools


def tool_status(store: CliTools, tool: str) -> dict:
    return next(s for s in store.status() if s["tool"] == tool)


@router.get("")
def list_tools(store: CliTools = Depends(tools)) -> dict:
    """Each tool: installed (version), logged in (account) and a login going on."""
    return {"data": store.status()}


@router.post("/{tool}/login")
def start_login(tool: str, store: CliTools = Depends(tools)) -> dict:
    """Starts logging the tool in; the status's login then has the link to open."""
    try:
        store.login(tool)
    except ToolError as e:
        raise HTTPException(404 if "Unknown" in str(e) else 409, str(e)) from None
    return tool_status(store, tool)


@router.post("/{tool}/login/code")
def send_code(tool: str, body: CodeRequest, store: CliTools = Depends(tools)) -> dict:
    """The code the login page showed, for Claude Code and Gemini CLI."""
    try:
        return store.send_code(tool, body.code)
    except ToolError as e:
        raise HTTPException(409, str(e)) from None


@router.delete("/{tool}/login", status_code=204)
def cancel_login(tool: str, store: CliTools = Depends(tools)) -> None:
    store.cancel(tool)


@router.post("/{tool}/logout")
def logout(tool: str, store: CliTools = Depends(tools)) -> dict:
    try:
        store.logout(tool)
    except ToolError as e:
        raise HTTPException(404, str(e)) from None
    return tool_status(store, tool)
