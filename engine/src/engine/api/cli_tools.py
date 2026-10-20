"""Logging the subscription tools of cli teachers in from the app: Claude Code, Codex CLI and
Gemini CLI. The logins are the engine's own, kept in its data folder."""

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel

from engine.api.deps import require_api_key
from engine.api.schemas import UNAUTHORIZED, CliLogin, CliTool, Router, errors, items
from engine.teachers.cli_tools import CliTools, ToolError

router = Router(
    prefix="/api/cli-tools",
    dependencies=[Depends(require_api_key)],
    tags=["teachers"],
    responses=UNAUTHORIZED,
)


class CodeRequest(BaseModel):
    code: str


def tools(request: Request) -> CliTools:
    return request.app.state.cli_tools


def tool_status(store: CliTools, tool: str) -> dict:
    return next(s for s in store.status() if s["tool"] == tool)


@router.get("", response_model=items(CliTool))
def list_tools(store: CliTools = Depends(tools)) -> dict:
    """Each tool: installed (version), logged in (account) and a login going on."""
    return {"data": store.status()}


@router.get("/logins", response_model=dict[str, CliLogin | None])
def logins(store: CliTools = Depends(tools)) -> dict:
    """Each tool's last login (or null), quick to ask while one goes on."""
    return store.login_states()


@router.post("/{tool}/login", response_model=CliLogin, responses=errors(404, 409))
def start_login(tool: str, store: CliTools = Depends(tools)) -> dict:
    """Starts logging the tool in and returns the login; its link follows in /logins."""
    try:
        return store.login(tool)
    except ToolError as e:
        raise HTTPException(404 if "Unknown" in str(e) else 409, str(e)) from None


@router.post("/{tool}/login/code", response_model=CliLogin, responses=errors(409))
def send_code(tool: str, body: CodeRequest, store: CliTools = Depends(tools)) -> dict:
    """The code the login page showed, for Claude Code and Gemini CLI."""
    try:
        return store.send_code(tool, body.code)
    except ToolError as e:
        raise HTTPException(409, str(e)) from None


@router.delete("/{tool}/login", status_code=204)
def cancel_login(tool: str, store: CliTools = Depends(tools)) -> None:
    store.cancel(tool)


@router.post("/{tool}/logout", response_model=CliTool, responses=errors(404))
def logout(tool: str, store: CliTools = Depends(tools)) -> dict:
    try:
        store.logout(tool)
    except ToolError as e:
        raise HTTPException(404, str(e)) from None
    return tool_status(store, tool)
