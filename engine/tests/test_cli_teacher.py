"""Teachers that run a subscription's command-line tool (Claude Code, Codex, Gemini CLI)."""

import json
import sys

import pytest

from engine.chat import ChatRequest, Message, TextDelta
from engine.teachers.providers import CliTeacher, RateLimited, TeacherError

# A stand-in for each tool: notes its arguments, stdin and environment, then prints what the
# real tool would. A prompt with "LOGIN" or "LIMIT" in it fails as the tools do.
FAKE = """#!{python}
import json, os, sys
tool = os.path.basename(sys.argv[0])
stdin = sys.stdin.read()
with open(os.path.join({seen!r}, tool + ".json"), "w") as f:
    json.dump({{"args": sys.argv[1:], "stdin": stdin, "cwd": os.getcwd(),
               "key": os.environ.get("ANTHROPIC_API_KEY"),
               "home": os.environ.get("CLAUDE_CONFIG_DIR")}}, f)
if "LOGIN" in stdin and tool == "gemini":
    print("Error:", json.dumps({{"error": {{"message": "Please set an Auth method"}}}}, indent=2),
          file=sys.stderr); sys.exit(41)
if "LOGIN" in stdin:
    print("Invalid API key · Please run /login", file=sys.stderr); sys.exit(1)
if tool == "claude":
    if "LIMIT" in stdin:
        print(json.dumps({{"type": "result", "is_error": True,
                          "result": "Claude usage limit reached. Resets at 3pm"}})); sys.exit(1)
    if "stream-json" in sys.argv:
        for piece in ["Кв", "ен."]:
            print(json.dumps({{"type": "stream_event", "event": {{"type": "content_block_delta",
                  "delta": {{"type": "text_delta", "text": piece}}}}}}))
        print(json.dumps({{"type": "result", "result": "Квен.",
                          "usage": {{"input_tokens": 9, "output_tokens": 3}}}}))
    else:
        print(json.dumps({{"type": "result", "is_error": False, "result": "Квен."}}))
elif tool == "codex":
    print(json.dumps({{"type": "thread.started", "thread_id": "t"}}))
    print(json.dumps({{"type": "error", "message": "Reconnecting... 2/5"}}))
    print(json.dumps({{"type": "item.completed", "item": {{"type": "reasoning", "text": "hm"}}}}))
    print(json.dumps({{"type": "item.completed",
                      "item": {{"type": "agent_message", "text": "Codex."}}}}))
    print(json.dumps({{"type": "turn.completed", "usage": {{"output_tokens": 2}}}}))
else:
    print(json.dumps({{"response": "Gemini.", "stats": {{}}}}, indent=2))
"""


@pytest.fixture
def tools(tmp_path, monkeypatch):
    seen = tmp_path / "seen"
    seen.mkdir()
    for tool in ("claude", "codex", "gemini"):
        program = tmp_path / tool
        program.write_text(FAKE.format(python=sys.executable, seen=str(seen)))
        program.chmod(0o755)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-for-the-cli")

    def ran(tool: str) -> dict:
        return json.loads((seen / f"{tool}.json").read_text())

    return tmp_path, ran


def test_each_tool_answers(tools):
    folder, ran = tools
    claude = CliTeacher(str(folder / "claude"), "sonnet")
    assert claude.complete("Be brief.", "Name a model.") == "Квен."
    call = ran("claude")
    assert call["args"][call["args"].index("--system-prompt") + 1] == "Be brief."
    assert call["args"][call["args"].index("--model") + 1] == "sonnet"
    assert call["args"][call["args"].index("--tools") + 1] == ""
    assert call["stdin"] == "Name a model."
    # The subscription answers, not an API key the engine has, in an empty folder.
    assert call["key"] is None
    assert "tinylm-teacher-" in call["cwd"]
    assert claude.last_served == {"model": "sonnet", "provider": "claude"}

    codex = CliTeacher(str(folder / "codex"))
    assert codex.complete("Be brief.", "Name a model.") == "Codex."
    call = ran("codex")
    assert call["args"][:2] == ["exec", "--json"] and "--model" not in call["args"]
    assert call["stdin"] == "Be brief.\n\nName a model."
    assert codex.name == "codex"

    gemini = CliTeacher(str(folder / "gemini"), "gemini-pro")
    assert gemini.complete("", "Name a model.") == "Gemini."
    assert ran("gemini")["stdin"] == "Name a model."


def test_chat_with_a_tool(tools):
    folder, ran = tools
    messages = [
        Message("system", "Be brief."),
        Message("user", "Hi"),
        Message("assistant", "Hello!"),
        Message("user", "Name a model."),
    ]
    events = list(CliTeacher(str(folder / "claude")).stream_chat(ChatRequest("x", messages)))
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Кв", "ен."]
    assert events[-1].usage.output_tokens == 3
    stdin = ran("claude")["stdin"]
    assert stdin.startswith("User: Hi\n\nAssistant: Hello!\n\nUser: Name a model.")

    events = list(CliTeacher(str(folder / "codex")).stream_chat(ChatRequest("x", messages)))
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Codex."]


def test_tool_errors(tools):
    folder, _ = tools
    with pytest.raises(TeacherError, match="not logged in: log in under Settings"):
        CliTeacher(str(folder / "codex")).complete("", "LOGIN")
    with pytest.raises(TeacherError, match="not logged in: log in under Settings"):
        CliTeacher(str(folder / "gemini")).complete("", "LOGIN")
    with pytest.raises(RateLimited, match="usage limit"):
        CliTeacher(str(folder / "claude")).complete("", "LIMIT")
    with pytest.raises(TeacherError, match="not installed"):
        CliTeacher(str(folder / "claude-missing")).complete("", "Hi")
    with pytest.raises(TeacherError, match="not a known tool"):
        CliTeacher("bash")


def test_saving_a_cli_teacher(client, tools, settings):
    folder, ran = tools
    body = {"kind": "cli", "base_url": str(folder / "claude"), "api_key": "ignored"}
    saved = client.post("/api/teachers", json=body).json()
    assert saved["name"] == "claude" and saved["model"] == "" and not saved["has_key"]
    assert client.post("/api/teachers/test", json=body).json()["reply"] == "Квен."
    # It answers with the engine's login (Settings › Subscriptions).
    assert ran("claude")["home"] == str(settings.data_dir / "cli-logins" / "claude")
    models = client.post("/api/teachers/models", json=body).json()["data"]
    assert "sonnet" in [m["id"] for m in models]
    assert client.post("/api/teachers", json={**body, "base_url": "bash"}).status_code == 422

    answer = client.post(
        "/v1/chat/completions",
        json={"model": f"teacher:{saved['ref']}", "messages": [{"role": "user", "content": "Hi"}]},
    ).json()
    assert answer["choices"][0]["message"]["content"] == "Квен."
