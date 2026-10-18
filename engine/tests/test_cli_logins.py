"""Logging the subscription tools in from the app (Settings › Subscriptions)."""

import sys
import time

import pytest

# Stand-ins for the tools' login commands. A code "good" logs in; anything else fails.
FAKE = """#!{python}
import json, os, sys, time
tool = os.path.basename(sys.argv[0]).removeprefix("fake-")
args = sys.argv[1:]
home = os.environ[{{"claude": "CLAUDE_CONFIG_DIR", "codex": "CODEX_HOME",
                   "gemini": "GEMINI_CLI_HOME"}}[tool]]
if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"):
    print("an API key reached the tool"); sys.exit(3)
marker = os.path.join(home, "logged-in")
if args == ["--version"]:
    print("1.2.3 (" + tool + ")"); sys.exit(0)
if tool == "claude":
    if args[:2] == ["auth", "status"]:
        on = os.path.exists(marker)
        print(json.dumps({{"loggedIn": on, "email": "me@example.com" if on else None}}))
    elif args[:2] == ["auth", "logout"]:
        os.remove(marker)
    elif args[:2] == ["auth", "login"]:
        print("If the browser didn't open, visit: https://claude.com/oauth?code=true&x=1")
        print("Paste code here if prompted > ", end="", flush=True)
        if sys.stdin.readline().strip() != "good":
            print("Invalid code"); sys.exit(1)
        open(marker, "w").close(); print("Login successful.")
elif tool == "codex":
    if args == ["login", "status"]:
        print("Logged in using ChatGPT" if os.path.exists(marker) else "Not logged in")
        sys.exit(0 if os.path.exists(marker) else 1)
    elif args == ["logout"]:
        os.remove(marker)
    elif args == ["login", "--device-auth"]:
        print("Open https://auth.openai.com/codex/device and enter ABCD-1234", flush=True)
        time.sleep(0.3)
        open(marker, "w").close(); print("Successfully logged in")
else:
    assert sys.stdin.isatty(), "Gemini CLI logs in only in a terminal"
    settings = json.load(open(os.path.join(home, ".gemini", "settings.json")))
    assert settings["security"]["auth"]["selectedType"] == "oauth-personal"
    while True:
        print("Please visit the following URL to authorize the application:")
        print("\\nhttps://accounts.google.com/o/oauth2/v2/auth?state=1\\n")
        print("Enter the authorization code: ", end="", flush=True)
        if sys.stdin.readline().strip() == "good":
            break
        print("Failed to authenticate with user code.")
    folder = os.path.join(home, ".gemini")
    with open(os.path.join(folder, "google_accounts.json"), "w") as f:
        json.dump({{"active": "me@gmail.com"}}, f)
    with open(os.path.join(folder, "oauth_creds.json"), "w") as f:
        f.write("{{}}")
    time.sleep(60)  # the chat would start here
"""


@pytest.fixture
def tools(client, tmp_path, monkeypatch):
    folder = tmp_path / "bin"
    folder.mkdir()
    for tool in ("claude", "codex", "gemini"):
        program = folder / f"fake-{tool}"
        program.write_text(FAKE.format(python=sys.executable))
        program.chmod(0o755)
    monkeypatch.setenv("PATH", f"{folder}:/usr/bin:/bin")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-for-the-tools")
    cli = client.app.state.cli_tools
    cli.programs = {tool: f"fake-{tool}" for tool in ("claude", "codex", "gemini")}
    return client


def status(client, tool: str) -> dict:
    return next(t for t in client.get("/api/cli-tools").json()["data"] if t["tool"] == tool)


def wait(client, tool: str, state: str, wants_code: bool | None = None) -> dict:
    for _ in range(100):
        current = status(client, tool)
        login = current["login"] or {}
        if login.get("state") == state and wants_code in (None, login.get("wants_code")):
            return current
        time.sleep(0.1)
    raise AssertionError(f"{tool}: {current}")


def test_status_of_the_tools(tools):
    data = tools.get("/api/cli-tools").json()["data"]
    assert [t["tool"] for t in data] == ["claude", "codex", "gemini"]
    assert all(t["installed"] and not t["logged_in"] and t["login"] is None for t in data)
    assert data[0]["version"] == "1.2.3 (claude)" and data[0]["name"] == "Claude Code"
    tools.app.state.cli_tools.programs["codex"] = "not-installed"
    assert not status(tools, "codex")["installed"]
    assert tools.post("/api/cli-tools/codex/login").status_code == 409
    assert tools.post("/api/cli-tools/grok/login").status_code == 404


def test_claude_login_with_a_code(tools, settings):
    started = tools.post("/api/cli-tools/claude/login").json()
    assert started["login"]["state"] == "running"
    login = wait(tools, "claude", "running", wants_code=True)["login"]
    assert login["url"] == "https://claude.com/oauth?code=true&x=1"
    assert (
        tools.post("/api/cli-tools/claude/login/code", json={"code": " good "}).status_code == 200
    )
    done = wait(tools, "claude", "done")
    assert done["logged_in"] and done["account"] == "me@example.com"
    # The login is the engine's, in its data folder.
    assert (settings.data_dir / "cli-logins" / "claude" / "logged-in").exists()

    out = tools.post("/api/cli-tools/claude/logout").json()
    assert not out["logged_in"] and out["login"] is None


def test_claude_login_with_a_wrong_code(tools):
    tools.post("/api/cli-tools/claude/login")
    wait(tools, "claude", "running", wants_code=True)
    tools.post("/api/cli-tools/claude/login/code", json={"code": "bad"})
    failed = wait(tools, "claude", "failed")
    assert failed["login"]["error"].endswith("Invalid code") and not failed["logged_in"]
    assert tools.post("/api/cli-tools/claude/login/code", json={"code": "x"}).status_code == 409


def test_codex_device_login(tools):
    tools.post("/api/cli-tools/codex/login")
    login = wait(tools, "codex", "running")["login"]
    assert not login["wants_code"]
    done = wait(tools, "codex", "done")
    assert "ABCD-1234" in done["login"]["output"]
    assert done["logged_in"] and done["account"] == "Logged in using ChatGPT"


def test_gemini_login_in_a_terminal(tools):
    tools.post("/api/cli-tools/gemini/login")
    wait(tools, "gemini", "running", wants_code=True)
    tools.post("/api/cli-tools/gemini/login/code", json={"code": "bad"})
    time.sleep(0.5)
    tools.post("/api/cli-tools/gemini/login/code", json={"code": "good"})
    done = wait(tools, "gemini", "done")
    assert "Failed to authenticate" in done["login"]["output"]
    assert done["logged_in"] and done["account"] == "me@gmail.com"
    assert not tools.post("/api/cli-tools/gemini/logout").json()["logged_in"]


def test_cancelling_a_login(tools):
    tools.post("/api/cli-tools/gemini/login")
    wait(tools, "gemini", "running", wants_code=True)
    assert tools.delete("/api/cli-tools/gemini/login").status_code == 204
    assert status(tools, "gemini")["login"]["state"] == "cancelled"
