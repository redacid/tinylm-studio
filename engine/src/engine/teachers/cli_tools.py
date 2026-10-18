"""Logging the subscription tools of cli teachers (Claude Code, Codex CLI, Gemini CLI) in from the
app, so that the logins are the engine's own and kept in its data folder (<data dir>/cli-logins,
on the volume in a container).

Each tool logs in through a link the user opens in a browser on any computer:
- Claude Code (claude auth login): the page shows a code, which is pasted back to the tool.
- Codex CLI (codex login --device-auth): the page asks for a one-time code the tool printed, and
  the tool notices by itself when the user confirmed it.
- Gemini CLI (interactive, with NO_BROWSER): like Claude Code. It only asks in a terminal, so it
  runs in a pseudo-terminal, and is stopped once it has saved its credentials.
"""

import json
import logging
import os
import re
import select
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from engine.teachers.providers import API_KEY_VARIABLES, CLI_TOOLS

log = logging.getLogger(__name__)

NAMES = {"claude": "Claude Code", "codex": "Codex CLI", "gemini": "Gemini CLI"}
# Where each tool keeps its login, under the engine's folder for it.
HOME_VARIABLES = {"claude": "CLAUDE_CONFIG_DIR", "codex": "CODEX_HOME", "gemini": "GEMINI_CLI_HOME"}
# Gemini CLI's choice of logging in with a Google account, which it needs before it answers.
GEMINI_SETTINGS = {"security": {"auth": {"selectedType": "oauth-personal"}}}
# A login nobody finishes is stopped; the tools give up after a few minutes themselves.
LOGIN_TIMEOUT = 15 * 60
OUTPUT_LIMIT = 4000
ESCAPES = re.compile(r"\x1b\[[0-9;?<>=]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-_]")
URL = re.compile(r"https://[^\s\"'<>]+")


class ToolError(Exception):
    pass


def clean(text: str) -> str:
    """Terminal output without colors and cursor moves."""
    text = ESCAPES.sub("", text).replace("\r\n", "\n")
    return "".join(c for c in text if c in "\n\t" or c >= " ")


@dataclass
class Login:
    """A login going on: the tool's output, the link to open and whether it waits for a code."""

    tool: str
    process: subprocess.Popen
    terminal: int | None = None  # the pseudo-terminal of Gemini CLI
    output: str = ""
    state: str = "running"  # running, done, failed, cancelled
    error: str | None = None
    started: float = field(default_factory=time.monotonic)
    code_sent: bool = False

    @property
    def url(self) -> str | None:
        """The link the tool printed last (Gemini CLI prints a new one after a wrong code)."""
        found = URL.findall(self.output)
        return found[-1].rstrip(".,)") if found else None

    @property
    def wants_code(self) -> bool:
        return self.tool != "codex" and self.state == "running" and self.url is not None

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "output": self.output[-OUTPUT_LIMIT:],
            "url": self.url,
            "wants_code": self.wants_code,
            "error": self.error,
        }


class CliTools:
    """The tools' logins: their status, logging in and out."""

    def __init__(self, home: Path, programs: dict[str, str] | None = None):
        self.home = home
        # The program of each tool; tests put stand-ins here.
        self.programs = programs or {tool: tool for tool in CLI_TOOLS}
        self.logins: dict[str, Login] = {}
        self._lock = threading.Lock()

    def folder(self, tool: str) -> Path:
        folder = self.home / tool
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        return folder

    def environment(self) -> dict:
        """The environment the tools run in: the engine's logins, and no API keys, so that they
        answer with the subscription."""
        env = {k: v for k, v in os.environ.items() if k not in API_KEY_VARIABLES}
        for tool, variable in HOME_VARIABLES.items():
            env[variable] = str(self.folder(tool))
        return env

    def program(self, tool: str) -> str | None:
        if tool not in CLI_TOOLS:
            raise ToolError(f"Unknown tool {tool}")
        return shutil.which(self.programs[tool])

    def status(self) -> list[dict]:
        """Every tool: installed (its version) and logged in (as whom, when the tool says)."""
        threads, results = [], {}
        for tool in CLI_TOOLS:
            thread = threading.Thread(target=lambda t=tool: results.__setitem__(t, self._status(t)))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        return [results[tool] for tool in CLI_TOOLS]

    def _status(self, tool: str) -> dict:
        program = self.program(tool)
        login = self.logins.get(tool)
        status = {
            "tool": tool,
            "name": NAMES[tool],
            "installed": program is not None,
            "version": None,
            "logged_in": False,
            "account": None,
            "login": login.to_dict() if login else None,
        }
        if program is None:
            return status
        version = self._run([program, "--version"])
        status["version"] = (
            version.stdout.strip().splitlines()[0] if version.stdout.strip() else None
        )
        status["logged_in"], status["account"] = getattr(self, f"_{tool}_status")(program)
        return status

    def _run(self, args: list[str], timeout: float = 30) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory(prefix="tinylm-tool-") as folder:
            try:
                return subprocess.run(
                    args, capture_output=True, text=True, cwd=folder,
                    env=self.environment(), timeout=timeout, stdin=subprocess.DEVNULL,
                )  # fmt: skip
            except (OSError, subprocess.TimeoutExpired) as e:
                return subprocess.CompletedProcess(args, 1, "", str(e))

    def _claude_status(self, program: str) -> tuple[bool, str | None]:
        done = self._run([program, "auth", "status", "--json"])
        try:
            data = json.loads(done.stdout)
        except ValueError:
            return False, None
        account = data.get("email") or data.get("subscriptionType")
        return bool(data.get("loggedIn")), account

    def _codex_status(self, program: str) -> tuple[bool, str | None]:
        done = self._run([program, "login", "status"])
        text = clean(done.stdout + done.stderr)
        lines = [line for line in text.splitlines() if line and not line.startswith("WARNING")]
        return done.returncode == 0, (lines[-1] if done.returncode == 0 and lines else None)

    def _gemini_status(self, program: str) -> tuple[bool, str | None]:
        folder = self.folder("gemini") / ".gemini"
        if not (folder / "oauth_creds.json").exists():
            return False, None
        try:
            accounts = json.loads((folder / "google_accounts.json").read_text())
            return True, accounts.get("active")
        except (OSError, ValueError, AttributeError):
            return True, None

    def login(self, tool: str) -> dict:
        """Starts logging a tool in; a login of it still going is stopped first."""
        program = self.program(tool)
        if program is None:
            raise ToolError(f"{NAMES[tool]} is not installed on the engine's computer")
        self.cancel(tool)
        folder = self.folder(tool)
        if tool == "gemini":
            self._gemini_settings(folder)
        args = {
            "claude": [program, "auth", "login", "--claudeai"],
            "codex": [program, "login", "--device-auth"],
            "gemini": [program, "--skip-trust"],
        }[tool]
        env = self.environment() | {"NO_BROWSER": "true", "BROWSER": "true", "TERM": "dumb"}
        cwd = tempfile.mkdtemp(prefix="tinylm-login-")
        if tool == "gemini":
            import pty  # Unix only, like Gemini CLI's login here

            terminal, child = pty.openpty()
            env["TERM"] = "xterm-256color"
            process = subprocess.Popen(
                args, stdin=child, stdout=child, stderr=child, cwd=cwd, env=env,
                start_new_session=True,
            )  # fmt: skip
            os.close(child)
            login = Login(tool, process, terminal=terminal)
        else:
            process = subprocess.Popen(
                args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=cwd, env=env, start_new_session=True,
            )  # fmt: skip
            login = Login(tool, process)
        with self._lock:
            self.logins[tool] = login
        threading.Thread(target=self._follow, args=(login, cwd), daemon=True).start()
        log.info("Logging %s in", NAMES[tool])
        return login.to_dict()

    def send_code(self, tool: str, code: str) -> dict:
        login = self.logins.get(tool)
        if login is None or not login.wants_code:
            raise ToolError(f"{NAMES.get(tool, tool)} is not waiting for a code")
        code = code.strip()
        if not code:
            raise ToolError("Paste the code the page showed")
        if login.terminal is not None:
            os.write(login.terminal, (code + "\r").encode())
        else:
            login.process.stdin.write((code + "\n").encode())
            login.process.stdin.flush()
        login.code_sent = True
        return login.to_dict()

    def cancel(self, tool: str) -> None:
        login = self.logins.get(tool)
        if login and login.state == "running":
            login.state = "cancelled"
            self._stop(login)

    def logout(self, tool: str) -> None:
        self.cancel(tool)
        program = self.program(tool)
        if tool == "gemini":
            for name in ("oauth_creds.json", "google_accounts.json"):
                (self.folder("gemini") / ".gemini" / name).unlink(missing_ok=True)
        elif program:
            self._run([program, "auth", "logout"] if tool == "claude" else [program, "logout"])
        self.logins.pop(tool, None)
        log.info("Logged %s out", NAMES[tool])

    def _gemini_settings(self, folder: Path) -> None:
        settings_file = folder / ".gemini" / "settings.json"
        settings_file.parent.mkdir(mode=0o700, exist_ok=True)
        try:
            settings = json.loads(settings_file.read_text())
        except (OSError, ValueError):
            settings = {}
        settings.setdefault("security", {}).setdefault("auth", {})["selectedType"] = (
            GEMINI_SETTINGS["security"]["auth"]["selectedType"]
        )
        settings_file.write_text(json.dumps(settings, indent=2))

    def _follow(self, login: Login, cwd: str) -> None:
        """Reads the tool's output until it is done, failed, stopped or timed out."""
        source = login.terminal if login.terminal is not None else login.process.stdout.fileno()
        credentials = self.folder("gemini") / ".gemini" / "oauth_creds.json"
        had_credentials = credentials.exists() and credentials.stat().st_mtime
        try:
            while login.state == "running":
                if time.monotonic() - login.started > LOGIN_TIMEOUT:
                    login.state, login.error = "failed", "The login was not finished in time"
                    break
                try:
                    ready, _, _ = select.select([source], [], [], 0.5)
                    # Nothing to read from a ready output is its end (a closed terminal too).
                    data = os.read(source, 65536) if ready else b""
                except (OSError, ValueError):  # closed by cancel()
                    ready, data = True, b""
                if data:
                    text = login.output + clean(data.decode(errors="replace"))
                    login.output = text[-4 * OUTPUT_LIMIT :]
                if login.tool == "gemini" and credentials.exists():
                    if credentials.stat().st_mtime != had_credentials:
                        login.state = "done"  # saved: the rest is Gemini CLI's chat
                        break
                if login.process.poll() is not None and not data:
                    if login.state != "running":  # cancelled meanwhile
                        break
                    if login.tool == "gemini" or login.process.returncode != 0:
                        login.state = "failed"
                        login.error = self._last_line(login.output) or "The tool stopped"
                    else:
                        login.state = "done"
                    break
        finally:
            self._stop(login)
            shutil.rmtree(cwd, ignore_errors=True)
            log.info("Login of %s: %s", NAMES[login.tool], login.state)

    @staticmethod
    def _last_line(text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else ""

    @staticmethod
    def _stop(login: Login) -> None:
        if login.process.poll() is None:
            login.process.kill()
            login.process.wait()
        if login.terminal is not None:
            try:
                os.close(login.terminal)
            except OSError:
                pass
            login.terminal = None
