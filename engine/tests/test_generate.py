"""Teachers (OpenAI-compatible, Anthropic, local) and generating dataset examples with them."""

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from engine.datasets.generate import CHECK_SYSTEM, detect_language, parse_json

TEXTS = {
    "witcher.md": (
        "Відьмак 3: Дикий Гін. Утопці слабкі проти знаку Ігні та олії проти трупоїдів. "
        "Бийтеся з ними подалі від води, бо там їх більше. " * 4
    ),
    "souls.md": (
        "Dark Souls 3. Юдекс Ґундир — перший бос гри. У другій фазі він перетворюється на "
        "чорну масу, тож атакуйте ззаду. Естус відновлює здоров'я. " * 4
    ),
    "short.md": "Замало тексту.",
}


class FakeTeacher(BaseHTTPRequestHandler):
    """Answers like an OpenAI-compatible API (/chat/completions, /models) and like the
    Anthropic API (/v1/messages, /v1/models)."""

    prompts: list[str] = []
    rate_limit_once = False

    def do_GET(self):  # noqa: N802
        if not self.authorized():
            return
        if self.path.startswith("/v1/models"):
            return self.reply(200, {"data": [{"id": "claude-x", "display_name": "Claude X"}]})
        models = [
            {"id": "big/paid", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
            {"id": "small/free:free", "pricing": {"prompt": "0", "completion": "0"}},
        ]
        self.reply(200, {"data": models})

    def do_POST(self):  # noqa: N802
        if not self.authorized():
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if FakeTeacher.rate_limit_once:
            FakeTeacher.rate_limit_once = False
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/v1/messages":
            system, prompt = body["system"], body["messages"][0]["content"]
        else:
            system, prompt = body["messages"][0]["content"], body["messages"][1]["content"]
        FakeTeacher.prompts.append(prompt)
        text = self.answer(system, prompt)
        if self.path == "/v1/messages":
            body = {"model": "claude-test-1", "content": [{"type": "text", "text": text}]}
            return self.reply(200, body)
        # OpenRouter names the model and the upstream provider that answered.
        message = {"role": "assistant", "content": text}
        body = {"model": "routed/model", "provider": "Chutes", "choices": [{"message": message}]}
        self.reply(200, body)

    def answer(self, system: str, prompt: str) -> str:
        if system == CHECK_SYSTEM:
            # Rejects the pairs whose answer says "вигадка" (made up).
            numbered = re.findall(r"^(\d+)\..*?\n   A: (.*)$", prompt, re.MULTILINE)
            good = [int(n) for n, answer in numbered if "вигадка" not in answer]
            return json.dumps({"good": good})
        if prompt == "Say OK.":
            return "OK"
        pairs = [
            {
                "question": "Що допомагає проти утопців?",
                "answer": "Знак Ігні та олія проти трупоїдів.",
            },
            {"question": "Де битися?", "answer": "Подалі від води, вигадка про болото."},
            {"question": "What about English?", "answer": "This one is in the wrong language."},
            {"question": "Коротко?", "answer": "Так."},
            {
                "question": "Скільки коштує гра?",
                "answer": "Про ціну тут немає інформації.",
                "unanswerable": True,
            },
        ]
        source = prompt.split('"""')[1]
        if "Dark Souls" in source:
            pairs[0] = {"question": "Хто перший бос?", "answer": "Юдекс Ґундир, атакуйте ззаду."}
        return (
            "Ось приклади:\n```json\n"
            + json.dumps({"examples": pairs}, ensure_ascii=False)
            + "\n```"
        )

    def authorized(self) -> bool:
        key = self.headers.get("Authorization", "").removeprefix("Bearer ") or self.headers.get(
            "x-api-key"
        )
        if key != "good-key":
            self.reply(401, {"error": {"message": "invalid key"}})
            return False
        return True

    def reply(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def teacher_server():
    FakeTeacher.prompts = []
    FakeTeacher.rate_limit_once = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeTeacher)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture
def library(client, store):
    """A collection of game texts and a dataset with one hand-written example of the style."""
    client.post("/api/collections", json={"name": "games"})
    for name, text in TEXTS.items():
        store.add_document("games", name, text, f"https://games.example/{name}")
    client.post("/api/datasets", json={"name": "hints", "eval_percent": 0})
    sample = [
        {"role": "user", "content": "Як зберегти гру?"},
        {"role": "assistant", "content": "Коротко: гра зберігається сама."},
    ]
    client.post("/api/datasets/hints/examples", json={"messages": sample})
    return client


def add_teacher(client, base_url: str, kind: str = "openai", key: str = "good-key") -> str:
    body = {"kind": kind, "base_url": base_url, "model": "small/free:free", "api_key": key}
    return client.post("/api/teachers", json=body).json()["ref"]


def wait(client, job_id: str) -> dict:
    deadline = time.time() + 30
    while time.time() < deadline:
        job = client.get(f"/api/generations/{job_id}").json()
        if job["state"] != "running":
            return job
        time.sleep(0.05)
    raise TimeoutError(job)


def test_language_and_json_helpers():
    assert detect_language("Знак Ігні допомагає проти утопців") == "uk"
    assert detect_language("Знак Игни помогает против утопцев, это быстро") == "ru"
    assert detect_language("Igni helps against drowners") == "en"
    assert detect_language("Так") is None
    assert parse_json('Sure!\n```json\n{"good": [1, 3]}\n```') == {"good": [1, 3]}
    assert parse_json('<think>{"x": 1}</think> [1, 2]') == [1, 2]
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_teachers_api(client, teacher_server):
    body = {
        "kind": "openai",
        "base_url": teacher_server,
        "model": "small/free:free",
        "api_key": "good-key",
    }
    saved = client.post("/api/teachers", json=body)
    assert saved.status_code == 201
    teacher = saved.json()
    assert teacher["has_key"] and "api_key" not in teacher and teacher["name"] == "small/free:free"

    listed = client.get("/api/teachers").json()["data"]
    assert [t["ref"] for t in listed] == [teacher["ref"]]  # echo is not offered as a teacher
    assert "good-key" not in json.dumps(listed)

    models = client.post(
        "/api/teachers/models", json={"base_url": teacher_server, "id": teacher["id"]}
    )
    assert [m["id"] for m in models.json()["data"]] == ["small/free:free", "big/paid"]
    assert models.json()["data"][0]["free"]
    assert client.post(
        "/api/teachers/test", json={**body, "api_key": None, "id": teacher["id"]}
    ).json() == {"reply": "OK", "served": {"model": "routed/model", "provider": "Chutes"}}
    bad = client.post("/api/teachers/test", json={**body, "api_key": "wrong"})
    assert bad.status_code == 502 and "rejected the API key" in bad.json()["detail"]
    down = client.post("/api/teachers/test", json={**body, "base_url": "http://127.0.0.1:9"})
    assert down.status_code == 502 and "not reachable" in down.json()["detail"]

    # Editing without a key keeps the saved one.
    changed = client.put(
        f"/api/teachers/{teacher['id']}", json={**body, "api_key": "", "per_minute": 20}
    )
    assert changed.json()["has_key"] and changed.json()["per_minute"] == 20

    anthropic = {
        "kind": "anthropic",
        "base_url": teacher_server,
        "model": "claude-x",
        "api_key": "good-key",
    }
    tested = client.post("/api/teachers/test", json=anthropic).json()
    assert tested == {"reply": "OK", "served": {"model": "claude-test-1", "provider": "Anthropic"}}
    assert client.post("/api/teachers/models", json=anthropic).json()["data"][0]["id"] == "claude-x"
    default = client.post("/api/teachers", json={"kind": "anthropic", "model": "claude-x"}).json()
    assert default["base_url"] == "https://api.anthropic.com" and not default["has_key"]

    assert client.delete(f"/api/teachers/{teacher['id']}").status_code == 204
    assert client.delete(f"/api/teachers/{teacher['id']}").status_code == 404


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
def test_generate_examples(library, teacher_server, kind):
    client = library
    ref = add_teacher(client, teacher_server, kind)
    FakeTeacher.rate_limit_once = True  # waited out and retried
    options = {
        "collection": "games",
        "teacher": ref,
        "chunks": 1,
        "languages": ["uk"],
        "instructions": "Коротка порада.",
        "system_prompt": "Ти помічник гравця.",
    }
    assert client.get("/api/datasets/hints/generate", params={"collection": "games"}).json() == {
        "unused_chunks": 2  # the short text is too short to generate from
    }
    started = client.post("/api/datasets/hints/generate", json=options)
    assert started.status_code == 202
    job = wait(client, started.json()["id"])
    assert job["state"] == "done", job
    assert (job["chunks_done"], job["generated"], job["added"]) == (1, 5, 2)
    assert job["rejected"] == {"language": 1, "length": 1, "not supported by the text": 1}
    assert any("rate limit" in line for line in job["log"])
    # Two requests: writing the pairs and checking them.
    # Each request with the model and provider that answered it and what came back.
    served = {"openai": ("routed/model", "Chutes"), "anthropic": ("claude-test-1", "Anthropic")}
    model, provider = served[kind]
    requests = [
        (r["step"], r["model"], r["provider"], r["ok"], r["result"]) for r in job["requests"]
    ]
    assert requests == [
        ("write", model, provider, True, "5 pairs written"),
        ("check", model, provider, True, "2 of 3 pairs supported by the text; 2 added"),
    ]

    # The prompt carried the style: instructions, language and the hand-written example.
    prompt = FakeTeacher.prompts[0]
    assert "Коротка порада." in prompt and "Ukrainian" in prompt and "Як зберегти гру?" in prompt

    generated = client.get("/api/datasets/hints/examples", params={"origin": "generated"}).json()
    assert generated["total"] == 2
    example = generated["data"][0]
    assert example["messages"][0] == {"role": "system", "content": "Ти помічник гравця."}
    assert example["source"].startswith("https://games.example/")
    assert {e["messages"][2]["content"] for e in generated["data"]} >= {
        "Про ціну тут немає інформації."
    }
    manual = client.get("/api/datasets/hints/examples", params={"origin": "manual"}).json()
    assert manual["total"] == 1

    # The next run continues with the chunk not used yet, then there is nothing left.
    second = wait(client, client.post("/api/datasets/hints/generate", json=options).json()["id"])
    # The same unanswerable question again is a duplicate, stored once.
    assert (second["chunks_done"], second["added"], second["duplicates"]) == (1, 1, 1)
    third = wait(client, client.post("/api/datasets/hints/generate", json=options).json()["id"])
    assert third["chunks_total"] == 0 and "No unused chunks" in third["log"][0]
    assert client.get("/api/datasets/hints").status_code in (200, 405)


def test_generation_errors(library, teacher_server):
    client = library
    base = {"collection": "games", "teacher": "local:missing"}
    assert client.post("/api/datasets/hints/generate", json=base).status_code == 404
    ref = add_teacher(client, teacher_server, key="wrong-key")
    cases = [
        ({"collection": "nope", "teacher": ref}, 404),
        ({"collection": "games", "teacher": ref, "languages": ["de"]}, 422),
        ({"collection": "games", "teacher": "999"}, 404),
    ]
    for body, status in cases:
        assert client.post("/api/datasets/nope/generate", json=body).status_code == 404
        assert client.post("/api/datasets/hints/generate", json=body).status_code == status, body

    # A rejected key fails every chunk; the run stops after a few with the reason.
    for name in ("a", "b", "c"):
        client.app.state.knowledge.add_document(
            "games", name, f"Довгий текст про гру {name}. " * 30, name
        )
    job = wait(
        client,
        client.post("/api/datasets/hints/generate", json={**base, "teacher": ref}).json()["id"],
    )
    assert job["state"] == "error" and "rejected the API key" in job["error"]
    assert job["failed_chunks"] == 3
    # The failed requests name the model asked for, and the error.
    failed = job["requests"][-1]
    assert (failed["step"], failed["ok"]) == ("write", False)
    assert failed["model"] == "small/free:free" and "rejected the API key" in failed["result"]

    # Failed chunks stay unused for the next run.
    unused = client.get("/api/datasets/hints/generate", params={"collection": "games"}).json()
    assert unused["unused_chunks"] == 5


def test_local_teacher(library):
    """Installed models are teachers too; the echo test model cannot write JSON, so every chunk
    fails, but the run goes through the local runtime."""
    client = library
    client.app.state.teachers.models.installed = lambda: ["echo", "tiny"]
    refs = [t["ref"] for t in client.get("/api/teachers").json()["data"]]
    assert refs == ["local:tiny"]
