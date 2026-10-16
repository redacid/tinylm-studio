"""The chat answering with a saved teacher (an API model) instead of an installed model."""

import json

import httpx
import pytest

from engine.teachers import providers
from engine.teachers import store as teacher_store
from engine.teachers.providers import AnthropicTeacher, OpenAICompatibleTeacher

seen: list[dict] = []


def sse(*events: dict, done: bool = True) -> bytes:
    lines = [f"data: {json.dumps(e)}\n\n" for e in events]
    return ("".join(lines) + ("data: [DONE]\n\n" if done else "")).encode()


def provider(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    seen.append({"url": str(request.url), **body})
    if body["model"] == "broken":
        return httpx.Response(401, json={"error": {"message": "bad key"}})
    if body["model"] == "claude-new" and "temperature" in body:
        message = "`temperature` is deprecated for this model."
        return httpx.Response(
            400, json={"error": {"type": "invalid_request_error", "message": message}}
        )
    if body["model"] == "thinker":  # spends the whole answer thinking
        events = [
            {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "Hm"}},
            {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}},
        ]
        return httpx.Response(200, content=sse(*events, done=False))
    if request.url.path.endswith("/v1/messages") and not body.get("stream"):
        return httpx.Response(200, json={"content": [{"type": "text", "text": "Квен."}]})
    if request.url.path.endswith("/v1/messages"):
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 12}}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Знак "}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Ігні."}},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 3},
            },
        ]
        return httpx.Response(200, content=sse(*events, done=False))
    chunks = [
        {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
        {"choices": [{"delta": {"content": "Знак "}}]},
        {"choices": [{"delta": {"content": "Аард."}, "finish_reason": "stop"}]},
    ]
    return httpx.Response(200, content=sse(*chunks))


@pytest.fixture
def teachers(client, monkeypatch):
    seen.clear()
    monkeypatch.setattr(providers, "NO_SAMPLING", set())

    def fake_client(config):
        kind = AnthropicTeacher if config.kind == "anthropic" else OpenAICompatibleTeacher
        teacher = kind(
            config.base_url, config.api_key, config.model, config.name, options=config.options
        )
        teacher.client = httpx.Client(transport=httpx.MockTransport(provider))
        return teacher

    monkeypatch.setattr(teacher_store, "client", fake_client)
    refs = {}
    for kind, model in [
        ("openai", "big/model:free"),
        ("anthropic", "claude"),
        ("anthropic", "claude-new"),
        ("anthropic", "thinker"),
        ("openai", "broken"),
    ]:
        body = {"kind": kind, "base_url": f"https://{kind}.test", "model": model, "api_key": "k"}
        refs[model] = client.post("/api/teachers", json=body).json()["ref"]
    return refs


def ask(client, model: str, stream: bool = False):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Short answers."},
            {"role": "user", "content": "Знак проти утопців?"},
            {"role": "assistant", "content": "Ігні."},
            {"role": "user", "content": "А інший?"},
        ],
        "max_tokens": 50,
        "stream": stream,
    }
    return client.post("/v1/chat/completions", json=body)


def test_chat_with_an_openai_compatible_teacher(client, teachers):
    response = ask(client, f"teacher:{teachers['big/model:free']}")
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "Знак Аард."
    sent = seen[-1]
    assert sent["url"] == "https://openai.test/chat/completions" and sent["stream"] is True
    assert [m["role"] for m in sent["messages"]] == ["system", "user", "assistant", "user"]
    assert sent["max_tokens"] == 50

    streamed = ask(client, f"teacher:{teachers['big/model:free']}", stream=True)
    texts = [
        json.loads(line[6:])["choices"][0]["delta"].get("content", "")
        for line in streamed.text.splitlines()
        if line.startswith("data: {")
    ]
    assert "".join(texts) == "Знак Аард."


def test_chat_with_an_anthropic_teacher(client, teachers):
    response = ask(client, f"teacher:{teachers['claude']}")
    assert response.json()["choices"][0]["message"]["content"] == "Знак Ігні."
    assert response.json()["usage"]["completion_tokens"] == 3
    sent = seen[-1]
    assert sent["system"] == "Short answers."
    assert [m["role"] for m in sent["messages"]] == ["user", "assistant", "user"]


def test_teacher_errors_in_the_chat(client, teachers):
    answer = ask(client, f"teacher:{teachers['broken']}").json()
    assert "rejected the API key" in answer["choices"][0]["message"]["content"]
    assert ask(client, "teacher:999").status_code == 404
    assert ask(client, "teacher:x").status_code == 404


def test_claude_models_that_refuse_a_temperature(client, teachers):
    """Newer Claude models answer HTTP 400 to temperature: asked again without, and remembered."""
    response = ask(client, f"teacher:{teachers['claude-new']}")
    assert response.json()["choices"][0]["message"]["content"] == "Знак Ігні."
    assert "temperature" in seen[-2] and "temperature" not in seen[-1]
    seen.clear()
    ask(client, f"teacher:{teachers['claude-new']}", stream=True)
    assert len(seen) == 1 and "temperature" not in seen[0]

    teacher = AnthropicTeacher("https://anthropic.test", "k", "claude-new")
    teacher.client = httpx.Client(transport=httpx.MockTransport(provider))
    assert teacher.complete("system", "prompt") == "Квен."
    old = AnthropicTeacher("https://anthropic.test", "k", "claude")
    old.client = httpx.Client(transport=httpx.MockTransport(provider))
    old.complete("system", "prompt", temperature=0.3)
    assert seen[-1]["temperature"] == 0.3


def test_answer_settings_of_a_teacher(client, teachers):
    """Max tokens, temperature and thinking saved with a teacher apply to every request."""
    ref = teachers["claude"]
    body = {"kind": "anthropic", "base_url": "https://anthropic.test", "model": "claude"}
    settings = {"max_tokens": 4000, "temperature": 0.2, "thinking": "on"}
    saved = client.put(f"/api/teachers/{ref}", json={**body, **settings}).json()
    assert {k: saved[k] for k in settings} == settings
    ask(client, f"teacher:{ref}")
    sent = seen[-1]
    assert sent["max_tokens"] == 4000 and sent["temperature"] == 0.2
    assert sent["thinking"] == {"type": "adaptive"}

    client.put(f"/api/teachers/{ref}", json={**body, "thinking": "off"})
    ask(client, f"teacher:{ref}")
    assert seen[-1]["thinking"] == {"type": "disabled"} and seen[-1]["max_tokens"] == 50
    client.put(f"/api/teachers/{ref}", json=body)
    ask(client, f"teacher:{ref}")
    assert "thinking" not in seen[-1]

    openai = {"kind": "openai", "base_url": "https://openai.test", "model": "big/model:free"}
    client.put(f"/api/teachers/{teachers['big/model:free']}", json={**openai, "thinking": "off"})
    ask(client, f"teacher:{teachers['big/model:free']}")
    assert seen[-1]["reasoning"] == {"enabled": False}


def test_an_answer_spent_on_thinking(client, teachers):
    answer = ask(client, f"teacher:{teachers['thinker']}").json()["choices"][0]
    assert "used all 50 tokens before answering" in answer["message"]["content"]
    assert answer["finish_reason"] == "length"


def test_older_database_gets_the_answer_settings(tmp_path):
    import sqlite3

    from engine.rag.embeddings import HashEmbedder
    from engine.rag.store import KnowledgeStore
    from engine.teachers.store import TeacherStore

    path = tmp_path / "knowledge.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE teachers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL,"
        " base_url TEXT NOT NULL, model TEXT NOT NULL, api_key TEXT, per_minute REAL,"
        " created_at TEXT NOT NULL);"
        "INSERT INTO teachers VALUES (1, 'old', 'anthropic', 'https://a.test', 'claude', 'k',"
        " NULL, '2026-01-01');"
    )
    old.commit()
    old.close()
    teacher = TeacherStore(KnowledgeStore(path, HashEmbedder()), models=None).get(1)
    assert teacher.thinking == "default" and teacher.max_tokens is None
