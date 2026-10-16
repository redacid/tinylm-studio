"""The chat answering with a saved teacher (an API model) instead of an installed model."""

import json

import httpx
import pytest

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

    def fake_client(config):
        kind = AnthropicTeacher if config.kind == "anthropic" else OpenAICompatibleTeacher
        teacher = kind(config.base_url, config.api_key, config.model, config.name)
        teacher.client = httpx.Client(transport=httpx.MockTransport(provider))
        return teacher

    monkeypatch.setattr(teacher_store, "client", fake_client)
    refs = {}
    for kind, model in [
        ("openai", "big/model:free"),
        ("anthropic", "claude"),
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
