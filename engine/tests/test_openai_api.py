"""Contract tests: the official OpenAI SDK talks to the engine."""

import openai
import pytest
from fastapi.testclient import TestClient

from engine.config import Settings
from engine.main import create_app


def sdk(client: TestClient, api_key: str = "unused") -> openai.OpenAI:
    return openai.OpenAI(base_url="http://testserver/v1", api_key=api_key, http_client=client)


def test_list_models(client):
    ids = [m.id for m in sdk(client).models.list()]
    assert ids == ["echo"]


def test_chat_completion(client):
    response = sdk(client).chat.completions.create(
        model="echo",
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Привіт, як справи?"},
        ],
    )
    choice = response.choices[0]
    assert choice.message.content == "Echo: Привіт, як справи?"
    assert choice.finish_reason == "stop"
    assert response.usage.completion_tokens == 4


def test_streaming_with_usage(client):
    stream = sdk(client).chat.completions.create(
        model="echo",
        messages=[{"role": "user", "content": "one two three"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = list(stream)
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == "Echo: one two three"
    assert [c.choices[0].finish_reason for c in chunks if c.choices][-1] == "stop"
    assert chunks[-1].usage.completion_tokens == 4


def test_max_tokens_and_stop(client):
    api = sdk(client)
    messages = [{"role": "user", "content": "a b c d e"}]
    short = api.chat.completions.create(model="echo", messages=messages, max_tokens=2)
    assert short.choices[0].finish_reason == "length"
    stopped = api.chat.completions.create(model="echo", messages=messages, stop=[" c"])
    assert stopped.choices[0].message.content == "Echo: a b"


def test_unknown_model(client):
    with pytest.raises(openai.NotFoundError):
        sdk(client).chat.completions.create(
            model="missing", messages=[{"role": "user", "content": "hi"}]
        )


def test_api_key_is_enforced(tmp_path):
    client = TestClient(create_app(Settings(data_dir=tmp_path, enable_echo=True, api_key="secret")))
    with pytest.raises(openai.AuthenticationError):
        sdk(client, "wrong").models.list()
    assert [m.id for m in sdk(client, "secret").models.list()] == ["echo"]
