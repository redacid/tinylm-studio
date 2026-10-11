"""Contract tests: the official Anthropic SDK talks to the engine."""

import anthropic
import pytest
from fastapi.testclient import TestClient

from engine.config import Settings
from engine.main import create_app


def sdk(client: TestClient, api_key: str = "unused") -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url="http://testserver", api_key=api_key, http_client=client)


def test_list_models(client):
    assert [m.id for m in sdk(client).models.list()] == ["echo"]


def test_create_message(client):
    message = sdk(client).messages.create(
        model="echo",
        max_tokens=100,
        system="Be brief.",
        messages=[{"role": "user", "content": [{"type": "text", "text": "Hello there"}]}],
    )
    assert message.content[0].text == "Echo: Hello there"
    assert message.stop_reason == "end_turn"
    assert message.usage.output_tokens == 3


def test_streaming(client):
    with sdk(client).messages.stream(
        model="echo", max_tokens=100, messages=[{"role": "user", "content": "Как дела?"}]
    ) as stream:
        text = "".join(stream.text_stream)
        final = stream.get_final_message()
    assert text == "Echo: Как дела?"
    assert final.stop_reason == "end_turn"
    assert final.usage.output_tokens == 3


def test_stop_reasons(client):
    api = sdk(client)
    messages = [{"role": "user", "content": "a b c d"}]
    assert (
        api.messages.create(model="echo", max_tokens=2, messages=messages).stop_reason
        == "max_tokens"
    )
    stopped = api.messages.create(
        model="echo", max_tokens=50, messages=messages, stop_sequences=[" c"]
    )
    assert stopped.stop_reason == "stop_sequence"
    assert stopped.stop_sequence == " c"
    assert stopped.content[0].text == "Echo: a b"


def test_unknown_model(client):
    with pytest.raises(anthropic.NotFoundError):
        sdk(client).messages.create(
            model="missing", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
        )


def test_api_key_is_enforced(tmp_path):
    client = TestClient(create_app(Settings(data_dir=tmp_path, enable_echo=True, api_key="secret")))
    with pytest.raises(anthropic.AuthenticationError):
        sdk(client, "wrong").models.list()
    assert [m.id for m in sdk(client, "secret").models.list()] == ["echo"]


def test_timings_are_reported(client):
    message = sdk(client).messages.create(
        model="echo", max_tokens=64, messages=[{"role": "user", "content": "one two three"}]
    )
    assert message.model_extra["timings"]["predicted_n"] == 4
