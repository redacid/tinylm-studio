import json
import logging

from fastapi.testclient import TestClient

from engine import logs
from engine.config import Settings
from engine.engine_settings import mask_proxy, unmask_proxy, with_saved
from engine.main import create_app
from engine.models.hub import hub_token


def test_defaults_come_from_the_environment(client):
    current = client.get("/api/settings").json()
    assert current["saved"] == []
    assert current["threads"] is None
    assert current["rag_top_k"] == 4
    assert current["log_level"] == "INFO"
    assert current["defaults"]["repetition_penalty"] == 1.1


def test_token_is_saved_but_never_sent_back(client, settings):
    token = "hf_abcdefghijklmnopqrstuvwxyz1234"
    saved = client.put("/api/settings", json={"hf_token": token}).json()
    assert token not in json.dumps(saved)
    assert saved["hf_token_set"] and saved["hf_token_source"] == "app"
    assert saved["hf_token_hint"] == "hf_…1234"
    assert saved["saved"] == ["hf_token"]
    # In effect at once, for downloads and Hub searches too.
    assert client.app.state.models.hf_token == token
    assert client.get("/api/models").json()["hf_token"] is True
    file = settings.workspace_dir / "engine-settings.json"
    assert json.loads(file.read_text())["hf_token"] == token
    assert file.stat().st_mode & 0o077 == 0

    # "" switches the token off, even one from the environment; null goes back to it.
    off = client.put("/api/settings", json={"hf_token": ""}).json()
    assert not off["hf_token_set"] and client.app.state.models.token is False
    assert client.get("/api/models").json()["hf_token"] is False
    back = client.put("/api/settings", json={"hf_token": None}).json()
    assert back["saved"] == []
    assert hub_token(client.app.state.models.hf_token) is None


def test_environment_token_and_saved_values_win(tmp_path, store):
    env = Settings(data_dir=tmp_path, hf_token="hf_fromtheenvironment9876", threads=2)
    client = TestClient(create_app(env, knowledge_store=store))
    current = client.get("/api/settings").json()
    assert current["hf_token_source"] == "environment"
    assert current["hf_token_hint"] == "hf_…9876"
    assert current["threads"] == 2

    client.put("/api/settings", json={"threads": 6, "repetition_penalty": 1.3})
    assert client.app.state.models.runtime_options == {"threads": 6, "repetition_penalty": 1.3}
    assert client.app.state.exports.threads == 6
    # A restarted engine (and the CLI) starts with the saved values.
    again = Settings(data_dir=tmp_path, threads=2)
    assert with_saved(again).threads == 6
    restarted = TestClient(create_app(again, knowledge_store=store))
    current = restarted.get("/api/settings").json()
    assert current["threads"] == 6 and current["defaults"]["threads"] == 2
    assert current["saved"] == ["repetition_penalty", "threads"]


def test_proxy_password_is_masked(client):
    proxy = "socks5://user:secret@127.0.0.1:1080"
    saved = client.put("/api/settings", json={"proxy": proxy}).json()
    assert saved["proxy"] == "socks5://user:****@127.0.0.1:1080"
    assert client.app.state.crawls.crawler.proxy == proxy
    assert client.app.state.settings.proxy == proxy  # what search uses
    # Sent back with the mask and a new host, the password stays.
    client.put("/api/settings", json={"proxy": "socks5://user:****@10.0.0.1:1080"})
    assert client.app.state.settings.proxy == "socks5://user:secret@10.0.0.1:1080"
    client.put("/api/settings", json={"proxy": ""})
    assert client.app.state.crawls.crawler.proxy is None


def test_invalid_values_are_refused(client, tmp_path):
    for change in (
        {"proxy": "ftp://host"},
        {"threads": 0},
        {"rag_top_k": 100},
        {"repetition_penalty": 5},
        {"log_level": "LOUD"},
        {"llama_cpp_dir": str(tmp_path / "missing")},
        {"color": "red"},
    ):
        assert client.put("/api/settings", json=change).status_code == 422, change
    answer = client.put("/api/settings", json={"proxy": "ftp://host"}).json()
    assert answer["detail"].startswith("proxy: not an http://")
    assert client.get("/api/settings").json()["saved"] == []
    folder = tmp_path / "llama.cpp"
    folder.mkdir()
    client.put("/api/settings", json={"llama_cpp_dir": str(folder)})
    assert client.app.state.exports.llama_cpp_dir == folder


def test_rag_top_k_and_log_level_apply_at_once(client):
    client.put("/api/settings", json={"rag_top_k": 7})
    assert client.app.state.settings.rag_top_k == 7

    handler = logging.NullHandler()
    handler.set_name(logs.HANDLER_NAME)
    logging.getLogger().addHandler(handler)
    try:
        client.put("/api/settings", json={"log_level": "WARNING"})
        assert handler.level == logging.WARNING
        client.put("/api/settings", json={"log_level": None})
        assert handler.level == logging.INFO
    finally:
        logging.getLogger().removeHandler(handler)


def test_proxy_masking():
    assert mask_proxy("http://h:3128") == "http://h:3128"
    assert mask_proxy("http://u:p@h:3128") == "http://u:****@h:3128"
    assert unmask_proxy("http://u:****@h:1", "http://u:p@old:2") == "http://u:p@h:1"
    assert unmask_proxy("http://u:new@h:1", "http://u:p@old:2") == "http://u:new@h:1"
