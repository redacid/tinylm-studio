from fastapi.testclient import TestClient

from engine.config import Settings
from engine.main import create_app


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_system_info_reports_hardware(client):
    body = client.get("/api/system").json()
    assert body["hardware"]["cpu_cores"] >= 1
    assert body["hardware"]["memory_gb"] > 0


def test_workspace_is_created(tmp_path):
    TestClient(create_app(Settings(data_dir=tmp_path)))
    assert (tmp_path / "workspace").is_dir()


def test_only_health_answers_without_the_api_key(tmp_path):
    client = TestClient(create_app(Settings(data_dir=tmp_path, api_key="secret")))
    assert client.get("/health").status_code == 200
    assert client.get("/api/system").status_code == 401
    assert client.get("/api/system", headers={"authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/system", headers={"authorization": "Bearer secret"}).status_code == 200


def test_health_reports_the_build_commit(tmp_path, monkeypatch):
    from engine import commit

    monkeypatch.setenv("TINYLM_BUILD_COMMIT", "abc1234")
    commit.cache_clear()
    try:
        body = TestClient(create_app(Settings(data_dir=tmp_path))).get("/health").json()
    finally:
        commit.cache_clear()
    assert body == {"status": "ok", "version": "0.1.0", "commit": "abc1234"}
