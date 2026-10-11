from fastapi.testclient import TestClient

from engine.config import Settings
from engine.main import create_app


def make_client(tmp_path) -> TestClient:
    return TestClient(create_app(Settings(data_dir=tmp_path)))


def test_health(tmp_path):
    response = make_client(tmp_path).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_system_info_reports_hardware(tmp_path):
    body = make_client(tmp_path).get("/api/system").json()
    assert body["hardware"]["cpu_cores"] >= 1
    assert body["hardware"]["memory_gb"] > 0


def test_workspace_is_created(tmp_path):
    make_client(tmp_path)
    assert (tmp_path / "workspace").is_dir()
