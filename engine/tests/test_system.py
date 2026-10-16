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


def test_log_file_and_its_end(tmp_path, client, settings):
    import logging

    from engine import logs

    path = logs.setup(settings.data_dir, "INFO")
    try:
        log = logging.getLogger("engine.test")
        log.info("first")
        try:
            raise ValueError("broken")
        except ValueError:
            log.exception("second")
        access = logging.getLogger("uvicorn.access")
        access.info('%s - "%s %s HTTP/%s" %d', "1.2.3.4:5", "GET", "/health", "1.1", 200)
        access.info('%s - "%s %s HTTP/%s" %d', "1.2.3.4:5", "POST", "/api/x", "1.1", 201)
        access.info('%s - "%s %s HTTP/%s" %d', "1.2.3.4:5", "GET", "/api/y", "1.1", 500)
        for handler in logging.getLogger().handlers:
            handler.flush()
        lines = client.get("/api/logs?lines=4").json()["lines"]
        assert client.get("/api/logs").json()["file"] == str(path)
    finally:
        for handler in list(logging.getLogger().handlers):
            logging.getLogger().removeHandler(handler)
            handler.close()
        logging.captureWarnings(False)
    # The successful GET is at DEBUG, below the file's level; the traceback stays with its line.
    assert [line.split(" ", 3)[2:] for line in lines][0] == ["INFO", "engine.test: first"]
    assert " ERROR engine.test: second\nTraceback" in lines[1]
    assert lines[1].endswith("ValueError: broken")
    assert '"POST /api/x HTTP/1.1" 201' in lines[2]
    assert '"GET /api/y HTTP/1.1" 500' in lines[3]
    assert len(lines) == 4
