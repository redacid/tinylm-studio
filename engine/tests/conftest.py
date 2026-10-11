import pytest
from fastapi.testclient import TestClient

from engine.config import Settings
from engine.main import create_app


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path, enable_echo=True)


@pytest.fixture
def client(settings) -> TestClient:
    return TestClient(create_app(settings))
