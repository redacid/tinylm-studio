import os

import pytest
from fastapi.testclient import TestClient

from engine.api import schemas
from engine.config import Settings
from engine.main import create_app
from engine.rag.embeddings import HashEmbedder
from engine.rag.store import KnowledgeStore

# A field a response sends but its schema leaves out fails the test instead of being dropped.
schemas.STRICT = True


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """The engine's variables of the developer's .env (which `make test` exports, an API key or a
    proxy, say) stay out of the tests; TINYLM_TEST_* ones are for the tests themselves."""
    for name in list(os.environ):
        if (name.startswith("TINYLM_") and not name.startswith("TINYLM_TEST_")) or name in (
            "HF_TOKEN",
            "SEARXNG_SECRET",
        ):
            monkeypatch.delenv(name)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path, enable_echo=True)


@pytest.fixture
def store(settings) -> KnowledgeStore:
    # The hash embedder needs no model download.
    return KnowledgeStore(settings.knowledge_db, HashEmbedder())


@pytest.fixture
def client(settings, store) -> TestClient:
    return TestClient(create_app(settings, knowledge_store=store))
