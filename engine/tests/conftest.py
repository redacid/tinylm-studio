import pytest
from fastapi.testclient import TestClient

from engine.api import schemas
from engine.config import Settings
from engine.main import create_app
from engine.rag.embeddings import HashEmbedder
from engine.rag.store import KnowledgeStore

# A field a response sends but its schema leaves out fails the test instead of being dropped.
schemas.STRICT = True


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
