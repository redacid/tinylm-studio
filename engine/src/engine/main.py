"""FastAPI application factory."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from engine import __version__
from engine.api import admin, anthropic, knowledge, openai, system
from engine.api.deps import AuthError, is_anthropic
from engine.config import Settings
from engine.models.manager import ModelManager
from engine.rag.embeddings import HashEmbedder, SentenceTransformerEmbedder
from engine.rag.store import KnowledgeStore


def open_knowledge(settings: Settings) -> KnowledgeStore:
    if settings.embedding_model == "hash":
        return KnowledgeStore(settings.knowledge_db, HashEmbedder())
    cache = settings.data_dir / "embeddings"
    embedder = SentenceTransformerEmbedder(settings.embedding_model, cache)
    return KnowledgeStore(settings.knowledge_db, embedder)


def create_app(
    settings: Settings | None = None,
    models: ModelManager | None = None,
    knowledge_store: KnowledgeStore | None = None,
) -> FastAPI:
    settings = settings or Settings()
    settings.workspace_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="tinylm-studio engine", version=__version__)
    app.state.settings = settings
    app.state.models = models or ModelManager(
        settings.models_dir,
        enable_echo=settings.enable_echo,
        hf_token=settings.hf_token,
        runtime_options={
            "threads": settings.threads,
            "repetition_penalty": settings.repetition_penalty,
        },
    )

    app.state.knowledge = knowledge_store or open_knowledge(settings)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.exception_handler(AuthError)
    def auth_error(request: Request, _: AuthError):
        if is_anthropic(request):
            return anthropic.error(401, "invalid x-api-key", "authentication_error")
        return openai.error(
            401, "Incorrect API key provided", "invalid_request_error", "invalid_api_key"
        )

    app.include_router(system.router)
    app.include_router(openai.router)
    app.include_router(anthropic.router)
    app.include_router(admin.router)
    app.include_router(knowledge.router)
    return app
