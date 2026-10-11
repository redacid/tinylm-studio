"""FastAPI application factory."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from engine import __version__
from engine.api import admin, anthropic, knowledge, openai, search, system, web
from engine.api.deps import AuthError, is_anthropic
from engine.config import Settings
from engine.models.manager import ModelManager
from engine.rag.embeddings import HashEmbedder, SentenceTransformerEmbedder
from engine.rag.store import KnowledgeStore
from engine.web.crawler import Crawler, CrawlManager
from engine.web.search import BraveSearch, SearchError, SearchProvider, SearxngSearch


def open_knowledge(settings: Settings) -> KnowledgeStore:
    if settings.embedding_model == "hash":
        return KnowledgeStore(settings.knowledge_db, HashEmbedder())
    cache = settings.data_dir / "embeddings"
    embedder = SentenceTransformerEmbedder(settings.embedding_model, cache, settings.hf_token)
    return KnowledgeStore(settings.knowledge_db, embedder)


def crawler(settings: Settings, store: KnowledgeStore) -> Crawler:
    return Crawler(store, settings.proxy, settings.cdp_url, settings.user_agent)


def searcher(settings: Settings) -> SearchProvider:
    if settings.search_provider == "brave":
        if not settings.brave_api_key:
            raise SearchError("Set TINYLM_BRAVE_API_KEY to search with Brave, or use SearXNG.")
        return BraveSearch(settings.brave_api_key, settings.proxy)
    return SearxngSearch(settings.searxng_url)


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
    app.state.crawls = CrawlManager(crawler(settings, app.state.knowledge))
    # A new search client per request, so a changed setting or a missing key shows up at once.
    app.state.searcher = lambda: searcher(settings)

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
    app.include_router(web.router)
    app.include_router(search.router)
    return app
