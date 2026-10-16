"""FastAPI application factory."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from engine import __version__
from engine.api import (
    admin,
    anthropic,
    datasets,
    evaluations,
    exports,
    generations,
    knowledge,
    openai,
    search,
    system,
    teachers,
    training,
    web,
)
from engine.api.deps import AuthError, is_anthropic
from engine.config import Settings
from engine.datasets.generate import GenerationManager, Generator
from engine.datasets.store import DatasetStore
from engine.evaluation.evaluate import EvaluationManager
from engine.export.jobs import ExportManager
from engine.models.bundles import bundled_collection, import_bundles_in_background
from engine.models.manager import ModelManager
from engine.models.transfer import ModelUploads
from engine.rag.embeddings import HashEmbedder, SentenceTransformerEmbedder
from engine.rag.store import KnowledgeStore
from engine.rag.transfer import ImportManager
from engine.teachers.store import TeacherStore
from engine.training.jobs import TrainingManager
from engine.web.crawl_settings import CrawlSettingsStore
from engine.web.crawler import Crawler, CrawlManager
from engine.web.search_settings import SearchSettingsStore


def open_knowledge(settings: Settings) -> KnowledgeStore:
    if settings.embedding_model == "hash":
        return KnowledgeStore(settings.knowledge_db, HashEmbedder())
    cache = settings.data_dir / "embeddings"
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model, cache, settings.hf_token, settings.embedding_device
    )
    return KnowledgeStore(settings.knowledge_db, embedder)


def crawler(settings: Settings, store: KnowledgeStore) -> Crawler:
    return Crawler(store, settings.proxy, crawl_settings(settings).get)


def crawl_settings(settings: Settings) -> CrawlSettingsStore:
    return CrawlSettingsStore(settings.workspace_dir / "crawl-settings.json", settings)


def search_settings(settings: Settings) -> SearchSettingsStore:
    return SearchSettingsStore(settings.workspace_dir / "search-settings.json", settings)


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
    app.state.imports = ImportManager(app.state.knowledge)
    app.state.crawl_settings = crawl_settings(settings)
    app.state.crawls = CrawlManager(crawler(settings, app.state.knowledge))
    # A new search client per request, so a changed setting or a missing key shows up at once.
    app.state.search_settings = search_settings(settings)
    app.state.datasets = DatasetStore(app.state.knowledge)
    app.state.teachers = TeacherStore(app.state.knowledge, app.state.models)
    app.state.models.teachers = app.state.teachers
    app.state.generations = GenerationManager(
        Generator(app.state.knowledge, app.state.datasets, app.state.teachers)
    )
    app.state.evaluations = EvaluationManager(
        settings.workspace_dir / "evaluations",
        app.state.datasets,
        app.state.models,
        app.state.teachers,
    )
    app.state.exports = ExportManager(
        settings.workspace_dir / "exports",
        settings.models_dir,
        settings.data_dir / "tools",
        settings.llama_cpp_dir,
        settings.threads,
        knowledge=app.state.knowledge,
        on_done=lambda name: import_bundles_in_background(app.state.models, app.state.knowledge),
    )
    app.state.model_uploads = ModelUploads(
        app.state.models,
        settings.workspace_dir / "uploads",
        collections=lambda: {c["name"] for c in app.state.knowledge.collections()},
        on_installed=lambda name: import_bundles_in_background(
            app.state.models, app.state.knowledge
        ),
    )
    # Collections that came with models (an export or a model folder copied in).
    if any(bundled_collection(p) for p in settings.models_dir.glob("*")):
        import_bundles_in_background(app.state.models, app.state.knowledge)
    app.state.training = TrainingManager(
        settings.workspace_dir / "training", settings.models_dir, app.state.datasets
    )

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
    app.include_router(datasets.router)
    app.include_router(training.router)
    app.include_router(teachers.router)
    app.include_router(generations.router)
    app.include_router(evaluations.router)
    app.include_router(exports.router)
    return app
