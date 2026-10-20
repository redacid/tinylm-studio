"""Management API for web search: find pages and videos, and keep results as links."""

from typing import Literal

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from engine.api.deps import knowledge, require_api_key
from engine.api.schemas import (
    UNAUTHORIZED,
    PublicSearchSettings,
    Router,
    SearchOutcome,
    UploadResult,
    errors,
    items,
)
from engine.rag.store import CollectionNotFound, KnowledgeStore
from engine.web.search import SearchError, SearchKind, SearchResult, link_document, search
from engine.web.search_settings import Provider, SearchSettings, validation_message

router = Router(
    prefix="/api", dependencies=[Depends(require_api_key)], tags=["search"], responses=UNAUTHORIZED
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    kind: SearchKind = "web"
    limit: int = Field(10, ge=1, le=50)
    # Result language, such as "uk", "ru" or "en"; any language when empty.
    language: str | None = None
    # Domain filters for this search only; the saved ones (Settings > Search) when left out.
    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None


class Link(BaseModel):
    title: str
    url: str = Field(pattern=r"^https?://")
    snippet: str = ""
    kind: SearchKind = "web"


class LinksRequest(BaseModel):
    links: list[Link] = Field(min_length=1)


class SearchSettingsChanges(BaseModel, extra="allow"):
    """The settings to save; the others keep their value. API keys: a string sets it, ""
    removes it."""

    provider: Provider | None = None
    # A list, or one string with domains separated by commas or new lines.
    include_domains: list[str] | str | None = None
    exclude_domains: list[str] | str | None = None
    searxng_url: str | None = None
    brave_api_key: str | None = None
    tavily_api_key: str | None = None
    tavily_depth: Literal["basic", "advanced"] | None = None
    tavily_topic: Literal["general", "news"] | None = None
    exa_api_key: str | None = None
    exa_type: Literal["auto", "fast", "instant"] | None = None
    perplexity_api_key: str | None = None
    serper_api_key: str | None = None
    serper_country: str | None = None


@router.post("/search", response_model=SearchOutcome, responses=errors(422, 503))
def find(body: SearchRequest, request: Request) -> dict:
    """Finds pages or videos with the provider and domain filter of the search settings."""
    saved = request.app.state.search_settings.get()
    changes = body.model_dump(include={"include_domains", "exclude_domains"}, exclude_none=True)
    try:
        current = saved.model_copy(
            update=SearchSettings(**changes).model_dump(include=set(changes))
        )
    except ValidationError as e:
        raise HTTPException(422, validation_message(e)) from None
    try:
        provider = request.app.state.search_settings.provider(current)
        try:
            outcome = search(
                provider,
                body.query,
                body.kind,
                body.limit,
                body.language or None,
                current.domain_filter,
            )
        finally:
            provider.close()
    except SearchError as e:
        raise HTTPException(503, str(e)) from None
    return {
        "provider": provider.name,
        "data": [r.to_dict() for r in outcome.results],
        # Results dropped by the domain filter, and the queries sent (with `site:` when the
        # provider cannot filter by domain itself).
        "hidden": outcome.hidden,
        "queries": outcome.queries,
        "include_domains": current.include_domains,
        "exclude_domains": current.exclude_domains,
    }


@router.get("/search/settings", response_model=PublicSearchSettings)
def get_settings(request: Request) -> dict:
    return request.app.state.search_settings.get().public()


@router.put("/search/settings", response_model=PublicSearchSettings, responses=errors(422))
def put_settings(body: SearchSettingsChanges, request: Request) -> dict:
    """Saves the given fields only. API keys: a string sets it, "" removes it, left out keeps it."""
    changes = body.model_dump(exclude_unset=True)
    unknown = set(changes) - set(SearchSettings.model_fields)
    if unknown:
        raise HTTPException(422, f"Unknown settings: {', '.join(sorted(unknown))}")
    try:
        return request.app.state.search_settings.update(changes).public()
    except ValidationError as e:
        raise HTTPException(422, validation_message(e)) from None


@router.post("/collections/{name}/links", response_model=items(UploadResult), responses=errors(404))
def add_links(name: str, body: LinksRequest, store: KnowledgeStore = Depends(knowledge)) -> dict:
    """Keeps search results as small documents (title, address, description), so answers can
    cite them, for example trailers. To index a page's full text, crawl it instead."""
    results = []
    for link in body.links:
        title, text = link_document(SearchResult(**link.model_dump()))
        try:
            results.append(store.add_document(name, title, text, link.url))
        except CollectionNotFound:
            raise HTTPException(404, f"Unknown collection '{name}'") from None
    return {"data": results}
