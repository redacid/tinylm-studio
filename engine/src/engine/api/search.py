"""Management API for web search: find pages and videos, and keep results as links."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from engine.api.deps import knowledge, require_api_key
from engine.rag.store import CollectionNotFound, KnowledgeStore
from engine.web.search import SearchError, SearchKind, SearchResult, link_document, search
from engine.web.search_settings import SearchSettings, validation_message

router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)], tags=["search"])


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


@router.post("/search")
def find(body: SearchRequest, request: Request) -> dict:
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


@router.get("/search/settings")
def get_settings(request: Request) -> dict:
    return request.app.state.search_settings.get().public()


@router.put("/search/settings")
def put_settings(changes: dict, request: Request) -> dict:
    """Saves the given fields only. API keys: a string sets it, "" removes it, left out keeps it."""
    unknown = set(changes) - set(SearchSettings.model_fields)
    if unknown:
        raise HTTPException(422, f"Unknown settings: {', '.join(sorted(unknown))}")
    try:
        return request.app.state.search_settings.update(changes).public()
    except ValidationError as e:
        raise HTTPException(422, validation_message(e)) from None


@router.post("/collections/{name}/links")
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
