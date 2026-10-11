"""Management API for web search: find pages and videos, and keep results as links."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from engine.api.deps import knowledge, require_api_key
from engine.rag.store import CollectionNotFound, KnowledgeStore
from engine.web.search import SearchError, SearchKind, SearchResult, link_document

router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)], tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    kind: SearchKind = "web"
    limit: int = Field(10, ge=1, le=50)
    # Result language, such as "uk", "ru" or "en"; any language when empty.
    language: str | None = None


class Link(BaseModel):
    title: str
    url: str = Field(pattern=r"^https?://")
    snippet: str = ""
    kind: SearchKind = "web"


class LinksRequest(BaseModel):
    links: list[Link] = Field(min_length=1)


@router.post("/search")
def search(body: SearchRequest, request: Request) -> dict:
    try:
        provider = request.app.state.searcher()
        try:
            results = provider.search(body.query, body.kind, body.limit, body.language or None)
        finally:
            provider.close()
    except SearchError as e:
        raise HTTPException(503, str(e)) from None
    return {"provider": provider.name, "data": [r.to_dict() for r in results]}


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
