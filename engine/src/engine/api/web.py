"""Management API for crawling websites into a collection."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from engine.api.deps import knowledge, require_api_key
from engine.rag.store import CollectionNotFound, KnowledgeStore
from engine.web.crawler import CrawlManager, CrawlOptions

router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)], tags=["web"])


class CrawlRequest(BaseModel):
    urls: list[str] = Field(min_length=1)
    max_pages: int = Field(50, ge=1, le=10000)
    max_depth: int = Field(2, ge=0, le=10)
    # Requests per minute to each site; defaults to TINYLM_CRAWL_PER_MINUTE.
    per_minute: float | None = Field(None, gt=0, le=120)
    use_browser: bool = False
    respect_robots: bool = True


def crawls(request: Request) -> CrawlManager:
    return request.app.state.crawls


@router.post("/collections/{name}/crawl", status_code=202)
def start_crawl(
    name: str,
    body: CrawlRequest,
    request: Request,
    store: KnowledgeStore = Depends(knowledge),
    manager: CrawlManager = Depends(crawls),
) -> dict:
    try:
        store.documents(name)
    except CollectionNotFound:
        raise HTTPException(404, f"Unknown collection '{name}'") from None
    bad = [u for u in body.urls if not u.startswith(("http://", "https://"))]
    if bad:
        raise HTTPException(422, f"Not a web address: {bad[0]}")
    per_minute = body.per_minute or request.app.state.settings.crawl_per_minute
    options = CrawlOptions(**body.model_dump(exclude={"per_minute"}), per_minute=per_minute)
    return manager.start(name, options).to_dict()


@router.get("/crawls")
def list_crawls(manager: CrawlManager = Depends(crawls)) -> dict:
    return {"data": [job.to_dict() for job in manager.jobs.values()]}


@router.get("/crawls/{job_id}")
def get_crawl(job_id: str, manager: CrawlManager = Depends(crawls)) -> dict:
    if job_id not in manager.jobs:
        raise HTTPException(404, f"No crawl {job_id}")
    return manager.jobs[job_id].to_dict()


@router.post("/crawls/{job_id}/cancel")
def cancel_crawl(job_id: str, manager: CrawlManager = Depends(crawls)) -> dict:
    if job_id not in manager.jobs:
        raise HTTPException(404, f"No crawl {job_id}")
    job = manager.jobs[job_id]
    job.cancelled.set()
    return job.to_dict()
