"""Management API for crawling websites into a collection, and for saved websites."""

from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from engine.api.deps import knowledge, require_api_key
from engine.rag.store import CollectionNotFound, KnowledgeStore
from engine.web.crawler import CrawlJob, CrawlManager, CrawlOptions
from engine.web.sites import Website, WebsiteStore

router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)], tags=["web"])


class CrawlRequest(BaseModel):
    urls: list[str] = Field(min_length=1)
    # New pages to download; pages downloaded by earlier crawls are walked through for free.
    max_pages: int = Field(50, ge=1, le=100000)
    max_depth: int = Field(2, ge=0, le=100)
    # Requests per minute to each site; defaults to TINYLM_CRAWL_PER_MINUTE.
    per_minute: float | None = Field(None, gt=0, le=120)
    use_browser: bool = False
    respect_robots: bool = True

    @field_validator("urls")
    @classmethod
    def web_addresses(cls, urls: list[str]) -> list[str]:
        urls = [u.strip() for u in urls if u.strip()]
        bad = [u for u in urls if not u.startswith(("http://", "https://"))]
        if bad:
            raise ValueError(f"Not a web address: {bad[0]}")
        if not urls:
            raise ValueError("Give at least one web address")
        return urls


class WebsiteRequest(CrawlRequest):
    name: str = ""  # defaults to the first address's host


def crawls(request: Request) -> CrawlManager:
    return request.app.state.crawls


def websites(request: Request) -> WebsiteStore:
    return request.app.state.crawls.crawler.pages


def per_minute(body: CrawlRequest, request: Request) -> float:
    return body.per_minute or request.app.state.settings.crawl_per_minute


def to_website(body: WebsiteRequest, request: Request) -> Website:
    fields = body.model_dump(exclude={"per_minute", "name"})
    name = body.name.strip() or urlsplit(body.urls[0]).hostname or body.urls[0]
    return Website(name=name, per_minute=per_minute(body, request), **fields)


def options(website: Website) -> CrawlOptions:
    return CrawlOptions(
        website.urls,
        max_pages=website.max_pages,
        max_depth=website.max_depth,
        per_minute=website.per_minute,
        use_browser=website.use_browser,
        respect_robots=website.respect_robots,
    )


def require_collection(store: KnowledgeStore, name: str) -> None:
    try:
        store.count_documents(name)
    except CollectionNotFound:
        raise HTTPException(404, f"Unknown collection '{name}'") from None


def require_website(sites: WebsiteStore, website_id: int) -> Website:
    website = sites.website(website_id)
    if website is None:
        raise HTTPException(404, f"No website {website_id}")
    return website


def require_job(manager: CrawlManager, job_id: str) -> CrawlJob:
    if job_id not in manager.jobs:
        raise HTTPException(404, f"No crawl {job_id}")
    return manager.jobs[job_id]


@router.post("/collections/{name}/crawl", status_code=202)
def start_crawl(
    name: str,
    body: CrawlRequest,
    request: Request,
    store: KnowledgeStore = Depends(knowledge),
    manager: CrawlManager = Depends(crawls),
) -> dict:
    """A one-off crawl that is not saved."""
    require_collection(store, name)
    crawl = CrawlOptions(
        **body.model_dump(exclude={"per_minute"}), per_minute=per_minute(body, request)
    )
    return manager.start(name, crawl).to_dict()


@router.get("/collections/{name}/websites")
def list_websites(
    name: str, store: KnowledgeStore = Depends(knowledge), sites: WebsiteStore = Depends(websites)
) -> dict:
    require_collection(store, name)
    return {"data": [w.to_dict() for w in sites.websites(name)]}


@router.post("/collections/{name}/websites", status_code=201)
def add_website(
    name: str,
    body: WebsiteRequest,
    request: Request,
    store: KnowledgeStore = Depends(knowledge),
    sites: WebsiteStore = Depends(websites),
) -> dict:
    require_collection(store, name)
    return sites.add_website(name, to_website(body, request)).to_dict()


@router.put("/websites/{website_id}")
def update_website(
    website_id: int,
    body: WebsiteRequest,
    request: Request,
    sites: WebsiteStore = Depends(websites),
) -> dict:
    require_website(sites, website_id)
    return sites.update_website(website_id, to_website(body, request)).to_dict()


@router.delete("/websites/{website_id}", status_code=204)
def delete_website(website_id: int, sites: WebsiteStore = Depends(websites)) -> None:
    if not sites.delete_website(website_id):
        raise HTTPException(404, f"No website {website_id}")


@router.post("/websites/{website_id}/crawl", status_code=202)
def crawl_website(
    website_id: int,
    sites: WebsiteStore = Depends(websites),
    manager: CrawlManager = Depends(crawls),
) -> dict:
    """Runs a saved website again: only pages not downloaded before count toward max_pages."""
    website = require_website(sites, website_id)
    running = [
        j for j in manager.jobs.values() if j.website_id == website_id and j.state == "running"
    ]
    if running:
        raise HTTPException(409, f"{website.name} is already being crawled")
    return manager.start(website.collection, options(website), website_id).to_dict()


@router.get("/crawls")
def list_crawls(manager: CrawlManager = Depends(crawls)) -> dict:
    return {"data": [job.to_dict() for job in manager.jobs.values()]}


@router.get("/crawls/{job_id}")
def get_crawl(job_id: str, manager: CrawlManager = Depends(crawls)) -> dict:
    return require_job(manager, job_id).to_dict()


@router.post("/crawls/{job_id}/cancel")
def cancel_crawl(job_id: str, manager: CrawlManager = Depends(crawls)) -> dict:
    job = require_job(manager, job_id)
    job.cancelled.set()
    return job.to_dict()
