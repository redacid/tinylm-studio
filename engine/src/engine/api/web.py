"""Management API for crawling websites into a collection, and for saved websites."""

from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError, field_validator

from engine.api.deps import knowledge, require_api_key
from engine.rag.store import CollectionNotFound, KnowledgeStore
from engine.web.crawl_settings import CrawlSettings, CrawlSettingsStore
from engine.web.crawler import CrawlJob, CrawlManager, CrawlOptions
from engine.web.search_settings import validation_message
from engine.web.sites import Website, WebsiteStore

router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)], tags=["web"])


class CrawlRequest(BaseModel):
    urls: list[str] = Field(min_length=1)
    # Left out: the crawler settings (Settings > Crawler, or TINYLM_CRAWL_*) give them.
    # New pages to download; pages downloaded by earlier crawls are walked through for free.
    max_pages: int | None = Field(None, ge=1, le=100000)
    max_depth: int | None = Field(None, ge=0, le=100)
    # Requests per minute to each site.
    per_minute: float | None = Field(None, gt=0, le=120)
    use_browser: bool | None = None
    respect_robots: bool = True
    # Follow only links under the start addresses' folders, not the whole site.
    only_under_start: bool = False
    # Links never followed: a part of the address ("/tag/", "?sort="), or a pattern with * that
    # covers the whole address ("*/forum/*"). Letter case is ignored.
    exclude: list[str] = Field(default_factory=list, max_length=200)

    @field_validator("exclude")
    @classmethod
    def patterns(cls, patterns: list[str]) -> list[str]:
        found = []
        for text in patterns:
            text = text.strip()
            if len(text) > 300:
                raise ValueError(f"Pattern longer than 300 characters: {text[:40]}…")
            if text and text not in found:
                found.append(text)
        return found

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


def crawl_settings(request: Request) -> CrawlSettingsStore:
    return request.app.state.crawl_settings


def with_defaults(body: CrawlRequest, request: Request) -> dict:
    """The request's crawl options, with the crawler settings for those left out."""
    defaults = crawl_settings(request).get()
    fields = body.model_dump(exclude={"name"})
    for name in ("max_pages", "max_depth", "per_minute", "use_browser"):
        if fields[name] is None:
            fields[name] = getattr(defaults, name)
    return fields


def to_website(body: WebsiteRequest, request: Request) -> Website:
    name = body.name.strip() or urlsplit(body.urls[0]).hostname or body.urls[0]
    return Website(name=name, **with_defaults(body, request))


def options(website: Website) -> CrawlOptions:
    return CrawlOptions(
        website.urls,
        max_pages=website.max_pages,
        max_depth=website.max_depth,
        per_minute=website.per_minute,
        use_browser=website.use_browser,
        respect_robots=website.respect_robots,
        only_under_start=website.only_under_start,
        exclude=website.exclude,
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
    return manager.start(name, CrawlOptions(**with_defaults(body, request))).to_dict()


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


class CrawlSettingsChanges(BaseModel, extra="forbid"):
    respect_robots: bool | None = None
    per_minute: float | None = None
    max_pages: int | None = None
    max_depth: int | None = None
    use_browser: bool | None = None
    user_agent: str | None = None
    cdp_url: str | None = None


@router.get("/crawl/settings")
def get_crawl_settings(store: CrawlSettingsStore = Depends(crawl_settings)) -> CrawlSettings:
    return store.get()


@router.put("/crawl/settings")
def put_crawl_settings(
    body: CrawlSettingsChanges, store: CrawlSettingsStore = Depends(crawl_settings)
) -> CrawlSettings:
    """Changes the fields given; the others keep their value. Applies from the next crawl."""
    try:
        return store.update(body.model_dump(exclude_unset=True))
    except ValidationError as e:
        raise HTTPException(422, validation_message(e)) from None
