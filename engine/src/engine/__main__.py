"""Command-line entry point.

python -m engine [serve] [--host H] [--port P]   run the API server (default)
python -m engine download MODEL_ID               download a catalog model
python -m engine models                          list catalog and installed models
python -m engine bench [MODEL_ID ...]            benchmark installed models
python -m engine ingest COLLECTION PATH ...      add files or folders to a collection
python -m engine crawl COLLECTION URL ...        add pages of websites to a collection
python -m engine search QUERY [--add COLLECTION] find pages or videos (and add them)
"""

import argparse
import sys
import threading
import time

from engine.config import Settings


def exit_with_parent() -> None:
    """Stops the engine when the process that started it (the desktop app) is gone, so a crashed
    or killed app does not leave an engine behind that keeps the knowledge base locked."""
    import os

    parent = os.getppid()

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(2)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def serve(settings: Settings, args: argparse.Namespace) -> None:
    import os

    import uvicorn

    from engine.main import create_app

    if os.environ.get("TINYLM_EXIT_WITH_PARENT"):
        exit_with_parent()
    settings.host, settings.port = args.host, args.port
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")


def download(settings: Settings, args: argparse.Namespace) -> None:
    from engine.models.manager import ModelManager

    manager = ModelManager(settings.models_dir, hf_token=settings.hf_token)
    job = manager.download(args.model_id)
    print(f"Downloading {args.model_id} into {settings.models_dir}…", flush=True)
    while job.state == "downloading":
        time.sleep(1)
    if job.state == "error":
        sys.exit(f"Download failed: {job.error}")
    print("Done.")


def models(settings: Settings, _: argparse.Namespace) -> None:
    from engine.models.manager import ModelManager

    manager = ModelManager(settings.models_dir)
    for entry in manager.catalog():
        mark = "installed" if entry["installed"] else ""
        print(f"{entry['id']:<24} {entry['params']:>6}  {mark:<9}  {entry['description']}")
    custom = set(manager.installed()) - {e["id"] for e in manager.catalog()}
    for model_id in sorted(custom):
        print(f"{model_id:<24} {'?':>6}  installed  (custom)")


def bench(settings: Settings, args: argparse.Namespace) -> None:
    from engine.bench import benchmark
    from engine.models.manager import ModelManager

    installed = ModelManager(settings.models_dir).installed()
    model_ids = args.model_ids or installed
    missing = [m for m in model_ids if m not in installed]
    if missing or not model_ids:
        sys.exit(
            f"Not installed: {', '.join(missing) or 'no models'}. Run: make download MODEL=..."
        )
    report = benchmark(
        settings.models_dir,
        model_ids,
        args.max_tokens,
        settings.workspace_dir / "benchmarks",
        {"threads": settings.threads, "repetition_penalty": settings.repetition_penalty},
    )
    print(f"Report: {report}")


def ingest(settings: Settings, args: argparse.Namespace) -> None:
    from pathlib import Path

    from engine.main import open_knowledge
    from engine.rag.convert import SUPPORTED_SUFFIXES, convert

    store = open_knowledge(settings)
    store.create_collection(args.collection)
    files = []
    for path in map(Path, args.paths):
        files += sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for path in files:
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            doc = convert(path.name, path.read_bytes())
            title = doc.title or path.name
            result = store.add_document(args.collection, title, doc.text, str(path))
        except Exception as e:
            print(f"skipped  {path}: {e}")
            continue
        state = "same as an earlier file" if result["duplicate"] else f"{result['chunks']} chunks"
        print(f"added    {path} ({state})", flush=True)


def crawl(settings: Settings, args: argparse.Namespace) -> None:
    from engine.web.crawler import CrawlOptions

    options = CrawlOptions(
        args.urls,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        per_minute=args.per_minute or settings.crawl_per_minute,
        use_browser=args.browser,
        respect_robots=not args.ignore_robots,
    )
    run_crawl(settings, args.collection, options)


def run_crawl(settings: Settings, collection: str, options) -> None:
    from engine.main import crawler, open_knowledge
    from engine.web.crawler import CrawlJob

    store = open_knowledge(settings)
    store.create_collection(collection)
    job = CrawlJob(collection, options)
    printed = 0
    worker = threading.Thread(target=crawler(settings, store).run, args=(job,))
    worker.start()
    try:
        while worker.is_alive():
            worker.join(0.5)
            new = min(job.notes - printed, len(job.log))
            for line in job.log[len(job.log) - new :]:
                print(line, flush=True)
            printed = job.notes
    except KeyboardInterrupt:
        job.cancelled.set()
        worker.join()
    print(
        f"{job.state}: {job.fetched} new pages fetched, {job.known} downloaded before, "
        f"{job.indexed} added, {job.duplicates} duplicates, {job.skipped} skipped, "
        f"{job.failed} failed"
    )
    if job.error:
        sys.exit(job.error)


def search(settings: Settings, args: argparse.Namespace) -> None:
    from engine.main import open_knowledge, searcher
    from engine.web.crawler import CrawlOptions
    from engine.web.search import SearchError, link_document

    kind = "video" if args.video else "web"
    try:
        provider = searcher(settings)
        results = provider.search(args.query, kind, args.limit, args.language)
        provider.close()
    except SearchError as e:
        sys.exit(str(e))
    if not results:
        print("Nothing found.")
    for i, result in enumerate(results, 1):
        print(f"{i:2}. {result.title}\n    {result.url}")
    if not args.add or not results:
        return
    if kind == "video":
        # Videos are kept as links (title, address, description) that answers can cite.
        store = open_knowledge(settings)
        store.create_collection(args.add)
        for result in results:
            store.add_document(args.add, *link_document(result), source=result.url)
        print(f"Added {len(results)} video links to {args.add}.")
    else:
        urls = [r.url for r in results]
        per_minute = settings.crawl_per_minute
        options = CrawlOptions(urls, max_pages=len(urls), max_depth=0, per_minute=per_minute)
        run_crawl(settings, args.add, options)


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(prog="python -m engine", description="tinylm-studio engine")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("serve", help="run the API server (default)")
    for target in (parser, p):
        target.add_argument("--host", default=settings.host)
        target.add_argument("--port", type=int, default=settings.port)
    p.set_defaults(func=serve)

    p = sub.add_parser("download", help="download a catalog model")
    p.add_argument("model_id")
    p.set_defaults(func=download)

    sub.add_parser("models", help="list catalog and installed models").set_defaults(func=models)

    p = sub.add_parser("bench", help="benchmark installed models")
    p.add_argument("model_ids", nargs="*")
    p.add_argument("--max-tokens", type=int, default=128)
    p.set_defaults(func=bench)

    p = sub.add_parser("ingest", help="add files or folders to a collection")
    p.add_argument("collection")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=ingest)

    p = sub.add_parser("crawl", help="add pages of websites to a collection")
    p.add_argument("collection")
    p.add_argument("urls", nargs="+")
    p.add_argument(
        "--max-pages", type=int, default=50, help="new pages to download (earlier ones are skipped)"
    )
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--per-minute", type=float, help="requests per minute to each site")
    p.add_argument("--browser", action="store_true", help="fetch through the browser at CDP_URL")
    p.add_argument("--ignore-robots", action="store_true", help="do not follow robots.txt")
    p.set_defaults(func=crawl)

    p = sub.add_parser("search", help="find pages or videos on the web (SearXNG or Brave)")
    p.add_argument("query")
    p.add_argument("--video", action="store_true", help="find videos (trailers) instead of pages")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--language", help="result language, such as uk, ru or en")
    p.add_argument(
        "--add", metavar="COLLECTION", help="add the found pages (or video links) to a collection"
    )
    p.set_defaults(func=search)

    args = parser.parse_args()
    getattr(args, "func", serve)(settings, args)


if __name__ == "__main__":
    main()
