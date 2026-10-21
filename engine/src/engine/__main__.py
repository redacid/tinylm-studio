"""Command-line entry point.

python -m engine [serve] [--host H] [--port P]   run the API server (default)
python -m engine download MODEL_ID               download a catalog model
python -m engine models                          list catalog and installed models
python -m engine bench [MODEL_ID ...]            benchmark installed models
python -m engine ingest COLLECTION PATH ...      add files or folders to a collection
python -m engine crawl COLLECTION URL ...        add pages of websites to a collection
python -m engine search QUERY [--add COLLECTION] find pages or videos (and add them);
                 [--include a.com,b.com] [--exclude c.com]   only / never these sites
python -m engine openapi [--output FILE]         write the API's OpenAPI schema (JSON)
"""

import argparse
import logging
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


def log_hardware() -> None:
    """What the engine runs on; in the background, as importing torch takes a few seconds."""
    from engine import hardware

    info = hardware.detect()
    gpu = "none, models run on the CPU"
    if "cuda" in info.accelerators:
        import torch

        gpu = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
    elif "mps" in info.accelerators:
        gpu = "Apple (MPS)"
    logging.getLogger("engine").info(
        "Hardware: %s, %d cores, %.1f GB memory; GPU: %s",
        info.cpu,
        info.cpu_cores,
        info.memory_gb,
        gpu,
    )


def serve(settings: Settings, args: argparse.Namespace) -> None:
    import os

    import uvicorn

    from engine import __version__, commit, logs
    from engine.engine_settings import with_saved
    from engine.main import create_app

    if os.environ.get("TINYLM_EXIT_WITH_PARENT"):
        exit_with_parent()
    settings.host, settings.port = args.host, args.port
    # create_app applies the saved engine settings itself; the log starts with their level.
    path = logs.setup(settings.data_dir, with_saved(settings).log_level)
    log = logging.getLogger("engine")
    log.info(
        "tinylm-studio engine %s, commit %s, Python %s",
        __version__,
        commit() or "unknown",
        sys.version.split()[0],
    )
    log.info(
        "Data in %s; log in %s; listening on %s:%d", settings.data_dir, path, args.host, args.port
    )
    log.info("API key: %s", "required (TINYLM_API_KEY)" if settings.api_key else "not required")
    if not settings.api_key and args.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("No TINYLM_API_KEY: anyone who can reach the port can use the engine")
    threading.Thread(target=log_hardware, daemon=True).start()
    # log_config=None: uvicorn logs through the handlers above.
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_config=None)


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
            # The full path, so the chat can link to the file.
            result = store.add_document(args.collection, title, doc.text, str(path.resolve()))
        except Exception as e:
            print(f"skipped  {path}: {e}")
            continue
        state = "same as an earlier file" if result["duplicate"] else f"{result['chunks']} chunks"
        print(f"added    {path} ({state})", flush=True)


def crawl(settings: Settings, args: argparse.Namespace) -> None:
    from engine.main import crawl_settings
    from engine.web.crawler import CrawlOptions

    defaults = crawl_settings(settings).get()
    options = CrawlOptions(
        args.urls,
        max_pages=args.max_pages or defaults.max_pages,
        max_depth=defaults.max_depth if args.max_depth is None else args.max_depth,
        per_minute=args.per_minute or defaults.per_minute,
        use_browser=defaults.use_browser if args.browser is None else args.browser,
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
    from pydantic import ValidationError

    from engine.main import crawl_settings, open_knowledge, search_settings
    from engine.web.crawler import CrawlOptions
    from engine.web.search import SearchError, link_document
    from engine.web.search import search as find
    from engine.web.search_settings import SearchSettings, validation_message

    kind = "video" if args.video else "web"
    store = search_settings(settings)
    changes = {}
    if args.include is not None:
        changes["include_domains"] = args.include
    if args.exclude is not None:
        changes["exclude_domains"] = args.exclude
    try:
        # Flags win over the saved settings for this search only.
        update = SearchSettings(**changes).model_dump(include=set(changes))
        current = store.get().model_copy(update=update)
        provider = store.provider(current)
        outcome = find(provider, args.query, kind, args.limit, args.language, current.domain_filter)
        provider.close()
    except SearchError as e:
        sys.exit(str(e))
    except ValidationError as e:
        sys.exit(validation_message(e))
    results = outcome.results
    if outcome.hidden:
        print(f"({outcome.hidden} results hidden by the domain filter)")
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
        per_minute = crawl_settings(settings).get().per_minute
        options = CrawlOptions(urls, max_pages=len(urls), max_depth=0, per_minute=per_minute)
        run_crawl(settings, args.add, options)


def openapi(_: Settings, args: argparse.Namespace) -> None:
    """The schema /openapi.json serves, without starting a server or touching the data folder:
    the app is made on an empty temporary one."""
    import json
    import tempfile
    from pathlib import Path

    from engine.main import create_app

    with tempfile.TemporaryDirectory(prefix="tinylm-openapi-") as folder:
        app = create_app(Settings(data_dir=Path(folder), embedding_model="hash"))
        schema = json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).write_text(schema)
    else:
        sys.stdout.write(schema)


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
    # Options left out come from the crawler settings (Settings > Crawler, or TINYLM_CRAWL_*).
    p.add_argument("--max-pages", type=int, help="new pages to download (earlier ones are skipped)")
    p.add_argument("--max-depth", type=int, help="link hops from a start address")
    p.add_argument("--per-minute", type=float, help="requests per minute to each site")
    p.add_argument(
        "--browser",
        action=argparse.BooleanOptionalAction,
        help="fetch through the browser at the CDP address (default: the crawler settings)",
    )
    p.add_argument(
        "--ignore-robots", action="store_true", help="do not read robots.txt for this crawl"
    )
    p.set_defaults(func=crawl)

    p = sub.add_parser("search", help="find pages or videos on the web (SearXNG or Brave)")
    p.add_argument("query")
    p.add_argument("--video", action="store_true", help="find videos (trailers) instead of pages")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--language", help="result language, such as uk, ru or en")
    p.add_argument(
        "--include",
        metavar="DOMAINS",
        help="only these sites or patterns like *.ua, comma-separated (default: saved settings)",
    )
    p.add_argument(
        "--exclude",
        metavar="DOMAINS",
        help="never these sites or patterns like *.ru, comma-separated (default: saved settings)",
    )
    p.add_argument(
        "--add", metavar="COLLECTION", help="add the found pages (or video links) to a collection"
    )
    p.set_defaults(func=search)

    p = sub.add_parser("openapi", help="write the API's OpenAPI schema (JSON)")
    p.add_argument("--output", help="the file to write (default: standard output)")
    p.set_defaults(func=openapi)

    args = parser.parse_args()
    func = getattr(args, "func", serve)
    if func not in (serve, openapi):
        # The commands use what was saved in the app's Settings > Engine options too.
        from engine.engine_settings import with_saved

        settings = with_saved(settings)
    func(settings, args)


if __name__ == "__main__":
    main()
