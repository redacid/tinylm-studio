"""Command-line entry point.

python -m engine [serve] [--host H] [--port P]   run the API server (default)
python -m engine download MODEL_ID               download a catalog model
python -m engine models                          list catalog and installed models
python -m engine bench [MODEL_ID ...]            benchmark installed models
python -m engine ingest COLLECTION PATH ...      add files or folders to a collection
"""

import argparse
import sys
import time

from engine.config import Settings


def serve(settings: Settings, args: argparse.Namespace) -> None:
    import uvicorn

    from engine.main import create_app

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

    args = parser.parse_args()
    getattr(args, "func", serve)(settings, args)


if __name__ == "__main__":
    main()
