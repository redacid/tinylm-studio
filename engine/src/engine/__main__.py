"""Command-line entry point: python -m engine --port 8765."""

import argparse

import uvicorn

from engine.config import Settings
from engine.main import create_app


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Run the tinylm-studio engine.")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    args = parser.parse_args()

    settings.host, settings.port = args.host, args.port
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
