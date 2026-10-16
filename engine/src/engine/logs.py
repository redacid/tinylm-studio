"""The engine's log: to the console (in color on a terminal) and to <data dir>/logs/engine.log.

Each line is "date time.ms LEVEL logger: message"; a traceback follows its line. The desktop app
shows the file's end (GET /api/logs), also for an engine in a RunPod pod. Successful GET
requests (the app polls status every few seconds) are logged at DEBUG, so the log keeps what
was changed and what failed; TINYLM_LOG_LEVEL=DEBUG shows them too.
"""

import logging
import logging.handlers
import os
import re
import sys
from collections import deque
from pathlib import Path

FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 5_000_000
BACKUPS = 3

COLORS = {"DEBUG": "2", "INFO": "36", "WARNING": "33", "ERROR": "31", "CRITICAL": "1;31"}


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        color = COLORS.get(record.levelname)
        return f"\033[{color}m{line}\033[0m" if color else line


class QuietPolling(logging.Filter):
    """Demotes access lines of successful GET requests to DEBUG."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        # uvicorn.access: (client, method, path, http version, status)
        if isinstance(args, tuple) and len(args) == 5 and args[1] == "GET" and args[4] < 400:
            record.levelno, record.levelname = logging.DEBUG, "DEBUG"
        return True


def log_file(data_dir: Path) -> Path:
    return data_dir / "logs" / "engine.log"


def setup(data_dir: Path, level: str = "INFO") -> Path:
    """Sends all logging (uvicorn's, warnings) to the console and the log file; returns it."""
    path = log_file(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    level_no = logging.getLevelName(level.upper())
    if not isinstance(level_no, int):
        level_no = logging.INFO

    file = logging.handlers.RotatingFileHandler(
        path, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8"
    )
    file.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
    console = logging.StreamHandler(sys.stderr)
    color = sys.stderr.isatty() and not os.environ.get("NO_COLOR")
    console.setFormatter((ColorFormatter if color else logging.Formatter)(FORMAT, DATE_FORMAT))

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in (file, console):
        handler.setLevel(level_no)
        root.addHandler(handler)
    # Loggers stay at DEBUG so a demoted access line still reaches the handlers, which decide.
    root.setLevel(logging.DEBUG)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
    logging.getLogger("uvicorn.access").addFilter(QuietPolling())
    # Libraries that are chatty at DEBUG.
    for name in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub", "asyncio"):
        logging.getLogger(name).setLevel(max(level_no, logging.INFO))
    logging.captureWarnings(True)
    return path


LINE = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d")


def tail(path: Path, lines: int) -> list[str]:
    """The last lines of the log; a traceback counts as part of its entry's line."""
    if not path.is_file():
        return []
    entries: deque[str] = deque(maxlen=lines)
    with path.open(encoding="utf-8", errors="replace") as f:
        # Only the end of a big file is read.
        size = path.stat().st_size
        if size > lines * 2000:
            f.seek(size - lines * 2000)
            f.readline()  # a partial line
        for line in f:
            line = line.rstrip("\n")
            if LINE.match(line) or not entries:
                entries.append(line)
            else:
                entries[-1] += "\n" + line
    return list(entries)
