"""tinylm-studio engine."""

import functools
import os
import subprocess
from pathlib import Path

__version__ = "0.1.0"


@functools.cache
def commit() -> str | None:
    """The git commit the engine was built from: TINYLM_BUILD_COMMIT, set in the container image,
    or the checkout the engine runs from (with "-dirty" when it has uncommitted changes); None
    when neither is known."""
    built = os.environ.get("TINYLM_BUILD_COMMIT", "").strip()
    if built:
        return built
    here = Path(__file__).resolve().parent
    try:
        run = functools.partial(
            subprocess.run, cwd=here, capture_output=True, text=True, timeout=5, check=True
        )
        # Only this repository's own checkout: not a copy installed in some other repository.
        run(["git", "ls-files", "--error-unmatch", "__init__.py"])
        sha = run(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
        dirty = run(["git", "status", "--porcelain", "--", str(here.parent.parent)]).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{sha}-dirty" if sha and dirty else sha or None
