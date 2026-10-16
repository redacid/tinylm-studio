"""Fails when a dependency of the installed engine is missing: the engine image installs the
engine without its dependencies, which come from the base image (Dockerfile.base)."""

import re
import sys
from importlib.metadata import PackageNotFoundError, requires, version

missing = []
for requirement in requires("tinylm-engine") or []:
    if "extra ==" in requirement:
        continue
    name = re.split(r"[\s\[<>=!~;(]", requirement, maxsplit=1)[0]
    try:
        version(name)
    except PackageNotFoundError:
        missing.append(name)
if missing:
    sys.exit(f"Not in the base image: {', '.join(missing)}. Build a new base: make base")
