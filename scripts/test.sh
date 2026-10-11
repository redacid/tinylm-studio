#!/usr/bin/env bash
# Run engine linting and tests in the engine environment.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

bash "$REPO_ROOT/scripts/setup-engine.sh" --dev
PYTHON="$(engine_python)"
cd "$REPO_ROOT/engine"
"$PYTHON" -m ruff check .
"$PYTHON" -m pytest -q
