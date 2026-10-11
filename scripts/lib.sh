#!/usr/bin/env bash
# Shared helpers: resolve the Python interpreter used by the engine.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/tinylm-studio"
TINYLM_VENV="${TINYLM_VENV:-$DATA_DIR/venv}"

# TINYLM_PYTHON points to a user-owned environment; otherwise use the managed venv.
engine_python() {
    if [[ -n "${TINYLM_PYTHON:-}" ]]; then
        echo "$TINYLM_PYTHON"
    else
        echo "$TINYLM_VENV/bin/python"
    fi
}
