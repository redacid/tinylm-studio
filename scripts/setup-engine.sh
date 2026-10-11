#!/usr/bin/env bash
# Create (or update) the engine's Python environment and install the engine into it.
# Usage: scripts/setup-engine.sh [--dev]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

EXTRAS=""
[[ "${1:-}" == "--dev" ]] && EXTRAS="[dev]"
TARGET="$REPO_ROOT/engine$EXTRAS"

if [[ -n "${TINYLM_PYTHON:-}" ]]; then
    echo "Using user-provided environment: $TINYLM_PYTHON"
    PYTHON="$TINYLM_PYTHON"
else
    if [[ ! -x "$TINYLM_VENV/bin/python" ]]; then
        echo "Creating managed environment at $TINYLM_VENV"
        mkdir -p "$(dirname "$TINYLM_VENV")"
        if command -v uv >/dev/null 2>&1; then
            uv venv --quiet "$TINYLM_VENV"
        else
            python3 -m venv "$TINYLM_VENV"
        fi
    fi
    PYTHON="$TINYLM_VENV/bin/python"
fi

echo "Installing engine$EXTRAS"
if command -v uv >/dev/null 2>&1; then
    uv pip install --quiet --python "$PYTHON" -e "$TARGET"
else
    "$PYTHON" -m pip install --quiet --upgrade pip
    "$PYTHON" -m pip install --quiet -e "$TARGET"
fi

"$PYTHON" -c "import engine; print('Engine', engine.__version__, 'ready in', __import__('sys').prefix)"
