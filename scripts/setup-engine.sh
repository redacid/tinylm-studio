#!/usr/bin/env bash
# Create (or update) the engine's Python environment and install the engine into it.
# Usage: scripts/setup-engine.sh [--dev]
#
# TINYLM_TORCH selects the PyTorch build:
#   cpu (default)  CPU-only wheels from download.pytorch.org (small download)
#   auto           let uv detect the installed GPU driver (CUDA/ROCm)
#   cu126, cu128…  a specific CUDA build
#   pypi           default PyPI wheels (CUDA-enabled on Linux, large download)
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

TORCH="${TINYLM_TORCH:-cpu}"
echo "Installing engine$EXTRAS (PyTorch build: $TORCH)"
if command -v uv >/dev/null 2>&1; then
    TORCH_ARGS=()
    [[ "$TORCH" != "pypi" ]] && TORCH_ARGS=(--torch-backend "$TORCH")
    uv pip install --quiet --python "$PYTHON" "${TORCH_ARGS[@]}" -e "$TARGET"
else
    TORCH_ARGS=()
    case "$TORCH" in
        pypi|auto) ;;
        *) TORCH_ARGS=(--extra-index-url "https://download.pytorch.org/whl/$TORCH") ;;
    esac
    "$PYTHON" -m pip install --quiet --upgrade pip
    "$PYTHON" -m pip install --quiet "${TORCH_ARGS[@]}" -e "$TARGET"
fi

"$PYTHON" -c "import engine; print('Engine', engine.__version__, 'ready in', __import__('sys').prefix)"
