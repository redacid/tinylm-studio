#!/usr/bin/env bash
# Create (or update) the engine's Python environment and install the engine into it.
# Usage: scripts/setup-engine.sh [--dev]
#
# TINYLM_TORCH selects the PyTorch build:
#   cpu (default)  CPU-only wheels from download.pytorch.org (small download)
#   auto           let uv detect the installed GPU driver (CUDA/ROCm)
#   cu126, cu128…  a specific CUDA build
#   pypi           default PyPI wheels (CUDA-enabled on Linux, large download)
#
# TINYLM_LLAMA selects the llama.cpp (llama-cpp-python) build:
#   cpu (default)  prebuilt CPU wheels, no compiler needed
#   source         rebuild from source, optimized for this CPU (-DGGML_NATIVE=ON);
#                  add GPU flags via CMAKE_ARGS, e.g. CMAKE_ARGS="-DGGML_CUDA=on"
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
LLAMA="${TINYLM_LLAMA:-cpu}"
LLAMA_ARGS=()
LLAMA_REBUILD=()
LLAMA_PIP_REBUILD=()
if [[ "$LLAMA" == "source" ]]; then
    # Force a rebuild even when a prebuilt wheel is already installed.
    export CMAKE_ARGS="-DGGML_NATIVE=ON ${CMAKE_ARGS:-}"
    LLAMA_REBUILD=(--reinstall-package llama-cpp-python --no-binary llama-cpp-python)
    LLAMA_PIP_REBUILD=(--force-reinstall --no-deps --no-binary llama-cpp-python llama-cpp-python)
else
    LLAMA_ARGS=(--extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/$LLAMA")
fi

echo "Installing engine$EXTRAS (PyTorch: $TORCH, llama.cpp: $LLAMA)"
if command -v uv >/dev/null 2>&1; then
    TORCH_ARGS=()
    [[ "$TORCH" != "pypi" ]] && TORCH_ARGS=(--torch-backend "$TORCH")
    # uv reuses a previously built wheel; drop it so CMAKE_ARGS take effect.
    [[ "$LLAMA" == "source" ]] && uv cache clean --quiet llama-cpp-python
    uv pip install --quiet --python "$PYTHON" \
        "${TORCH_ARGS[@]}" "${LLAMA_ARGS[@]}" "${LLAMA_REBUILD[@]}" -e "$TARGET"
else
    TORCH_ARGS=()
    case "$TORCH" in
        pypi|auto) ;;
        *) TORCH_ARGS=(--extra-index-url "https://download.pytorch.org/whl/$TORCH") ;;
    esac
    "$PYTHON" -m pip install --quiet --upgrade pip
    "$PYTHON" -m pip install --quiet "${TORCH_ARGS[@]}" "${LLAMA_ARGS[@]}" -e "$TARGET"
    [[ ${#LLAMA_PIP_REBUILD[@]} -gt 0 ]] && "$PYTHON" -m pip install --quiet "${LLAMA_PIP_REBUILD[@]}"
fi

"$PYTHON" -c "import engine; print('Engine', engine.__version__, 'ready in', __import__('sys').prefix)"
