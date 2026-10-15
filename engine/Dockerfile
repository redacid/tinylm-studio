# The tinylm-studio engine as a container: the API on port 8765, all state under /workspace.
#
#   make image                 tinylm-engine:<version>-<commit>-cpu, and :cpu
#   make image VARIANT=cuda    tinylm-engine:<version>-<commit>-cuda, and :cuda
#
# make passes ENGINE_VERSION and GIT_COMMIT: they become the image's OCI labels, and the engine
# reports the commit in /health and its log. A plain docker build leaves the commit unknown.
#
# The CUDA variant runs llama.cpp (GGUF answers) and PyTorch (fine-tuning) on an NVIDIA GPU, for
# RunPod pods and GPU nodes; see docs/deployment.md. llama.cpp is compiled here rather than taken
# from prebuilt wheels: without -march=native, so the image runs on any x86-64 CPU, and for the
# common GPU generations (Volta to Hopper).
ARG VARIANT=cpu
ARG CUDA_VERSION=12.6.3

FROM ubuntu:24.04 AS build-cpu
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build-cuda
FROM ubuntu:24.04 AS runtime-cpu
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS runtime-cuda

# llama-cpp-python, built once into a wheel.
FROM build-${VARIANT} AS llama
ARG VARIANT
ARG LLAMA_CPP_PYTHON="llama-cpp-python>=0.3.0"
ARG CUDA_ARCHITECTURES="70;75;80;86;89;90"
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/build && /opt/build/bin/pip install --quiet --upgrade pip wheel
RUN CMAKE_ARGS="-DGGML_NATIVE=OFF"; \
    if [ "$VARIANT" = cuda ]; then \
        CMAKE_ARGS="$CMAKE_ARGS -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES \
            -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined"; \
    fi; \
    CMAKE_ARGS="$CMAKE_ARGS" /opt/build/bin/pip wheel --no-deps --no-binary llama-cpp-python \
        -w /wheels "$LLAMA_CPP_PYTHON"

FROM runtime-${VARIANT}
ARG VARIANT
# The engine version is the package's own; passing it only checks that it matches.
ARG ENGINE_VERSION=""
ARG GIT_COMMIT=""
# Where PyTorch comes from: CPU-only wheels, or CUDA ones for the cuda variant.
ARG TORCH_INDEX=""
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv libgomp1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/engine
ENV PATH=/opt/engine/bin:$PATH
COPY --from=llama /wheels /wheels
COPY pyproject.toml /src/engine/
COPY src /src/engine/src
RUN INDEX="$TORCH_INDEX"; \
    if [ -z "$INDEX" ]; then \
        if [ "$VARIANT" = cuda ]; then INDEX=https://download.pytorch.org/whl/cu126; \
        else INDEX=https://download.pytorch.org/whl/cpu; fi; \
    fi; \
    pip install --quiet --no-cache-dir /wheels/*.whl \
    && pip install --quiet --no-cache-dir --extra-index-url "$INDEX" /src/engine \
    && rm -rf /wheels /src \
    && python -c "import engine, llama_cpp, torch; print('engine', engine.__version__, 'llama.cpp', llama_cpp.__version__, 'torch', torch.__version__)" \
    && if [ -n "$ENGINE_VERSION" ]; then \
        python -c "import engine, sys; v = engine.__version__; sys.exit(None if v == '$ENGINE_VERSION' else f'engine is {v}, not $ENGINE_VERSION')"; \
    fi

LABEL org.opencontainers.image.title="tinylm-studio engine (${VARIANT})" \
      org.opencontainers.image.version="${ENGINE_VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}"

# Listen outside the container, keep everything on the volume mounted at /workspace (RunPod
# mounts its volumes there), and let the desktop app's renderer (origin "null" or "file://")
# call the API.
ENV TINYLM_HOST=0.0.0.0 \
    TINYLM_PORT=8765 \
    TINYLM_DATA_DIR=/workspace/tinylm-studio \
    TINYLM_CORS_ORIGINS='["null", "file://"]' \
    HF_HOME=/workspace/tinylm-studio/hf-cache \
    PYTHONUNBUFFERED=1 \
    TINYLM_BUILD_COMMIT=${GIT_COMMIT}
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fsS "http://127.0.0.1:${TINYLM_PORT}/health" || exit 1
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "-m", "engine", "serve"]
