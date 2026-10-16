# The tinylm-studio engine as a container: the API on port 8765, all state under /workspace.
#
#   make image                 tinylm-engine:<version>-<commit>-cpu, and :cpu
#   make image VARIANT=cuda    tinylm-engine:<version>-<commit>-cuda, and :cuda
#
# make passes ENGINE_VERSION and GIT_COMMIT: they become the image's OCI labels, and the engine
# reports the commit in /health and its log. A plain docker build leaves the commit unknown.
#
# The CUDA variant runs llama.cpp (GGUF answers) and PyTorch (fine-tuning) on an NVIDIA GPU, for
# RunPod pods and GPU nodes; see docs/deployment.md. All of that is in the base image
# (Dockerfile.base); this one adds the engine's code, so a build takes a minute.

# The base: the system, Python, llama.cpp, PyTorch and the engine's dependencies, built by
# Dockerfile.base (make base). make image passes the base that matches the dependencies.
ARG BASE=tinylm-engine-base:cpu
FROM ${BASE}
ARG VARIANT=cpu
# The engine version is the package's own; passing it only checks that it matches.
ARG ENGINE_VERSION=""
ARG GIT_COMMIT=""
COPY pyproject.toml /src/engine/
COPY src /src/engine/src
COPY check-dependencies.py /usr/local/bin/check-dependencies.py
# --no-deps: the dependencies are in the base. The build fails if the base lacks one, e.g. one
# added to pyproject.toml without a new base.
RUN pip install --quiet --no-cache-dir --no-deps /src/engine \
    && rm -rf /src \
    && python -c "import engine; print('engine', engine.__version__)" \
    && python /usr/local/bin/check-dependencies.py \
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
