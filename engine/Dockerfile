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
# The command-line tools of subscriptions, for cli teachers: Claude Code, Codex CLI and Gemini
# CLI, on their own Node.js. Before the engine's code, so this layer is reused across builds.
# Their logins are kept on the volume (see the ENV below); log in with docker exec, see
# docs/deployment.md.
ARG NODE_VERSION=22.23.3
ARG CLAUDE_CODE_VERSION=2.1.302
ARG CODEX_VERSION=0.163.0
ARG GEMINI_CLI_VERSION=0.64.0
RUN set -e; \
    arch=$(uname -m | sed 's/x86_64/x64/; s/aarch64/arm64/'); \
    curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${arch}.tar.gz" \
        | tar -xz -C /opt; \
    mv /opt/node-v${NODE_VERSION}-linux-${arch} /opt/node; \
    export PATH=/opt/node/bin:$PATH; \
    npm install --global --no-fund --no-audit --no-update-notifier --loglevel=error \
        @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION} \
        @openai/codex@${CODEX_VERSION} \
        @google/gemini-cli@${GEMINI_CLI_VERSION}; \
    npm cache clean --force; \
    echo "claude $(claude --version)"; codex --version; echo "gemini $(gemini --version)"
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
# mounts its volumes there), the logins of the command-line tools included, and let the desktop
# app's renderer (origin "null" or "file://") call the API. NO_BROWSER: Gemini CLI prints its
# login link instead of opening a browser; DISABLE_AUTOUPDATER keeps Claude Code at its version.
ENV TINYLM_HOST=0.0.0.0 \
    TINYLM_PORT=8765 \
    TINYLM_DATA_DIR=/workspace/tinylm-studio \
    TINYLM_CORS_ORIGINS='["null", "file://"]' \
    HF_HOME=/workspace/tinylm-studio/hf-cache \
    PYTHONUNBUFFERED=1 \
    TINYLM_BUILD_COMMIT=${GIT_COMMIT} \
    PATH=/opt/node/bin:$PATH \
    CLAUDE_CONFIG_DIR=/workspace/tinylm-studio/cli-logins/claude \
    CODEX_HOME=/workspace/tinylm-studio/cli-logins/codex \
    GEMINI_CLI_HOME=/workspace/tinylm-studio/cli-logins/gemini \
    NO_BROWSER=true \
    DISABLE_AUTOUPDATER=1
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fsS "http://127.0.0.1:${TINYLM_PORT}/health" || exit 1
# --chmod: the script runs even when the checkout lost its executable bit.
COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "-m", "engine", "serve"]
