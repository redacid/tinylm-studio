# tinylm-studio developer commands.
# Run `make help` to list targets. Variables can be overridden, e.g. `make run-engine PORT=9000`.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Local settings such as HF_TOKEN or TINYLM_* overrides (see .env.example). Variables from .env
# are exported, so the engine and the desktop app see them too.
-include .env
export $(shell sed -n 's/^[[:space:]]*\(export[[:space:]]\+\)\?\([A-Za-z_][A-Za-z0-9_]*\)=.*/\2/p' .env 2>/dev/null)

# Managed engine environment (same defaults as scripts/lib.sh).
DATA_DIR    ?= $(or $(XDG_DATA_HOME),$(HOME)/.local/share)/tinylm-studio
TINYLM_VENV ?= $(DATA_DIR)/venv
# Set TINYLM_PYTHON to use your own environment instead of the managed one.
PYTHON      := $(or $(TINYLM_PYTHON),$(TINYLM_VENV)/bin/python)

HOST ?= 127.0.0.1
PORT ?= 8765
# Model id for `make download` / `make bench` (see `make models`).
MODEL ?= gemma-3-4b-gguf
# Set ECHO=1 to offer the built-in "echo" model (no weights needed).
ECHO ?=

ENGINE_DIR  := engine
DESKTOP_DIR := desktop
DESKTOP_DEPS := $(DESKTOP_DIR)/node_modules/.package-lock.json

export TINYLM_VENV

##@ Setup

.PHONY: setup
setup: setup-engine setup-desktop ## Install everything needed to run the app

.PHONY: setup-engine
setup-engine: ## Create/update the engine virtual environment (runtime dependencies)
	bash scripts/setup-engine.sh

.PHONY: setup-dev
setup-dev: setup-desktop ## Install everything plus development tools (pytest, ruff)
	bash scripts/setup-engine.sh --dev

.PHONY: setup-desktop
setup-desktop: $(DESKTOP_DEPS) ## Install desktop (Electron) dependencies from the lockfile

# Reinstalled automatically whenever package-lock.json changes.
$(DESKTOP_DEPS): $(DESKTOP_DIR)/package-lock.json
	cd $(DESKTOP_DIR) && npm ci
	touch $@

##@ Build

.PHONY: build
build: build-desktop ## Build all components

.PHONY: build-desktop
build-desktop: $(DESKTOP_DEPS) ## Compile the desktop app TypeScript into desktop/dist
	cd $(DESKTOP_DIR) && npm run build

##@ Quality

.PHONY: lint
lint: lint-engine lint-desktop ## Run all linters and type checks

.PHONY: lint-engine
lint-engine: ## Lint the engine with ruff
	cd $(ENGINE_DIR) && $(PYTHON) -m ruff check .

.PHONY: lint-desktop
lint-desktop: $(DESKTOP_DEPS) ## Type-check the desktop app
	cd $(DESKTOP_DIR) && npm run typecheck

.PHONY: format
format: ## Auto-format and fix engine code with ruff
	cd $(ENGINE_DIR) && $(PYTHON) -m ruff check --fix . && $(PYTHON) -m ruff format .

.PHONY: test
test: ## Run engine tests
	cd $(ENGINE_DIR) && $(PYTHON) -m pytest -q

.PHONY: check
check: lint test ## Run linters and tests (use before committing)

##@ Run

.PHONY: run
run: build-desktop ## Start the desktop app (it sets up and starts the engine itself)
	cd $(DESKTOP_DIR) && npx electron .

.PHONY: run-engine
run-engine: ## Start only the engine on HOST:PORT (ECHO=1 adds the test "echo" model)
	$(if $(ECHO),TINYLM_ENABLE_ECHO=true) $(PYTHON) -m engine serve --host $(HOST) --port $(PORT)

.PHONY: health
health: ## Query a running engine's health and system info
	curl -fsS http://$(HOST):$(PORT)/health && echo
	curl -fsS http://$(HOST):$(PORT)/api/system && echo

.PHONY: info
info: ## Show which Python environment and paths are in use
	@echo "Python:       $(PYTHON)"
	@echo "Managed venv: $(TINYLM_VENV)"
	@echo "Data dir:     $(DATA_DIR)"
	@$(PYTHON) -c "import engine; print('Engine:     ', engine.__version__)" 2>/dev/null \
		|| echo "Engine:       not installed (run 'make setup-engine')"

##@ Models

.PHONY: models
models: ## List catalog and installed models
	$(PYTHON) -m engine models

.PHONY: download
download: ## Download a catalog model: make download MODEL=qwen3-1.7b-gguf
	$(PYTHON) -m engine download $(MODEL)

.PHONY: bench
bench: ## Benchmark installed models (all, or MODEL=...); report goes to the workspace
	$(PYTHON) -m engine bench $(if $(filter command line,$(origin MODEL)),$(MODEL))

##@ Data

.PHONY: ingest
ingest: ## Add files or folders to a collection: make ingest COLLECTION=games FILES=~/docs
	$(PYTHON) -m engine ingest $(COLLECTION) $(FILES)

.PHONY: crawl
crawl: ## Add website pages to a collection: make crawl COLLECTION=games URLS="https://..." [PAGES=50]
	$(PYTHON) -m engine crawl $(COLLECTION) $(URLS) $(if $(PAGES),--max-pages $(PAGES))

##@ Web search

SEARXNG_PORT ?= 8888
# Docker or Podman, whichever is installed.
CONTAINER ?= $(shell command -v docker 2>/dev/null || command -v podman 2>/dev/null)

.PHONY: searxng
searxng: ## Start a local SearXNG for web search on 127.0.0.1:SEARXNG_PORT (Docker or Podman)
	@test -n "$(CONTAINER)" || { echo "Install Docker or Podman, or set TINYLM_SEARXNG_URL to a running SearXNG."; exit 1; }
	mkdir -p "$(DATA_DIR)/searxng"
	test -f "$(DATA_DIR)/searxng/settings.yml" || cp scripts/searxng/settings.yml "$(DATA_DIR)/searxng/"
	-$(CONTAINER) rm -f tinylm-searxng >/dev/null 2>&1
	$(CONTAINER) run -d --name tinylm-searxng --restart unless-stopped \
		-p 127.0.0.1:$(SEARXNG_PORT):8080 -v "$(DATA_DIR)/searxng:/etc/searxng:Z" \
		-e GRANIAN_HOST=0.0.0.0 -e SEARXNG_SECRET=$$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n') \
		docker.io/searxng/searxng:latest
	@echo "SearXNG runs on http://127.0.0.1:$(SEARXNG_PORT) (TINYLM_SEARXNG_URL)"

.PHONY: searxng-stop
searxng-stop: ## Stop the local SearXNG
	$(CONTAINER) rm -f tinylm-searxng

.PHONY: search
search: ## Find pages on the web: make search Q="elden ring review" [VIDEO=1] [ADD=games] [INCLUDE=a.com,b.com] [EXCLUDE=c.com]
	$(PYTHON) -m engine search "$(Q)" $(if $(VIDEO),--video) $(if $(ADD),--add $(ADD)) $(if $(INCLUDE),--include $(INCLUDE)) $(if $(EXCLUDE),--exclude $(EXCLUDE))

##@ Browser for crawling

CDP_PORT ?= 9222

.PHONY: chrome
chrome: ## Start a headless Chrome for crawling with Browser (CDP) on 127.0.0.1:CDP_PORT [CDP_PROXY=socks5://host:port]
	@test -n "$(CONTAINER)" || { echo "Install Docker or Podman, or start Chrome yourself (see the user guide)."; exit 1; }
	-$(CONTAINER) rm -f tinylm-chrome >/dev/null 2>&1
	$(CONTAINER) run -d --name tinylm-chrome --restart unless-stopped --shm-size=1g \
		-p 127.0.0.1:$(CDP_PORT):9222 \
		docker.io/chromedp/headless-shell:stable $(if $(CDP_PROXY),--proxy-server=$(CDP_PROXY))
	@echo "Chrome's DevTools run on http://127.0.0.1:$(CDP_PORT) (Settings > Crawler > Browser address)"

.PHONY: chrome-stop
chrome-stop: ## Stop the headless Chrome
	$(CONTAINER) rm -f tinylm-chrome

##@ Cleanup

.PHONY: clean
clean: ## Remove build outputs and caches
	rm -rf $(DESKTOP_DIR)/dist
	find $(ENGINE_DIR) -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name '*.egg-info' \) -prune -exec rm -rf {} +

.PHONY: distclean
distclean: clean ## Also remove node_modules and the managed engine venv
	rm -rf $(DESKTOP_DIR)/node_modules
	@if [ -z "$(TINYLM_PYTHON)" ]; then rm -rf "$(TINYLM_VENV)"; \
		else echo "Keeping user-provided environment $(TINYLM_PYTHON)"; fi

##@ Help

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} \
		/^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)} \
		/^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
