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

# The engine's OpenAPI schema, as /openapi.json serves it, turned into TypeScript types for the
# desktop app's client (desktop/src/renderer/api.ts). Run after changing an engine endpoint; the
# generated file is committed, so building the app needs no Python.
API_TYPES := $(DESKTOP_DIR)/src/renderer/engine-api.ts

.PHONY: api-types
api-types: $(DESKTOP_DEPS) ## Regenerate the desktop app's engine API types from the OpenAPI schema
	@schema=$$(mktemp --suffix=.json) && \
		$(PYTHON) -m engine openapi --output $$schema && \
		(cd $(DESKTOP_DIR) && npx openapi-typescript $$schema --default-non-nullable false --output $(CURDIR)/$(API_TYPES)); \
		status=$$?; rm -f $$schema; exit $$status

.PHONY: openapi
openapi: ## Print the engine's OpenAPI schema (also served at /openapi.json)
	@$(PYTHON) -m engine openapi

# Installable packages of the app, written to desktop/out. A package carries the engine's source
# and installs it into its own Python environment on the first start (python3 is a dependency).
# An .rpm needs rpmbuild on this computer (apt install rpm).
PACKAGES ?= deb rpm AppImage tar.gz

.PHONY: package
package: build-desktop ## Build installable packages of the app (PACKAGES="deb rpm AppImage tar.gz")
	echo "$(GIT_COMMIT)" > $(DESKTOP_DIR)/dist/build-commit
	cd $(DESKTOP_DIR) && npx electron-builder --linux $(PACKAGES)
	@ls -1 $(DESKTOP_DIR)/out/tinylm-studio-*

.PHONY: deb rpm appimage tarball
deb: ## Build a .deb package (Debian, Ubuntu, Mint)
	$(MAKE) package PACKAGES=deb
rpm: ## Build an .rpm package (Fedora, RHEL, openSUSE); needs rpmbuild
	$(MAKE) package PACKAGES=rpm
appimage: ## Build an AppImage, which runs on most distributions without installing
	$(MAKE) package PACKAGES=AppImage
tarball: ## Build a .tar.gz of the app, to unpack anywhere
	$(MAKE) package PACKAGES=tar.gz

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

# A menu entry so the app can be started from the system menu and its windows are grouped under
# its icon in the taskbar (StartupWMClass is the window class Electron gives the app).
APPLICATIONS_DIR ?= $(or $(XDG_DATA_HOME),$(HOME)/.local/share)/applications

.PHONY: desktop-entry
desktop-entry: ## Add tinylm-studio to the Linux application menu, with its icon
	mkdir -p $(APPLICATIONS_DIR)
	printf '%s\n' '[Desktop Entry]' 'Type=Application' 'Name=tinylm-studio' \
		'Comment=Chat with, fine-tune and evaluate small language models' \
		'Exec=make -C "$(CURDIR)" run' 'Icon=$(CURDIR)/$(DESKTOP_DIR)/assets/icon.png' \
		'Terminal=false' 'Categories=Development;' 'StartupWMClass=tinylm-studio-desktop' \
		> $(APPLICATIONS_DIR)/tinylm-studio.desktop
	@echo "Added $(APPLICATIONS_DIR)/tinylm-studio.desktop; delete it to remove the entry."

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
	-$(CONTAINER) network create $(NETWORK) >/dev/null 2>&1
	$(CONTAINER) run -d --name tinylm-searxng --restart unless-stopped --network $(NETWORK) \
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
# The languages the browser asks sites for (Accept-Language, navigator.languages); headless
# Chrome sends none by default. The same as the crawler's plain HTTP requests.
CDP_LANG ?= uk,en
# The container network of the helpers and the engine image (make run-image): there they reach
# each other by name, e.g. http://tinylm-chrome:9222, as 127.0.0.1 in a container is itself.
NETWORK ?= tinylm

.PHONY: chrome
chrome: ## Start a headless Chrome for crawling with Browser (CDP) on 127.0.0.1:CDP_PORT [CDP_PROXY=socks5://host:port] [CDP_LANG=uk,en]
	@test -n "$(CONTAINER)" || { echo "Install Docker or Podman, or start Chrome yourself (see the user guide)."; exit 1; }
	-$(CONTAINER) rm -f tinylm-chrome >/dev/null 2>&1
	-$(CONTAINER) network create $(NETWORK) >/dev/null 2>&1
	$(CONTAINER) run -d --name tinylm-chrome --restart unless-stopped --shm-size=1g --network $(NETWORK) \
		-p 127.0.0.1:$(CDP_PORT):9222 \
		docker.io/chromedp/headless-shell:stable --accept-lang=$(CDP_LANG) $(if $(CDP_PROXY),--proxy-server=$(CDP_PROXY))
	@echo "Chrome's DevTools run on http://127.0.0.1:$(CDP_PORT) (Settings > Crawler > Browser address);"
	@echo "from the engine image (make run-image): http://tinylm-chrome:9222"

.PHONY: chrome-stop
chrome-stop: ## Stop the headless Chrome
	$(CONTAINER) rm -f tinylm-chrome

##@ Container image

# The engine image: VARIANT=cpu or cuda (NVIDIA GPUs, for RunPod). IMAGE names it, e.g.
# IMAGE=docker.io/<you>/tinylm-engine to push it to Docker Hub. Each build is tagged
# <engine version>-<commit>-<variant> (e.g. 0.1.0-5a0d901-cuda), which never changes, and
# <variant>, which moves to the latest build. Uncommitted changes add "-dirty" to the commit.
IMAGE   ?= tinylm-engine
## VARIANT cpu cuda
VARIANT ?= cpu
ENGINE_VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' $(ENGINE_DIR)/pyproject.toml)
GIT_COMMIT     := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)$(if $(shell git status --porcelain 2>/dev/null),-dirty)
IMAGE_TAG      := $(ENGINE_VERSION)-$(GIT_COMMIT)-$(VARIANT)
# Its base (engine/Dockerfile.base): llama.cpp, PyTorch and the dependencies, built once, then
# only when they change. Tagged <variant>-<hash of Dockerfile.base and the dependency list>.
BASE_IMAGE ?= $(IMAGE)-base
BASE_HASH  := $(shell { cat $(ENGINE_DIR)/Dockerfile.base; sed -n '/^dependencies = \[/,/^\]/p' $(ENGINE_DIR)/pyproject.toml; } | sha256sum | cut -c1-12)
BASE_TAG   := $(VARIANT)-$(BASE_HASH)

.PHONY: base
base: ## Build the base image BASE_IMAGE:VARIANT-<hash> (llama.cpp, PyTorch; long, cuda most) [VARIANT=cuda]
	@test -n "$(CONTAINER)" || { echo "Install Docker or Podman first."; exit 1; }
	$(CONTAINER) build -f $(ENGINE_DIR)/Dockerfile.base --build-arg VARIANT=$(VARIANT) \
		-t $(BASE_IMAGE):$(BASE_TAG) -t $(BASE_IMAGE):$(VARIANT) $(ENGINE_DIR)
	@echo "Built $(BASE_IMAGE):$(BASE_TAG)"

.PHONY: base-push
base-push: ## Push the base image, so that other computers build the engine image without building it
	$(CONTAINER) image inspect $(BASE_IMAGE):$(BASE_TAG) >/dev/null 2>&1 || $(MAKE) base
	$(CONTAINER) push $(BASE_IMAGE):$(BASE_TAG)
	$(CONTAINER) push $(BASE_IMAGE):$(VARIANT)

.PHONY: image
image: ## Build the engine image IMAGE:<version>-<commit>-VARIANT and IMAGE:VARIANT [VARIANT=cuda]
	@test -n "$(CONTAINER)" || { echo "Install Docker or Podman first."; exit 1; }
	@$(CONTAINER) image inspect $(BASE_IMAGE):$(BASE_TAG) >/dev/null 2>&1 \
		|| $(CONTAINER) pull $(BASE_IMAGE):$(BASE_TAG) >/dev/null 2>&1 \
		|| { echo "No base image $(BASE_IMAGE):$(BASE_TAG) (new dependencies?): building it first, which takes long."; \
			$(MAKE) base; }
	$(CONTAINER) build --build-arg BASE=$(BASE_IMAGE):$(BASE_TAG) --build-arg VARIANT=$(VARIANT) \
		--build-arg ENGINE_VERSION=$(ENGINE_VERSION) --build-arg GIT_COMMIT=$(GIT_COMMIT) \
		-t $(IMAGE):$(IMAGE_TAG) -t $(IMAGE):$(VARIANT) $(ENGINE_DIR)
	@echo "Built $(IMAGE):$(IMAGE_TAG) on $(BASE_IMAGE):$(BASE_TAG)"

.PHONY: image-push
image-push: ## Build and push both tags (commit first; ALLOW_DIRTY=1 to push uncommitted changes)
	@case "$(GIT_COMMIT)" in *-dirty|unknown) test -n "$(ALLOW_DIRTY)" || { \
		echo "Commit your changes first: the image tag names the commit ($(GIT_COMMIT))."; exit 1; };; esac
	$(MAKE) image
	$(CONTAINER) push $(IMAGE):$(IMAGE_TAG)
	$(CONTAINER) push $(IMAGE):$(VARIANT)
	@echo "Pushed $(IMAGE):$(IMAGE_TAG): use it as the Engine image in Settings > Engine > RunPod"

.PHONY: run-image
run-image: ## Run IMAGE:VARIANT on 127.0.0.1:PORT with the data in DATA_DIR/container (needs TINYLM_API_KEY)
	@test -n "$(TINYLM_API_KEY)" || { echo "Set TINYLM_API_KEY (in .env or the command line)."; exit 1; }
	mkdir -p $(DATA_DIR)/container
	-$(CONTAINER) network create $(NETWORK) >/dev/null 2>&1
	$(CONTAINER) run --rm -it --network $(NETWORK) -p 127.0.0.1:$(PORT):8765 \
		-v $(DATA_DIR)/container:/workspace/tinylm-studio \
		-e TINYLM_CDP_URL=http://tinylm-chrome:9222 -e TINYLM_SEARXNG_URL=http://tinylm-searxng:8080 \
		-e TINYLM_API_KEY $(if $(HF_TOKEN),-e HF_TOKEN) $(if $(filter cuda,$(VARIANT)),--gpus all) $(IMAGE):$(VARIANT)

##@ Cleanup

.PHONY: clean
clean: ## Remove build outputs and caches
	rm -rf $(DESKTOP_DIR)/dist $(DESKTOP_DIR)/out
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
