# Implementation Plan: Narrow-Domain Small Language Model Studio

Status: approved direction, in progress · Last updated: 2026-10-15

## 0. Decisions so far

| Topic | Decision |
|---|---|
| Engine language | Python (section 2) |
| Python environment | The engine creates and manages its own virtual environment automatically; the user can point it to their own environment instead (section 3.6) |
| Model APIs | The engine exposes both OpenAI-compatible and Anthropic-compatible APIs (section 3.3) |
| Languages | Ukrainian, Russian and English, in data and in answers |
| Platforms (stage 1) | Linux only: Debian-based distributions first, Alpine as a second target (section 3.7) |
| Minimum hardware | Intel Core i7-class CPU, 32 GB RAM, no GPU required |
| Reference scenario | Video game assistant (section 3.8) |
| Repository | Local git repository for now; remote added later |
| Engine deployment | A container in a Kubernetes cluster; the desktop app and API clients connect to it. Running it locally from the desktop app stays supported ([deployment.md](deployment.md)) |
| GPU in the cloud | **Decided (2026-10-15):** the same engine image (`cuda` variant) on a RunPod GPU pod, or the `cpu` one on a CPU pod, chosen from RunPod's catalog with prices, which the desktop app creates, starts and stops through RunPod's REST API; not RunPod Serverless, which keeps no state between requests ([deployment.md](deployment.md#runpod)) |

## 1. Goal

A desktop application for building small, narrowly specialized language models:

- Runs on an ordinary laptop/desktop **without a powerful GPU**; uses a GPU (CUDA, Apple Metal/MPS) when present to speed things up.
- Lets the user **upload arbitrary data** (documents, tables, notes), analyzes it and uses it when answering.
- Lets the user **fine-tune** a small model on that data and serve it.
- Exposes the model through **OpenAI-compatible and Anthropic-compatible APIs**, so existing clients and SDKs can talk to it unchanged.
- Has a simple, understandable project structure.

Two parts:

1. **Engine** – a local service with an HTTP API: data ingestion, retrieval, training, inference.
2. **Desktop** – an Electron UI that drives the engine.

## 2. Key decision: Rust or Python for the engine

### Options compared

| Criterion | Rust (candle / burn) | Python (PyTorch + Hugging Face) |
|---|---|---|
| Fine-tuning (LoRA/QLoRA, SFT) | Possible but mostly hand-built; no mature PEFT/TRL equivalent; few ready training recipes | Mature: `transformers`, `peft`, `trl`, `accelerate`; LoRA on a 0.5–1.5B model is a few dozen lines |
| Model coverage | Inference for many architectures; training support for new architectures lags | New open models are supported on release day |
| CPU / GPU acceleration | Good CPU perf; CUDA/Metal in candle, wgpu in burn | CPU, CUDA, ROCm, Apple MPS out of the box |
| Document parsing (PDF, DOCX, HTML, XLSX…) | Thin ecosystem | Rich ecosystem (`markitdown`, `pymupdf`, `pandas`, …) |
| Embeddings / RAG | Possible (candle + fastembed-rs) | Trivial (`sentence-transformers`, `fastembed`) |
| Fast CPU inference | Good | Use `llama.cpp` via `llama-cpp-python` (C++ core, GGUF, quantization) |
| Packaging size | Small single binary | Large (PyTorch CPU ≈ 200–300 MB, with CUDA ≈ 2 GB+) |
| Development speed / hiring / examples | Slower; fewer examples for training | Fastest; almost every tutorial and paper is in Python |

### Recommendation: Python for the engine

Training is the core of the product, and in 2026 the training ecosystem is still overwhelmingly Python. With Rust we would spend most of the effort re-implementing LoRA, optimizers, data collation, checkpointing and tokenizer edge cases instead of building the product. The heavy numeric work in Python runs in native code anyway (PyTorch kernels, llama.cpp), so the "Python is slow" argument does not apply here.

Concretely:

- **Python 3.12 + FastAPI** for the HTTP service, managed with **uv**.
- **PyTorch + transformers + peft + trl** for fine-tuning.
- **llama.cpp (`llama-cpp-python`)** for fast quantized CPU/GPU inference of trained models (GGUF).
- Rust stays an option for later, narrowly scoped work (e.g. a small inference-only sidecar) if packaging size or startup time becomes a real problem. The HTTP API boundary makes such a swap invisible to the UI.

## 3. Architecture

```
┌──────────────────────────── Desktop (Electron) ────────────────────────────┐
│  Renderer: React + TypeScript                                              │
│   Chat · Data sources · Datasets · Training · Models · Settings            │
│  Main process: starts/stops the engine, picks a free port, health checks   │
└───────────────▲────────────────────────────────────────────────────────────┘
                │ HTTP + SSE on 127.0.0.1 (generated TypeScript client)
┌───────────────┴──────────────────────── Engine (Python) ───────────────────┐
│  API layer (FastAPI)                                                        │
│   /v1/chat/completions, /v1/models, /v1/embeddings   (OpenAI-compatible)    │
│   /v1/messages                                       (Anthropic-compatible) │
│   /api/*  management: sources, datasets, jobs, models, settings             │
│                                                                             │
│  Core modules                                                               │
│   ingest   – parse files → normalized Markdown → chunks                     │
│   rag      – embeddings + hybrid search, prompt augmentation                │
│   datasets – build training sets (user-provided or generated from sources)  │
│   training – LoRA/QLoRA SFT jobs, progress events, checkpoints              │
│   inference– model runtime (transformers for dev, llama.cpp for prod)       │
│   registry – base models, adapters, merged/quantized exports                │
│   hardware – detect CPU/CUDA/MPS, choose dtype, batch size, threads         │
│                                                                             │
│  Storage: SQLite (sqlite-vec vectors + FTS5) + workspace folder (files)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 How "learning from uploaded data" works

Two complementary mechanisms, both driven by the same uploaded sources:

1. **Retrieval (RAG)** – available immediately, CPU-cheap. Uploaded files are parsed, chunked, embedded and indexed. At answer time the relevant chunks are added to the prompt. Best for facts that change or that must be quoted precisely.
2. **Fine-tuning (LoRA)** – changes the model's behavior: domain vocabulary, answer format, style, narrow tasks (classification, extraction, structured output). Training data comes from:
   - user-provided examples (JSONL / CSV of prompt → answer), or
   - pairs generated from the uploaded sources by a "teacher" model (see open question 1).

A small model plus RAG plus a LoRA adapter is the practical way to get a useful narrow assistant without a big GPU.

### 3.2 Models

- Default base models in the 0.3–1.7B range (candidates at time of writing: Qwen3 0.6B/1.7B, Gemma 3 270M/1B, SmolLM family). The final shortlist is picked in Phase 1 by quick benchmarks on the target hardware and languages.
- Training on CPU: LoRA on a ≤0.6B model with a few thousand short examples is feasible (hours, not days). GPU makes it 10–50× faster and enables 1–3B models with QLoRA.
- After training: merge the adapter, convert to GGUF, quantize (Q4_K_M / Q8_0) and serve through llama.cpp.

### 3.3 API compatibility

Requirement (confirmed by the user): the engine supports both APIs as first-class, equal protocols.


- **OpenAI**: `POST /v1/chat/completions` (incl. `stream: true` SSE), `GET /v1/models`, `POST /v1/embeddings`.
- **Anthropic**: `POST /v1/messages` (incl. streaming events `message_start`, `content_block_delta`, `message_stop`, …), `x-api-key` / `anthropic-version` headers accepted.
- Both translate into one internal request type (`ChatRequest`) handled by the same inference + RAG pipeline, so behaviour is identical across protocols.
- Model id selects base model / adapter / RAG collection, e.g. `support-bot` or `support-bot+docs`.
- Optional local API key; bound to `127.0.0.1` by default, LAN exposure is an explicit setting.
- Compatibility verified by contract tests that call the engine with the official `openai` and `anthropic` SDKs.
- Out of MVP scope: tool use / function calling, images, batch APIs.

### 3.4 Desktop app

- Electron + React + TypeScript + Vite; packaged with electron-builder.
- The main process launches the engine as a child process, waits for `/health`, and shuts it down on exit; or, as Settings > Engine says, connects to a remote engine (address and API key) or a RunPod pod it manages (`desktop/src/main/runpod.ts`).
- Screens: **Chat** (pick model, see cited sources), **Data** (drag-and-drop files and folders, ingestion status), **Datasets** (preview/edit examples), **Training** (configure, start, live loss chart, stop/resume), **Models** (list, export, delete, copy API endpoint snippet), **Settings** (hardware, API keys, storage path).
- The renderer only talks HTTP; it never touches Python directly. Its client (`desktop/src/renderer/api.ts`) takes the types of the engine's requests and responses from the engine's OpenAPI schema: `make api-types` generates `engine-api.ts` from it with openapi-typescript (see [api.md](api.md#openapi-schema)).
- Look: one stylesheet (`desktop/src/renderer/styles.css`) with color tokens for a light and a dark theme, picked by the system setting. Section titles, table headers, side lists and cards get their own colors so a screen does not read as one block of text; the main action of a form is a filled button, a confirmed delete a red one.
- Icons: glyphs of the Tabler Icons font (`@tabler/icons-webfont`, MIT), bundled by Vite, so they need no network. `desktop/src/renderer/Icon.tsx` lists the glyphs used and gives each a color by what it does; `icons.css` loads the font, and state badges get their icon from their class.
- Long operations: `Busy.tsx` has `useBusy()`, whose `run(label, work)` shows the label with a spinner while the work runs (and lets the work update it with its progress), and `<Spinner>`, which appears only after 250 ms so quick loads do not flicker. Screens use one for loading their lists and one for imports, exports and uploads.
- App icon: `desktop/assets/icon.svg` (a chat bubble shaped as a brain of two hemispheres, with a "t" and a spark), rendered to `icon.png` (512 px) for the window, the taskbar and the macOS Dock; the window header shows `icon.svg` next to the name (Vite keeps SVGs as files, since the CSP allows no data: images); `make desktop-entry` adds a Linux menu entry (`desktopName` in package.json and `StartupWMClass` tie the window to it).

### 3.5 Hardware acceleration

`hardware` module detects available devices at start-up and picks defaults: CPU (thread count, fp32/bf16), CUDA (fp16/bf16, optional 4-bit QLoRA), Apple MPS. The user can override in Settings. Packaging ships the CPU build by default; GPU-enabled PyTorch is an optional download from inside the app.

### 3.6 Python environment management

- On first start the desktop app (or `scripts/setup-engine.sh` for headless use) creates a dedicated virtual environment with `uv` and installs the engine and its dependencies into it.
- Default location: `${XDG_DATA_HOME:-~/.local/share}/tinylm-studio/venv`.
- The user can override it with the `TINYLM_PYTHON` environment variable or in Settings, pointing at the Python interpreter of their own environment. The engine checks that the environment has the required packages and reports what is missing instead of modifying a user-owned environment.
- Optional extras are installed on demand: `gpu` (CUDA PyTorch), `train` (peft, trl), `web` (crawler).

### 3.7 Platform notes

- **Debian/Ubuntu (glibc)** is the primary target: PyTorch, llama.cpp and Electron all ship prebuilt binaries for it.
- **Alpine (musl)** has no official PyTorch wheels and Electron's prebuilt binaries target glibc. Plan: support Alpine for the **engine** by building llama.cpp from source and running training inside a Debian-based container (Docker/Podman), or via `gcompat`; the desktop UI on Alpine uses the distribution's own `electron` package. Alpine is treated as a second target after the Debian build is complete.
- Sizing for an i7 + 32 GB RAM machine without GPU: inference of 1–3B quantized models is interactive; LoRA training of ≤1B models on a few thousand examples is practical; full 3B training is not.

### 3.8 Reference scenario: video game assistant

Used as the end-to-end test case for every phase.

1. **Sources:** the user provides a list of gaming websites, or the app discovers them via web search (open question 7).
2. **Collection:** a polite crawler (robots.txt, rate limits, caching) fetches articles, reviews and guides; pages go through the normal ingestion pipeline; each chunk keeps its source URL, game title and date.
3. **Knowledge:** the RAG index is organized per game; game titles and aliases (including Ukrainian/Russian names) are normalized to one entity.
4. **Tasks the model performs:**
   - write an original review of a game, synthesized from several sources, with citations;
   - find trailers (YouTube links from the crawled pages or via video search) and return them as links;
   - give walkthrough hints for a specific place or quest, spoiler-graded (hint first, full answer on request).
5. **Fine-tuning target:** answer format and tone (review structure, hint style), trained on examples generated from the sources.

This adds a `web` ingestion source (crawler + search connector) to Phase 2.

## 4. Project structure

```
tinylm-studio/
├── README.md               # what it is, how to run it in 3 commands
├── AGENTS.md / CLAUDE.md   # agent rules
├── docs/                   # architecture, API, decisions (ADRs)
├── engine/                 # Python service
│   ├── pyproject.toml      # uv-managed
│   ├── src/engine/
│   │   ├── main.py         # FastAPI app factory
│   │   ├── api/            # openai.py, anthropic.py, admin.py, schemas.py
│   │   ├── rag/            # file conversion, chunking, embeddings, vector store, grounding
│   │   ├── datasets/       # dataset building and validation
│   │   ├── training/       # LoRA jobs, job runner, progress events
│   │   ├── inference/      # runtimes: transformers, llama.cpp
│   │   ├── registry/       # models, adapters, exports
│   │   ├── hardware.py
│   │   └── storage.py      # SQLite + workspace paths
│   └── tests/
├── desktop/                # Electron app
│   ├── package.json
│   ├── src/main/           # process management, IPC, auto-update
│   ├── src/preload/
│   └── src/renderer/       # React UI
├── Makefile                # setup, build, test, run, clean (make help)
├── scripts/                # environment setup and release helpers
└── .github/workflows/      # lint, tests, builds
```

Two top-level code folders, one language each, one HTTP contract between them.

## 5. Milestones

Each phase ends with something usable.

### Phase 0 – Foundation (done)
- Repository, README, AGENTS.md, CI (ruff + pytest, eslint + vitest).
- Engine skeleton with `/health`, SQLite storage, workspace folder.
- Electron shell that starts the engine and shows its status.
- **Done when:** `make setup && make run` creates the engine environment, opens the app and shows "engine ready" on Debian-based Linux.

### Phase 1 – Local chat with compatible APIs (done)

Benchmark on the target machine (20 threads, 62 GB RAM, no GPU) with qwen3-0.6b through
PyTorch fp32 gave at most ~5 tokens/s (best at 4 threads), so the llama.cpp runtime with GGUF
models was moved forward from Phase 5 and is now the default inference path. PyTorch stays the
runtime for Hugging Face-format models and for fine-tuning.
- Inference runtime (transformers first) with a default small base model; model download manager.
- OpenAI- and Anthropic-compatible endpoints with streaming.
- Chat screen in the UI.
- Benchmark 3–4 candidate base models on CPU for speed and quality in the target languages.
- **Done when:** the official `openai` and `anthropic` Python SDKs pointed at `http://127.0.0.1:<port>` get streamed answers; chat works in the UI on CPU.

### Phase 2 – Data upload and RAG (done)

Done: web search for finding sites and trailers (local SearXNG by default, Brave Search API as an
option; pages are crawled or saved as websites, videos kept as citable links); website crawler
(same-site link following, per-site rate limit, robots.txt, Retry-After, HTTP/SOCKS proxy,
browser over CDP); hybrid search (vectors + FTS5 full text); saved websites and resumable crawls
that download only new pages; document paging and search; crawler settings kept by the engine
(robots.txt on or off, rate, defaults for new websites, user agent, browser address).

Done before: file upload and folder import, multilingual-e5-small embeddings, sqlite-vec collections,
grounded answers with sources through both APIs, Data screen and cited answers in the chat.
- Ingestion for TXT/MD, PDF, DOCX, HTML, CSV/XLSX, JSON (via `markitdown` + specific parsers); folder import; deduplication.
- Web source: crawler for a user-supplied site list and a search connector (reference scenario).
- Embeddings with a small multilingual model; sqlite-vec index; collections.
- Answers grounded in retrieved chunks with source citations.
- Data screen in the UI.
- **Done when:** a user drops a folder of documents, or a list of gaming sites, and gets cited answers about them through both the UI and the APIs.
- **Done (2026-10-18):** the SQL tab: read-only SQL over the data (collections, documents, chunks, the full-text index, websites, crawled pages) with the tables' fields, types and example queries. SQLite's authorizer on a read-only connection allows only reading those tables, so writing, `PRAGMA`, `ATTACH` and the teachers' API keys stay out of reach; queries stop after 15 s. `DELETE FROM` a data table counts its rows first and deletes them once confirmed, through the knowledge store, so search vectors and the full-text index stay in step.

### Phase 3 – Fine-tuning (in progress)

Done: datasets (JSONL/JSON/CSV import in OpenAI, ShareGPT, Alpaca and prompt/completion shapes,
validation with line numbers, a stable train/held-out split, search, editing, JSONL export) and the
Datasets screen; LoRA training with transformers + peft in a separate worker process, only the assistant
answers learned, checkpoints, stop and resume, progress as server-sent events; fine-tuned models
served as base + adapter (merged on load) in chat and through both APIs; Training screen with a
live loss chart; fine-tuned models listed on the Models screen. The Models screen also lists
every installed model with its size, deletes models, searches Hugging Face and downloads any
GGUF file or PyTorch model, and keeps per-model loading and answer settings. Optionally a
model has its own collection (used when a request names none), an export takes the collection
along in the model folder, and generated examples can carry sources in the prompt so the model
learns to answer from them.

Measured in the cloud on 4 cores with a randomly initialized model of Qwen3-0.6B's size (fp32,
LoRA rank 16, examples of ~390 tokens): batches of 4 took 18 s per step and 14 GB of RAM;
1 example × 4 accumulated steps took 15 s per step and 6.3 GB, so that is the default. At that
rate 1,000 examples × 3 epochs (750 steps) take about 3 hours on 4 cores. Still to do: a real run of
qwen3-0.6b on ~1,000 examples on the target machine, and a GPU run.
- Dataset import (JSONL/CSV), validation, train/eval split, preview/edit in UI.
- LoRA SFT job runner (one job at a time, background process), progress via SSE, cancel/resume from checkpoint.
- Model registry: base + adapter; serving adapters through the same APIs.
- Training screen with live loss chart.
- **Done when:** a LoRA fine-tune of a ≤0.6B model on ~1,000 examples completes on CPU and on GPU, and the tuned model is selectable in chat and via the API.

### Phase 4 – Dataset generation and evaluation
- **Done (2026-10-12):** generate Q&A pairs from a collection's chunks with a teacher model: a local installed model, any OpenAI-compatible API (OpenRouter with free models by default, OpenAI, Groq, Ollama, LM Studio) or Anthropic. Pairs are filtered by shape, length and language and checked against the text by the teacher; an unanswerable question per chunk teaches honest "no information" answers; the dataset's own examples steer the style. Each chunk is used once per dataset, so runs continue where the last one ended. Rate limits are paced (requests per minute) and waited out (429 + `Retry-After`). Examples carry their origin (manual, imported, generated) and source document.
- **Done (2026-10-18):** proof of concept of subscription teachers: a `cli` teacher runs Claude Code, Codex CLI or Gemini CLI headless (no tools, an empty folder, API keys removed from its environment), so a Claude, ChatGPT or Google subscription writes examples instead of an API key. The tool keeps its own login; the studio never handles OAuth tokens, which the providers do not allow third-party apps to use. The engine image has the three tools, with their logins on the volume. Settings › Subscriptions logs them in from the app (`/api/cli-tools`: `claude auth login`, `codex login --device-auth`, Gemini CLI in a pseudo-terminal), into the engine's own folder.
- **Done (2026-10-12):** evaluation. Up to four models answer a dataset's held-out examples; answers are scored by chrF similarity and, optionally, by a teacher acting as judge (1–5 with a reason, all answers to a question judged together under shuffled labels). Reports are kept per run and shown side by side on the Evaluation screen.
- **Done when:** a user can go from "folder of documents" to "fine-tuned model with a score report" without writing any data by hand.

### Phase 5 – Performance and distribution
- **Done (2026-10-12):** merge + GGUF export + quantization (Q8_0, Q4_K_M, F16) of fine-tuned models with llama.cpp's own converter (from the llama-cpp-python source package, so versions match) and quantizer; exported models are served through llama.cpp.
- Hardware auto-tuning; optional GPU PyTorch download.
- **Done (2026-10-21):** Settings › Engine options: the engine settings that only `.env` held
  (Hugging Face token, proxy, threads, repetition penalty, sources per answer, embedding device,
  llama.cpp folder, log level) are changed in the app, kept by the engine in
  `workspace/engine-settings.json` and put into effect without a restart (`/api/settings`); the
  token is never sent back and the proxy's password is masked.
- **Done (2026-10-17):** Linux packages of the app (deb, rpm, AppImage, tar.gz) built with
  electron-builder by `make package`. A package carries the engine's source and setup script as
  resources and, on the first start, installs a copy of the engine into its own environment
  (`~/.local/share/tinylm-studio/app-venv`), again after an update (a stamp names the build).
- **Done (2026-10-20):** releases. `make release VERSION=x.y.z` sets the version in the engine
  and the app, adds the commits since the previous tag to CHANGELOG.md, commits and tags; pushing
  the tag runs GitHub Actions, which test, build the deb and rpm and publish a GitHub release.
  Engine images are built for each tag too, and by hand. Builds at a release tag are named by the tag,
  others by the commit ([releasing.md](releasing.md)).
- Installers for macOS and Windows (dmg, exe), code signing, auto-update.
- **Done (2026-10-15):** a container image for the engine (CPU and CUDA), the desktop app
  connecting to a remote engine, and RunPod pods managed from Settings > Engine.
- Kubernetes: manifests or a Helm chart, CPU and memory limits read from cgroups (see
  [deployment.md](deployment.md#not-done-yet)).
- **Done (2026-10-20):** `make demo-gif` records a walkthrough GIF of the app (website to trained
  model, captions in English and Ukrainian) from a YAML scenario, driving the real app with
  Playwright ([demo.md](demo.md)); `make demo-tour` records a tour of every tab and its options, `make demo-api` the public API in Swagger UI.
- **Done when:** a non-developer installs the app from a release and completes the full flow.

## 6. Risks

| Risk | Mitigation |
|---|---|
| CPU training too slow for useful models | Keep models ≤0.6B by default, LoRA only, short sequences; rely on RAG for knowledge, fine-tuning for behavior |
| Small models weak in Ukrainian/Russian | Benchmark languages in Phase 1; prefer multilingual base and embedding models |
| Large installer because of PyTorch | CPU-only build by default; GPU runtime as optional download; Rust inference sidecar as a later option |
| API drift in OpenAI/Anthropic specs | Contract tests with official SDKs pinned in CI |
| Alpine lacks PyTorch/Electron binaries | Debian first; Alpine engine via container or source builds (section 3.7) |
| Crawled content and copyright | Store sources, generate original text with citations, respect robots.txt |
| "Any data" is unbounded | Start with text formats; OCR, images and audio are later, pluggable parsers |

## 7. Open questions

1. ~~Meaning of "supports Anthropic and OpenAI APIs".~~ **Decided (2026-10-11):** the model service must expose both OpenAI-compatible and Anthropic-compatible APIs (section 3.3). **Decided (2026-10-12):** teachers for dataset generation can be local models, Anthropic, or any OpenAI-compatible provider (OpenRouter with its free models first); external providers are opt-in, configured by the user with their own key.
2. ~~Is Rust a hard requirement?~~ **Decided:** Python, with an automatically managed, user-overridable virtual environment.
3. ~~Languages~~ **Decided:** Ukrainian, Russian and English. Base and embedding models must be multilingual.
4. ~~Platforms and hardware~~ **Decided:** Linux (Debian-based first, Alpine second), i7-class CPU with 32 GB RAM.
5. ~~Reference use case~~ **Decided:** video game assistant (section 3.8).
6. ~~Repository~~ **Decided:** local git repository `tinylm-studio` for now; a remote is added later.
7. ~~**Web search provider**~~ Decided: self-hosted SearXNG by default (free, no key; `make searxng`), Brave Search API as an option, plus user-supplied site lists. **Extended (2026-10-13):** Tavily, Exa, Perplexity and Serper as further providers, chosen with their keys and site filters (include/exclude domains) in Settings > Search; the engine keeps these settings.
