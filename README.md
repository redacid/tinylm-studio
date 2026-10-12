# tinylm-studio

A desktop studio for building small, narrowly specialized language models that run without a powerful GPU.

- **engine/** – Python service: data ingestion, retrieval, training and inference, exposed through OpenAI- and Anthropic-compatible APIs.
- **desktop/** – Electron app that starts the engine and drives it over HTTP.
- **docs/** – architecture and the [implementation plan](docs/implementation-plan.md).

## Requirements

- Linux (Debian-based; Alpine support is planned, see the plan)
- Python 3.11+ and [uv](https://docs.astral.sh/uv/) (falls back to `python3 -m venv` if uv is missing)
- Node.js 20+

## Quick start

```bash
make setup   # create the engine environment and install desktop dependencies
make run     # start the desktop app (it starts the engine)
```

Download a model (Models tab in the app, or):

```bash
make download MODEL=gemma-3-4b-gguf
```

For gated models and faster downloads, put a Hugging Face token into `.env`
(`cp .env.example .env`); every `make` target passes it on.

Add your own data (Data tab in the app, or a whole folder from the command line), then pick the
collection in the chat to get answers that cite it:

```bash
make ingest COLLECTION=games FILES=~/Documents/games
make crawl COLLECTION=games URLS="https://example.com/reviews/"  # run again for the next new pages
```

To find sites and trailers instead of typing addresses, start a local SearXNG once (Docker or
Podman) and search from the Data tab or the command line:

```bash
make searxng
make search Q="elden ring review" ADD=games
make search Q="elden ring trailer" VIDEO=1 ADD=games
```

To teach a model your answer style, fine-tune it with LoRA: download a model with PyTorch
weights (`make download MODEL=qwen3-0.6b`), import examples on the Datasets tab (JSONL or CSV;
[docs/examples/game-hints.jsonl](docs/examples/game-hints.jsonl) is a small sample), start a run
on the Training tab and pick the new model in the chat. See [docs/api.md](docs/api.md#fine-tuning).

Engine only, for API clients (see [docs/api.md](docs/api.md)):

```bash
make setup-engine
make run-engine PORT=8765
make health
```

The engine serves OpenAI-compatible (`/v1/chat/completions`) and Anthropic-compatible
(`/v1/messages`) APIs.

Run `make help` for all targets.

## Python environment

By default the engine lives in its own virtual environment at
`${XDG_DATA_HOME:-~/.local/share}/tinylm-studio/venv`, created automatically.

To use your own environment instead, set `TINYLM_PYTHON` to its interpreter:

```bash
export TINYLM_PYTHON=/path/to/your/venv/bin/python
scripts/setup-engine.sh   # installs the engine into that environment
```

`TINYLM_VENV` changes where the managed environment is created.

`TINYLM_LLAMA` picks the llama.cpp build: `cpu` (default, prebuilt wheel) or `source`
(compile; e.g. `CMAKE_ARGS="-DGGML_CUDA=on"` for NVIDIA GPUs).

`TINYLM_TORCH` picks the PyTorch build: `cpu` (default, small), `auto` (detect GPU),
a CUDA version such as `cu128`, or `pypi`.

## Tests

```bash
make setup-dev   # once: install pytest and ruff into the engine environment
make check       # lint + tests
```
