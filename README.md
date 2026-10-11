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

Engine only:

```bash
make setup-engine
make run-engine PORT=8765
make health
```

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

## Tests

```bash
make setup-dev   # once: install pytest and ruff into the engine environment
make check       # lint + tests
```
