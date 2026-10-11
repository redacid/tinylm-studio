# Engine API

The engine listens on `127.0.0.1:8765` by default (`make run-engine`). The desktop app starts
its own engine on a free port, shown in the window header.

## Authentication

Off by default. Set `TINYLM_API_KEY` to require a key; clients send it as
`Authorization: Bearer <key>` (OpenAI style) or `x-api-key: <key>` (Anthropic style).

## OpenAI-compatible

| Method | Path | Notes |
|---|---|---|
| GET | `/v1/models` | Installed models |
| POST | `/v1/chat/completions` | `stream`, `stream_options.include_usage`, `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `stop` |

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="unused")
reply = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "Коротко про The Witcher 3"}],
)
print(reply.choices[0].message.content)
```

## Anthropic-compatible

| Method | Path | Notes |
|---|---|---|
| GET | `/v1/models` | Same path as OpenAI; the Anthropic format is returned when the `anthropic-version` header is present |
| POST | `/v1/messages` | `system`, `stream`, `max_tokens`, `temperature`, `top_p`, `stop_sequences`; text content blocks |

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://127.0.0.1:8765", api_key="unused")
message = client.messages.create(
    model="qwen3-0.6b",
    max_tokens=256,
    messages=[{"role": "user", "content": "Дай подсказку по первому боссу Dark Souls"}],
)
print(message.content[0].text)
```

Both protocols run through the same internal request type and runtime, so the same request
with `temperature: 0` returns the same text through either API.

Not supported yet: tool use / function calling, images, logprobs, `n > 1`, batch APIs.

## Management (used by the desktop app)

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | Liveness and version |
| GET | `/api/system` | Python path and hardware |
| GET | `/api/models` | Catalog with install and download state |
| POST | `/api/models/{id}/download` | Starts a background download from Hugging Face |

Interactive docs: `http://127.0.0.1:8765/docs`.

## Models

- Catalog models are downloaded into `<data dir>/models/<id>` (`make download MODEL=qwen3-0.6b`).
- Any Hugging Face-format model folder (with `config.json`) copied into `<data dir>/models/` is
  served under the folder name.
- Gated models (Gemma, Llama) need `TINYLM_HF_TOKEN` and an accepted license on Hugging Face.
- `TINYLM_ENABLE_ECHO=true` (or `make run-engine ECHO=1`) adds an `echo` model that needs no
  weights, for testing clients and the UI.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `TINYLM_DATA_DIR` | `~/.local/share/tinylm-studio` | Models and workspace |
| `TINYLM_HOST` / `TINYLM_PORT` | `127.0.0.1` / `8765` | Listen address |
| `TINYLM_API_KEY` | none | Require an API key |
| `TINYLM_HF_TOKEN` | none | Hugging Face token for gated models |
| `TINYLM_ENABLE_ECHO` | `false` | Offer the `echo` test model |
| `TINYLM_CORS_ORIGINS` | `[]` | Allowed browser origins (JSON list) |
| `TINYLM_THREADS` | physical cores | CPU threads for inference |
| `TINYLM_REPETITION_PENALTY` | `1.1` | Discourages repeated tokens; small models loop without it |
