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
    model="qwen3-1.7b-gguf",
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
    model="qwen3-1.7b-gguf",
    max_tokens=256,
    messages=[{"role": "user", "content": "Дай подсказку по первому боссу Dark Souls"}],
)
print(message.content[0].text)
```

Both protocols run through the same internal request type and runtime, so the same request
with `temperature: 0` returns the same text through either API.

### Answers from uploaded data

Both chat endpoints accept an extra `collection` field. The engine then searches that collection
for the chunks most similar to the last user message, adds them to the system prompt as numbered
sources and asks the model to cite them as `[1]`, `[2]`. The response carries a `sources` list
(top level for non-streaming responses, in the first chunk or `message_start` event when
streaming) so a client can show where each citation comes from.

```python
reply = client.chat.completions.create(
    model="gemma-3-4b-gguf",
    messages=[{"role": "user", "content": "Де відбуваються події Blood and Wine?"}],
    extra_body={"collection": "games"},
)
print(reply.choices[0].message.content)
print([s["document"] for s in reply.sources])
```

With the Anthropic SDK, pass the same `extra_body` and read `message.model_extra["sources"]`.

Not supported yet: tool use / function calling, images, logprobs, `n > 1`, batch APIs.

## Management (used by the desktop app)

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | Liveness and version |
| GET | `/api/system` | Python path and hardware |
| GET | `/api/models` | Catalog with install and download state |
| POST | `/api/models/{id}/download` | Starts a background download from Hugging Face |
| GET | `/api/collections` | Collections with document and chunk counts |
| POST | `/api/collections` | `{"name": "games"}`: letters, digits, `_`, `-` |
| DELETE | `/api/collections/{name}` | Deletes the collection and its documents |
| GET | `/api/collections/{name}/documents` | Documents in a collection |
| POST | `/api/collections/{name}/documents` | Multipart upload, field `files` (one or more); returns a result per file |
| DELETE | `/api/collections/{name}/documents/{id}` | Deletes one document |
| POST | `/api/collections/{name}/search` | `{"query": "...", "k": 4}`: the most similar chunks |

Interactive docs: `http://127.0.0.1:8765/docs`.

## Models

- Catalog models are downloaded into `<data dir>/models/<id>` (`make download MODEL=qwen3-1.7b-gguf`).
- Two formats are supported, chosen per model folder:
  - **GGUF** (`*.gguf`, quantized) runs with llama.cpp: the fast option on CPU and the default.
  - **Hugging Face** (`config.json` + safetensors) runs with PyTorch; used as a base for fine-tuning.
- Any such model folder copied into `<data dir>/models/` is served under the folder name.
- Gated models (Gemma, Llama) need `TINYLM_HF_TOKEN` and an accepted license on Hugging Face.
- `TINYLM_ENABLE_ECHO=true` (or `make run-engine ECHO=1`) adds an `echo` model that needs no
  weights, for testing clients and the UI.

## Uploaded data

- Supported files: PDF, DOCX, PPTX, XLSX/XLS, CSV, HTML, XML (converted with markitdown), and
  plain text, Markdown, JSON, YAML.
- Each document is split into chunks of up to ~1200 characters along paragraph and sentence
  boundaries, embedded with `intfloat/multilingual-e5-small` (uk/ru/en and ~100 other languages,
  downloaded on first use into `<data dir>/embeddings`) and stored in
  `<data dir>/workspace/knowledge.db` (SQLite with sqlite-vec).
- The same file content added twice to one collection is stored once.
- Whole folders can be added from the command line:
  `make ingest COLLECTION=games FILES=~/Documents/games` (or `python -m engine ingest games PATH...`).

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `TINYLM_DATA_DIR` | `~/.local/share/tinylm-studio` | Models and workspace |
| `TINYLM_HOST` / `TINYLM_PORT` | `127.0.0.1` / `8765` | Listen address |
| `TINYLM_API_KEY` | none | Require an API key |
| `TINYLM_HF_TOKEN` | none | Hugging Face token for gated models |
| `TINYLM_ENABLE_ECHO` | `false` | Offer the `echo` test model |
| `TINYLM_CORS_ORIGINS` | `[]` | Allowed browser origins (JSON list) |
| `TINYLM_THREADS` | physical cores, at most 8 | CPU threads for inference |
| `TINYLM_REPETITION_PENALTY` | `1.1` | Discourages repeated tokens; small models loop without it |
| `TINYLM_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Embedding model for uploaded data; `hash` needs no download (tests, UI development) |
| `TINYLM_RAG_TOP_K` | `4` | Chunks added to the prompt per question |
