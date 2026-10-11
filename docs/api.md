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

### Timings

Every answer carries a `timings` object (top level for non-streaming responses; on the chunk
with `finish_reason` or the `message_delta` event when streaming). The names follow llama.cpp
server's `timings` field:

| Field | Meaning |
|---|---|
| `load_ms` | Loading the model, when this request had to load it |
| `retrieval_ms` | Searching the collection, when one was given |
| `prompt_n`, `prompt_ms`, `prompt_per_second` | Prompt tokens and the time until the first answer token |
| `predicted_n`, `predicted_ms`, `predicted_per_second` | Answer tokens and the time to generate the rest of them |
| `total_ms` | Sum of the above |

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

## Websites

`POST /api/collections/{name}/crawl` (or `make crawl COLLECTION=games URLS="https://site/ ..."`,
or the Data tab) adds pages of websites to a collection in the background:

- Starts from the given addresses and follows links **within the same sites** up to `max_depth`
  hops, until `max_pages` pages are fetched. Images, video, archives, scripts and styles are
  never requested. Pages with little text (listings, navigation) are followed but not indexed;
  web pages keep only their main text (menus and footers are dropped); PDFs are indexed too.
- **Politeness:** at most `per_minute` requests to each site (default `TINYLM_CRAWL_PER_MINUTE`
  = 20, with random jitter), robots.txt rules and `Crawl-delay` are followed (turn off with
  `respect_robots: false` only for your own sites), and an HTTP 429/503 answer pauses that site
  for its `Retry-After` time (60 s by default) before the page is retried.
- **Proxy:** `TINYLM_PROXY` sends all crawler traffic through an HTTP (`http://user:pass@host:port`)
  or SOCKS (`socks5://host:port`; `socks5h://` resolves names through the proxy) proxy.
- **Browser (CDP):** with `use_browser: true` pages are opened in a running Chrome/Chromium over
  the Chrome DevTools Protocol, so sites that build their content with JavaScript work, and the
  site sees a real browser. Start one with
  `chromium --remote-debugging-port=9222 --user-data-dir=$HOME/.cache/tinylm-browser` (add
  `--proxy-server=socks5://host:port` to use a proxy there) and set `TINYLM_CDP_URL` if it is not
  `http://127.0.0.1:9222`. robots.txt is still read over plain HTTP.

| Method | Path | Notes |
|---|---|---|
| POST | `/api/collections/{name}/crawl` | `{"urls": [...], "max_pages": 50, "max_depth": 2, "per_minute": 20, "use_browser": false, "respect_robots": true}` |
| GET | `/api/crawls` | All crawls since the engine started |
| GET | `/api/crawls/{id}` | Progress: `state`, `fetched`, `indexed`, `duplicates`, `skipped`, `failed`, `current`, `log` |
| POST | `/api/crawls/{id}/cancel` | Stops a crawl after the current page |

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `TINYLM_DATA_DIR` | `~/.local/share/tinylm-studio` | Models and workspace |
| `TINYLM_HOST` / `TINYLM_PORT` | `127.0.0.1` / `8765` | Listen address |
| `TINYLM_API_KEY` | none | Require an API key |
| `TINYLM_HF_TOKEN` | none | Hugging Face token: needed for gated models, gives faster downloads (the standard `HF_TOKEN` also works) |
| `TINYLM_ENABLE_ECHO` | `false` | Offer the `echo` test model |
| `TINYLM_CORS_ORIGINS` | `[]` | Allowed browser origins (JSON list) |
| `TINYLM_THREADS` | physical cores, at most 8 | CPU threads for inference |
| `TINYLM_REPETITION_PENALTY` | `1.1` | Discourages repeated tokens; small models loop without it |
| `TINYLM_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Embedding model for uploaded data; `hash` needs no download (tests, UI development) |
| `TINYLM_RAG_TOP_K` | `4` | Chunks added to the prompt per question |
| `TINYLM_PROXY` | none | HTTP or SOCKS proxy for website crawling |
| `TINYLM_CDP_URL` | `http://127.0.0.1:9222` | Browser DevTools endpoint for crawling with a browser |
| `TINYLM_CRAWL_PER_MINUTE` | `20` | Default requests per minute to each site |
| `TINYLM_USER_AGENT` | `tinylm-studio/0.1 (local research crawler)` | User-Agent of plain HTTP requests |
