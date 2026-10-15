# Engine API

The engine listens on `127.0.0.1:8765` by default (`make run-engine`). The desktop app starts
its own engine on a free port, shown in the window header. For using the app itself see the
[user guide](user-guide.md).

## Authentication

Off by default. Set `TINYLM_API_KEY` to require a key; clients send it as
`Authorization: Bearer <key>` (OpenAI style) or `x-api-key: <key>` (Anthropic style). With a key
set, only `GET /health` answers without it. The [container image](deployment.md#container-image)
does not start without a key. The desktop app sends the key when it uses a remote engine or a
RunPod pod (Settings > Engine).

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

`max_tokens` (OpenAI), `temperature` and `top_p` may be left out: the model's own settings are
used then (see [Model settings](#model-settings)), and after them 512 tokens, 0.7 and 1. Note
that the Anthropic API's own default temperature is 1.

### Answers from uploaded data

Both chat endpoints accept an extra `collection` field. Left out, the model's own collection is
used if its settings name one (see [Model settings](#model-settings)); `""` asks for none. The
response then also says which `collection` was used. The engine searches that collection
for the chunks that best match the last user message, adds them to the system prompt as numbered
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

### Sources count and sampling

More extension fields work in both chat endpoints:

| Field | Default | Meaning |
|---|---|---|
| `rag_top_k` | `TINYLM_RAG_TOP_K` (4) | How many chunks of `collection` to add, 1 to 20 |
| `repetition_penalty` | the model's, then `TINYLM_REPETITION_PENALTY` (1.1) | Above 1 makes repeated tokens less likely, up to 2 |
| `top_k` | the model's, then the runtime's (40 in llama.cpp) | Samples only from this many most likely tokens; 0 turns it off (a standard field in the Anthropic API) |
| `min_p` | the model's, then the runtime's (0.05 in llama.cpp) | Drops tokens less likely than this share of the most likely one |
| `thinking` | the model's, then `false` | `true` lets models with a reasoning mode (Qwen3) think first; the thinking is part of the answer text |

```python
client.chat.completions.create(
    model="games01",
    messages=[{"role": "user", "content": "Як пройти Круг бійні?"}],
    extra_body={"collection": "games", "rag_top_k": 2, "repetition_penalty": 1.25},
)
```

### The prompt as the model got it

`/v1/chat/completions` also accepts `"debug": true`. The response then carries a `prompt`
object (top level for non-streaming responses, in the first chunk when streaming): the request
after the engine prepared it, with the sources added to the system prompt. The chat screen uses
it to show what the model was given.

```json
"prompt": {
  "model": "echo",
  "messages": [
    {"role": "system", "content": "Answer the user's question using the numbered sources below. ..."},
    {"role": "user", "content": "Who is Geralt?"}
  ],
  "max_tokens": null, "temperature": null, "top_p": null, "stop": [], "repetition_penalty": null,
  "top_k": null, "min_p": null, "thinking": null
}
```

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
| GET | `/health` | Liveness, `version` and `commit`: the git commit the engine was built from (`TINYLM_BUILD_COMMIT` in the container image, else the checkout it runs from, with `-dirty` for uncommitted changes; `null` when unknown) |
| GET | `/api/system` | Python path and hardware; needs the API key when one is set (unlike `/health`) |
| GET | `/api/models` | `models`: installed models with `format`, `size` in bytes, `source` (`repo`, `file`), `loaded`, `settings` and `used_by` (fine-tuned models that need it); `downloads` with `state`, `total` and `downloaded` bytes; the catalog with install and download state; `trainable`: installed models that can be fine-tuned |
| POST | `/api/models/{id}/download` | Starts a background download of a catalog model from Hugging Face |
| GET | `/api/models/{id}/settings` | The model's settings (only the ones set) |
| PUT | `/api/models/{id}/settings` | Replaces them; see [Model settings](#model-settings); 422 for a value out of range |
| GET | `/api/hub/models` | `?q=gemma&format=gguf\|hf`: matching models on Hugging Face, most downloaded first: `repo`, `downloads`, `likes`, `gated`, `updated`, `params`, `context_length` (GGUF); 502 when Hugging Face cannot be reached |
| GET | `/api/hub/files` | `?repo=unsloth/gemma-3-4b-it-GGUF`: `gguf` files (`file`, `size`, `parts`, `quantization`; a split file is one entry) and `hf`: the size of the safetensors weights, or null |
| POST | `/api/hub/download` | `{"repo": "...", "file": "gemma-3-4b-it-Q4_K_M.gguf"}`: one GGUF file (all parts of a split one), or without `file` the Hugging Face-format weights; saved as `<data dir>/models/<file or repo name>`; 202, 409 when a model of that name is installed |
| GET | `/api/models/trained` | Fine-tuned models: `base`, `dataset`, `loss`, `eval_loss`, `settings`, `base_installed` |
| DELETE | `/api/models/{id}` | Deletes an installed model folder and its settings |
| GET | `/api/collections` | Collections with document and chunk counts |
| POST | `/api/collections` | `{"name": "games"}`: letters, digits, `_`, `-` |
| DELETE | `/api/collections/{name}` | Deletes the collection and its documents |
| GET | `/api/collections/{name}/documents` | `?query=&kind=all\|file\|web&sort=oldest\|newest\|name\|largest&offset=0&limit=50`: one page of documents and the matching `total`; `query` matches the name or the file path / web address, any letter case |
| POST | `/api/collections/{name}/documents` | Multipart upload, field `files` (one or more); returns a result per file |
| DELETE | `/api/collections/{name}/documents/{id}` | Deletes one document |
| POST | `/api/collections/{name}/search` | `{"query": "...", "k": 4}`: the best matching chunks |

Interactive docs: `http://127.0.0.1:8765/docs`.

## Models

- Catalog models are downloaded into `<data dir>/models/<id>` (`make download MODEL=qwen3-1.7b-gguf`).
- Two formats are supported, chosen per model folder:
  - **GGUF** (`*.gguf`, quantized) runs with llama.cpp: the fast option on CPU and the default.
  - **Hugging Face** (`config.json` + safetensors) runs with PyTorch; used as a base for fine-tuning.
- Any such model folder copied into `<data dir>/models/` is served under the folder name.
- A model downloaded through the engine gets `tinylm-source.json` with the repo (and file) it
  came from.
- `GET /api/models/trained` lists fine-tuned models with `format`: `adapter` (a LoRA adapter,
  served through PyTorch on top of its base model) or `gguf` (an export, with `quantization`
  and `exported_from`).

### Model settings

Each installed model has its own settings, kept in `<data dir>/models/.model-settings.json`.
All are optional; a left-out one uses the default.

| Field | Default | Meaning |
|---|---|---|
| `context_size` | 4096 | GGUF: context window in tokens, 256 to 262144 |
| `gpu_layers` | -1 (all) | GGUF: layers offloaded to the GPU in a GPU build of llama.cpp |
| `batch_size` | 512 | GGUF: prompt tokens per step |
| `threads` | `TINYLM_THREADS` | CPU threads |
| `max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `thinking` | as in the chat endpoints | Used for requests that leave them out |
| `collection` | none | Requests that name no `collection` answer from this one; `""` turns off a collection that came with the model; 422 for an unknown collection |
| `rag_top_k` | `TINYLM_RAG_TOP_K` (4) | Chunks of that collection per answer |

The first four apply when the model is loaded: saving a change to them unloads the model. The
others apply to the next request.

```bash
curl -X PUT http://127.0.0.1:8765/api/models/gemma-3-4b-gguf/settings \
  -H 'content-type: application/json' -d '{"context_size": 8192, "temperature": 0.6}'
```

### Exporting to GGUF

`POST /api/models/{id}/export` merges a fine-tuned model's adapter into its base model, converts
the result with llama.cpp's `convert_hf_to_gguf.py` and quantizes it, in a worker process
(`python -m engine.export.worker`). The new model folder `<data dir>/models/<name>/` holds
`<name>.<quantization>.gguf` and `tinylm.json`; it is served through llama.cpp and does not need
the base model. The converter comes from the source package of the installed llama-cpp-python on
PyPI (downloaded once into `<data dir>/tools/`) or from `TINYLM_LLAMA_CPP_DIR`.

| Method | Path | Notes |
|---|---|---|
| POST | `/api/models/{id}/export` | `{"quantization": "Q8_0" \| "Q4_K_M" \| "F16", "name": "game-hints-0.6b-q8_0", "collection": "games"}` (all optional; the name defaults to `<id>-<quantization>`; with `collection` the model folder takes it along as `collection.jsonl` and answers from it, and an engine that finds such a folder at start imports the collection if it has none of that name); 202, 422 for a model that is not fine-tuned, a missing base model or a taken name, 409 while another export is running |
| GET | `/api/exports` | Exports, oldest first |
| GET | `/api/exports/{id}` | `state` (`running`, `done`, `error`, `interrupted`), `step` of `steps` (`tools`, `merge`, `convert`, `quantize`, `save`), `notes` (downloads, warnings), `size` of the file in bytes, `error` |
- Gated models (Gemma, Llama) need a token (`TINYLM_HF_TOKEN` or `HF_TOKEN`) and an accepted license on Hugging Face. `GET /api/models` reports `hf_token`: whether downloads send a token (from the settings, `HF_TOKEN` or a saved `hf auth login`). A refused download explains the cause: no token (HTTP 401) or a token whose account has no access to the model yet (HTTP 403).
- `TINYLM_ENABLE_ECHO=true` (or `make run-engine ECHO=1`) adds an `echo` model that needs no
  weights, for testing clients and the UI.

## Uploaded data

- Supported files: PDF, DOCX, PPTX, XLSX/XLS, CSV, HTML, XML (converted with markitdown), and
  plain text, Markdown, JSON, YAML.
- Each document is split into chunks of up to ~1200 characters along paragraph and sentence
  boundaries, embedded with `intfloat/multilingual-e5-small` (uk/ru/en and ~100 other languages,
  downloaded on first use into `<data dir>/embeddings`) and stored in
  `<data dir>/workspace/knowledge.db` (SQLite with sqlite-vec).
- Search is hybrid: chunks are ranked by meaning (vectors) and by words (SQLite FTS5 with BM25;
  long words also match by prefix, so inflected forms find each other), and the two rankings are
  merged with reciprocal rank fusion. Exact names such as "Borderlands 3" are found even when the
  small embedding model blurs them. A source's `score` is the fused rank: higher is better, and it
  is only comparable within one search.
- The same file content added twice to one collection is stored once.
- Whole folders can be added from the command line:
  `make ingest COLLECTION=games FILES=~/Documents/games` (or `python -m engine ingest games PATH...`).

## Websites

`POST /api/collections/{name}/crawl` (or `make crawl COLLECTION=games URLS="https://site/ ..."`,
or the Data tab) adds pages of websites to a collection in the background:

- Starts from the given addresses and follows links **within the same sites** (the same host,
  `www.` aside; other subdomains are other sites) up to `max_depth` hops; with
  `only_under_start: true`, only links under the start addresses' folders (from
  `https://example.com/guides/`, only `/guides/…`); links matching `exclude` are skipped (a
  part of the address like `"/tag/"`, or a pattern with `*` for the whole address like
  `"*/forum/*"`; letter case is ignored); until `max_pages` **new** pages are
  downloaded. Images, video, archives, scripts and styles are never requested. Pages with little text (listings, navigation) are followed but not indexed;
  web pages keep only their main text (menus and footers are dropped); PDFs are indexed too.
- **Repeated crawls continue where the last one stopped.** Every downloaded page is remembered
  in the collection with its links. A later crawl into the same collection walks through those
  pages without requesting them and does not count them toward `max_pages`, so running a site
  again and again gradually downloads all of it (raise `max_depth` to reach deep pages).
  Deleting a document forgets its page, so the next crawl downloads it again. Pages with little
  text are remembered too: new links that later appear on them are not seen.
- **Saved websites** keep the addresses and settings of a crawl so it can be run again, edited
  or deleted (deleting a saved website keeps the pages already added).
- **Defaults:** `max_pages`, `max_depth`, `per_minute` and `use_browser` left out of a crawl or
  a new website come from the crawler settings (`GET /api/crawl/settings`; 50, 2, 20 and
  `false` unless changed). A crawl through the browser says so in its `log`.
- **Politeness:** at most `per_minute` requests to each site (with random jitter), robots.txt
  rules and `Crawl-delay` are followed, and an HTTP 429/503 answer pauses that site for its
  `Retry-After` time (60 s by default) before the page is retried. robots.txt is read only when
  both the crawl's `respect_robots` and the crawler settings' `respect_robots` are on: either
  turns it off (only for your own sites), and the crawl's `log` then says so.
- **Proxy:** `TINYLM_PROXY` sends all crawler traffic through an HTTP (`http://user:pass@host:port`)
  or SOCKS (`socks5://host:port`; `socks5h://` resolves names through the proxy) proxy.
- **Browser (CDP):** with `use_browser: true` pages are opened in a running Chrome/Chromium over
  the Chrome DevTools Protocol, so sites that build their content with JavaScript work, and the
  site sees a real browser. `make chrome` starts a headless Chrome in Docker or Podman
  (`chromedp/headless-shell`; `CDP_PROXY=socks5://host:port` for a proxy), or start one with
  `chromium --remote-debugging-port=9222 --user-data-dir=$HOME/.cache/tinylm-browser` (add
  `--proxy-server=socks5://host:port` to use a proxy there) and set `TINYLM_CDP_URL` if it is not
  `http://127.0.0.1:9222` (the crawler settings' `cdp_url` wins). robots.txt is still read over
  plain HTTP.

| Method | Path | Notes |
|---|---|---|
| POST | `/api/collections/{name}/crawl` | One-off crawl: `{"urls": [...], "max_pages": 50, "max_depth": 2, "per_minute": 20, "use_browser": false, "respect_robots": true, "only_under_start": false, "exclude": []}` |
| GET | `/api/collections/{name}/websites` | Saved websites of a collection, with `last_crawled_at` |
| POST | `/api/collections/{name}/websites` | Saves a website: the crawl fields above plus an optional `name` (defaults to the host) |
| PUT | `/api/websites/{id}` | Changes a saved website (same body) |
| DELETE | `/api/websites/{id}` | Deletes a saved website; its pages stay in the collection |
| POST | `/api/websites/{id}/crawl` | Crawls a saved website again; 409 while it is already running |
| GET | `/api/crawls` | All crawls since the engine started |
| GET | `/api/crawls/{id}` | Progress: `state`, `website_id`, `fetched` (new pages), `known` (downloaded before), `indexed`, `duplicates`, `skipped`, `failed`, `current`, `log`, and `outcomes`: per start address `added`, `duplicate`, `known`, `skipped` or `failed` (missing: not reached) |
| POST | `/api/crawls/{id}/cancel` | Stops a crawl after the current page |
| GET | `/api/crawl/settings` | `respect_robots`, `per_minute`, `max_pages`, `max_depth`, `use_browser`, `user_agent`, `cdp_url` |
| PUT | `/api/crawl/settings` | Any of those fields; fields left out keep their value. Saved in `workspace/crawl-settings.json`, wins over the `TINYLM_*` variables and applies to crawls started afterwards; 422 names a wrong value |

## Web search

`POST /api/search` (or `make search Q="elden ring review"`, or Find on the Data tab) finds pages
and videos for a collection, so sites and trailers do not have to be looked up by hand:

The provider and the site filters are search settings kept by the engine (Settings > Search in
the app, or the endpoints below); the variables in [Settings](#settings) are their defaults.

- **SearXNG** (default, `TINYLM_SEARCH_PROVIDER=searxng`): a self-hosted metasearch engine that
  asks Google, Bing, DuckDuckGo, YouTube and others. Free, no key. `make searxng` starts it in
  Docker or Podman on `http://127.0.0.1:8888` (`TINYLM_SEARXNG_URL`), with
  `<data dir>/searxng/settings.yml` (copied from `scripts/searxng/settings.yml`) that turns on
  the JSON output the engine needs; edit it to choose engines or give SearXNG an outgoing proxy
  (`outgoing.proxies`), then run `make searxng` again. An existing SearXNG works too if `json`
  is listed in its `search.formats`.
- **Hosted APIs**, each with its own key: `brave` (Brave Search), `tavily`, `exa`, `perplexity`
  (Search API) and `serper` (Google's results). Requests go through `TINYLM_PROXY` when it is
  set. Tavily, Exa and Perplexity have no video search: `kind: "video"` searches pages on
  YouTube and Vimeo through them.
- **Site filters**: `include_domains` (only these sites, subdomains included; empty: any) and
  `exclude_domains` (never these). Each entry is a domain or a pattern where `*` stands for any
  characters, dots included: `*.ua` (also written `.ua`) is the whole .ua zone, `*.google.*`
  covers `google.co.uk` and `maps.google.com`, `steam*.com` covers `steampowered.com`.
  Tavily, Exa and Perplexity (up to 20 sites, allow or deny list) take plain domains in their
  APIs, and Perplexity also `.ua` for `*.ua`. SearXNG, whose only site filter is its global
  `hostnames` plugin, Brave and Serper get one `site:<domain>` query per included site (up to
  5; `site:.ua` for `*.ua`). Other patterns are applied by the engine only: an include list
  with one is not passed on, and the search runs unrestricted. Every result is checked by the
  engine, which fetches up to 3 pages when the filter leaves too few.

What to do with the results:

- **Pages:** download and index just those pages (a crawl with `max_depth: 0`), or save their
  sites as websites to crawl further, now and later.
- **Videos** (trailers, walkthroughs): `POST /api/collections/{name}/links` keeps each one as a
  small document with its title, address and description, so answers can cite the link.

| Method | Path | Notes |
|---|---|---|
| POST | `/api/search` | `{"query": "...", "kind": "web" or "video", "limit": 10, "language": "uk"}` (language optional; `include_domains` / `exclude_domains` override the saved filters for this search): `provider`, results (`data`) with `title`, `url`, `snippet`, `site`, `thumbnail`, `source`, `hidden` (results dropped by the filters), `queries` sent and the filters used; 503 with the reason when the provider is not reachable or not set up |
| GET | `/api/search/settings` | Provider, options and filters; API keys only as `<provider>_api_key_set` |
| PUT | `/api/search/settings` | Any of `provider`, `include_domains`, `exclude_domains`, `searxng_url`, `brave_api_key`, `tavily_api_key`, `tavily_depth` (`basic`, `advanced`), `tavily_topic` (`general`, `news`), `exa_api_key`, `exa_type` (`auto`, `fast`, `instant`), `perplexity_api_key`, `serper_api_key`, `serper_country`; fields left out keep their value, `""` removes a key. Saved in `workspace/search-settings.json`; 422 names a wrong domain or pattern |
| POST | `/api/collections/{name}/links` | `{"links": [{"title": "...", "url": "https://...", "snippet": "...", "kind": "video"}]}` |

## Datasets

A dataset holds training examples: conversations whose assistant answers the model learns to
write. Examples live in `knowledge.db` next to the collections.

- Import JSONL, a JSON array or CSV. Each record is one of:
  - `{"messages": [{"role": "system" | "user" | "assistant", "content": "..."}, ...]}` (OpenAI chat;
    the last message must be the assistant's answer),
  - `{"conversations": [{"from": "human" | "gpt", "value": "..."}, ...]}` (ShareGPT),
  - `{"instruction": "...", "input": "...", "output": "..."}` (Alpaca),
  - `{"prompt": "...", "completion": "..."}` or `{"question": "...", "answer": "..."}`,

  optionally with `"system": "..."`. CSV files use the same names as column headers. Bad records
  are skipped and reported with their line numbers; the same example twice is stored once.
- `eval_percent` (default 10, at most 50) of the examples are held out from training to measure
  the model on data it has not seen. The split is fixed per example, so adding, editing or
  deleting other examples does not move an example between the parts.

| Method | Path | Notes |
|---|---|---|
| GET | `/api/datasets` | Datasets with `examples`, `train` and `eval` counts |
| POST | `/api/datasets` | `{"name": "game-hints", "eval_percent": 10}` |
| PATCH | `/api/datasets/{name}` | `{"eval_percent": 20}` |
| DELETE | `/api/datasets/{name}` | Deletes the dataset and its examples |
| POST | `/api/datasets/{name}/import` | Multipart upload, field `files`: `added`, `duplicates`, `invalid` and the first `errors` (`line`, `error`) per file |
| GET | `/api/datasets/{name}/examples` | `?query=&split=all\|train\|eval&origin=manual\|imported\|generated&offset=0&limit=20`: one page of examples (each with `split`, `origin` and, for generated ones, `source`) and the matching `total` |
| POST | `/api/datasets/{name}/examples` | `{"messages": [...]}` |
| PUT | `/api/datasets/{name}/examples/{id}` | `{"messages": [...]}` |
| DELETE | `/api/datasets/{name}/examples/{id}` | Deletes one example |
| POST | `/api/datasets/{name}/generate` | Starts generating examples, see below; 202 with the run, 404 for an unknown dataset, collection or teacher, 409 while another generation is running |
| GET | `/api/datasets/{name}/generate?collection=games` | `unused_chunks`: chunks of the collection this dataset has not been generated from |

### Generating examples

A teacher model reads chunks of a collection and writes question and answer pairs from them. The
pairs are checked (JSON shape, answer length, language, and optionally by the teacher against the
text) and the good ones are added with `origin: "generated"` and the document's `source`. Each
chunk is used once per dataset, so repeated runs continue with new text. Failed chunks stay
unused. One run at a time, in a background thread.

```json
POST /api/datasets/game-hints/generate
{
  "collection": "games",
  "teacher": "1",
  "chunks": 20,
  "per_chunk": 3,
  "languages": ["uk", "en"],
  "instructions": "A one-line tip, then numbered steps.",
  "system_prompt": "You are a helpful game guide.",
  "unanswerable": true,
  "verify": true,
  "with_sources": false,
  "sources": 4,
  "min_answer_chars": 20,
  "max_answer_chars": 2000
}
```

`teacher` is a `ref` from `GET /api/teachers`. Only `collection` and `teacher` are required.
`languages` empty means the language of each text; `unanswerable` adds a question the text does
not answer with an honest answer; `verify` makes a second request per chunk to drop unsupported
pairs; `with_sources` writes each example the way a chat with a collection is asked: its system
prompt gets `sources` chunks found for the question (the source text among them, numbered like
in grounded answers) and the answer ends with that source's number, e.g. `[2]`. Up to three examples of the dataset that are not generated are shown to the teacher as
style samples.

| Method | Path | Notes |
|---|---|---|
| GET | `/api/generations` | Runs since the engine started |
| GET | `/api/generations/{id}` | `state` (`running`, `done`, `cancelled`, `error`), `chunks_total`, `chunks_done`, `generated`, `added`, `duplicates`, `rejected` (reason → count), `failed_chunks`, `current` (document being read), `waiting` (seconds of a rate-limit pause), `requests` (the last 100 teacher requests, newest last: `document`, `step` (`write` or `check`), `model` and `provider` as the API reported them (OpenRouter names the routed model and the upstream provider, also in errors; Anthropic gives `"Anthropic"`, installed models `"local"`; otherwise the model asked for), `ok` (`null` while running), `result` (what came back, or the error) and `at`), the last 50 `log` lines, `error`, `options` |
| POST | `/api/generations/{id}/cancel` | Stops after the chunk in progress |

### Teachers

Saved API providers, plus installed models as local teachers. Keys are stored in `knowledge.db`
and never returned: responses only have `has_key`.

```json
POST /api/teachers
{
  "name": "Llama 3.3 70B (free)",
  "kind": "openai",
  "base_url": "https://openrouter.ai/api/v1",
  "model": "meta-llama/llama-3.3-70b-instruct:free",
  "api_key": "sk-or-...",
  "per_minute": 20
}
```

`kind` is `openai` (any OpenAI-compatible `/chat/completions` API; the base URL defaults to
OpenRouter) or `anthropic` (the Messages API; defaults to `https://api.anthropic.com`). `name`
defaults to the model. `per_minute` spaces requests out; HTTP 429 answers are waited out
(`Retry-After`) and retried up to five times.

| Method | Path | Notes |
|---|---|---|
| GET | `/api/teachers` | Saved teachers (`id`, `ref`, `name`, `kind`, `base_url`, `model`, `has_key`, `per_minute`), then installed models with `kind: "local"` and `ref: "local:<model id>"` |
| POST | `/api/teachers` | Saves a teacher; 201 |
| PUT | `/api/teachers/{id}` | The same body; an empty `api_key` keeps the saved key |
| DELETE | `/api/teachers/{id}` | 204 |
| POST | `/api/teachers/models` | The same body (or with `id` to use a saved key): the provider's models as `id`, `name`, `free`, free ones first; 502 with the provider's error |
| POST | `/api/teachers/test` | The same body: sends "Say OK." and returns `{"reply": "OK", "served": {"model": "...", "provider": "..."}}`; 502 with the reason (rejected key, not reachable, unknown model) |

## Fine-tuning

A training run teaches a base model with PyTorch weights (e.g. `qwen3-0.6b`) the answers of a
dataset with LoRA: small adapter matrices are trained while the model itself stays frozen. Only
the assistant's answers (with the chat template's end-of-turn marker) are learned; system and
user messages are context.

- One run at a time, in its own process (`python -m engine.training.worker`), so the engine
  stays responsive. On CPU it uses all physical cores (`threads` to change); `device: "auto"`
  picks an NVIDIA GPU (bf16 or fp16) when PyTorch sees one.
- Checkpoints are saved about ten times per run (`save_steps` to change). Stop saves one at the
  next step; a stopped, failed or interrupted run (the engine was closed) resumes from its last
  checkpoint.
- The result is a new model folder `<data dir>/models/<name>` with the adapter, the tokenizer and
  `tinylm.json` (base model, dataset, losses, settings). It is offered in chat and through both
  APIs under its name as long as its base model is installed; the adapter is merged into the base
  model's weights when it is loaded, so it answers as fast as the base model through PyTorch.
- Run files: `<data dir>/workspace/training/<id>/` (`job.json`, `events.jsonl`, `worker.log`,
  `checkpoints/`).

| Method | Path | Notes |
|---|---|---|
| POST | `/api/training/jobs` | `{"name": "game-hints-0.6b", "base": "qwen3-0.6b", "dataset": "game-hints"}` plus optional `epochs` (3), `learning_rate` (2e-4), `lora_rank` (16), `lora_alpha` (2 × rank), `lora_dropout` (0.05), `max_length` (1024 tokens; longer examples are cut), `batch_size` (1), `gradient_accumulation` (4: examples per optimizer step = batch × accumulation), `device` (`auto`, `cpu`, `cuda`, `mps`), `threads`, `save_steps`; 409 while another run is going |
| GET | `/api/training/jobs` | Runs with `state` (`running`, `stopping`, `stopped`, `done`, `error`, `interrupted`), `step`, `total_steps`, `epoch`, `loss`, `eval_loss`, `eta_seconds`, `resumable` |
| GET | `/api/training/jobs/{id}` | The same plus `history` (`step`, `loss`, `eval_loss`) and, for a failed run, the end of the worker's `log` |
| GET | `/api/training/jobs/{id}/events` | Server-sent events: `start`, `step`, `eval`, `checkpoint`, `stopped`, `done`, `error` from the worker and `state` changes; ends when the run is no longer going |
| POST | `/api/training/jobs/{id}/stop` | Saves a checkpoint and stops at the next step |
| POST | `/api/training/jobs/{id}/resume` | Continues from the last checkpoint |
| DELETE | `/api/training/jobs/{id}` | Deletes a run that is not going, with its checkpoints (not the finished model) |

## Evaluation

Models answer the held-out examples of a dataset (greedy decoding, so runs are repeatable), and
each answer is scored against the dataset's answer:

- `similarity`: chrF, a character n-gram F-score from 0 to 100 (recall weighted twice as much
  as precision). Cheap and language-independent; it measures wording, not correctness.
- `score` (with `judge`): 1 to 5 from a teacher, with a short `reason`. All models' answers to
  one question are judged in one request under shuffled letters.

One run at a time, in a background thread; models answer one after another (each is loaded
once), then the judge scores. Reports are saved to `<data dir>/workspace/evaluations/<id>.json`
after every step; a run that was going when the engine stopped is marked `interrupted`.

```json
POST /api/evaluations
{"dataset": "game-hints", "models": ["qwen3-0.6b", "game-hints-0.6b"], "judge": "1", "limit": 50, "max_tokens": 512}
```

| Method | Path | Notes |
|---|---|---|
| GET | `/api/evaluations` | Runs, oldest first, without the per-example answers |
| POST | `/api/evaluations` | Starts a run: `dataset` and `models` (1 to 4 installed models) required; `judge` is a teacher `ref`; 202, 404 for an unknown dataset or judge, 422 for no held-out examples or a model that is not installed, 409 while another run is going |
| GET | `/api/evaluations/{id}` | `state` (`running`, `done`, `cancelled`, `error`, `interrupted`), `phase` (`answering`, `judging`), `current` model, `done` of `total` steps, `summary` per model (`answers`, `similarity`, `judged`, `score`, `seconds`, `chars`) and `examples`: `messages`, `reference` and `answers` per model (`text`, `seconds`, `similarity`, `score`, `reason`) |
| POST | `/api/evaluations/{id}/stop` | Stops after the answer in progress |
| DELETE | `/api/evaluations/{id}` | Deletes a report that is not running |

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `TINYLM_DATA_DIR` | `~/.local/share/tinylm-studio` | Models and workspace |
| `TINYLM_HOST` / `TINYLM_PORT` | `127.0.0.1` / `8765` | Listen address; `0.0.0.0` in a container (see [deployment.md](deployment.md)) |
| `TINYLM_API_KEY` | none | Require an API key |
| `TINYLM_BUILD_COMMIT` | set by the container image | The git commit `/health` reports; outside the image the engine asks git |
| `TINYLM_HF_TOKEN` | none | Hugging Face token: needed for gated models, gives faster downloads (`HF_TOKEN` works too) |
| `TINYLM_ENABLE_ECHO` | `false` | Offer the `echo` test model |
| `TINYLM_CORS_ORIGINS` | `[]` | Allowed browser origins (JSON list) |
| `TINYLM_THREADS` | physical cores, at most 8 | CPU threads for inference |
| `TINYLM_REPETITION_PENALTY` | `1.1` | Discourages repeated tokens; small models loop without it |
| `TINYLM_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Embedding model for uploaded data; `hash` needs no download (tests, UI development) |
| `TINYLM_RAG_TOP_K` | `4` | Chunks added to the prompt per question |
| `TINYLM_PROXY` | none | HTTP or SOCKS proxy for website crawling |
| `TINYLM_CDP_URL` | `http://127.0.0.1:9222` | Browser DevTools endpoint for crawling with a browser |
| `TINYLM_CRAWL_PER_MINUTE` | `20` | Default requests per minute to each site |
| `TINYLM_CRAWL_RESPECT_ROBOTS` | `true` | `false`: robots.txt is not read for any crawl |
| `TINYLM_CRAWL_MAX_PAGES` / `TINYLM_CRAWL_MAX_DEPTH` | `50` / `2` | Defaults for new crawls and websites |
| `TINYLM_CRAWL_USE_BROWSER` | `false` | `true`: new crawls and websites load pages in the browser at `TINYLM_CDP_URL` |
| `TINYLM_USER_AGENT` | `tinylm-studio/0.1 (local research crawler)` | User-Agent of plain HTTP requests |
| `TINYLM_SEARCH_PROVIDER` | `searxng` | Web search: `searxng`, `brave`, `tavily`, `exa`, `perplexity` or `serper`. This and the search variables below are defaults; the saved search settings win |
| `TINYLM_SEARXNG_URL` | `http://127.0.0.1:8888` | SearXNG address (`make searxng` starts one) |
| `TINYLM_SEARCH_INCLUDE_DOMAINS` / `TINYLM_SEARCH_EXCLUDE_DOMAINS` | `[]` | Only / never these sites (JSON list of domains or patterns like `*.ua`) |
| `TINYLM_BRAVE_API_KEY`, `TINYLM_TAVILY_API_KEY`, `TINYLM_EXA_API_KEY`, `TINYLM_PERPLEXITY_API_KEY`, `TINYLM_SERPER_API_KEY` | none | API keys of the hosted search providers |
| `TINYLM_LLAMA_CPP_DIR` | none | A llama.cpp checkout whose `convert_hf_to_gguf.py` exports to GGUF; by default the converter is downloaded once from PyPI |
