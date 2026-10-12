# User guide

How to use tinylm-studio, screen by screen, and how to set it up for your network and hardware.
For the HTTP API see [api.md](api.md); for the design and roadmap see the
[implementation plan](implementation-plan.md).

- [Start](#start)
- [Chat](#chat)
- [Models](#models)
- [Data: collections, files, web search and websites](#data)
- [Datasets](#datasets)
  - [Generating examples from your documents](#generating-examples)
- [Teachers: OpenRouter, Anthropic and local models](#teachers)
- [Training](#training)
- [Settings](#settings)
  - [Proxy for crawling and search](#proxy)
  - [Crawling through a real browser (CDP)](#browser-cdp)
  - [Web search: SearXNG or Brave](#web-search)
  - [Hardware: threads and GPU](#hardware)
- [Where data is stored](#where-data-is-stored)
- [Command line](#command-line)

## Start

```bash
make setup   # once: the engine's Python environment and the desktop app's dependencies
make run     # start the app; it starts the engine and shows "Engine ready" in the top right
```

After pulling changes that add Python packages (see `engine/pyproject.toml`), run
`make setup-engine` again: the app installs the engine only on its very first start.

Settings go into `.env` in the repository (`cp .env.example .env`); every `make` target and the
app read it.

## Chat

![Chat with answers from a collection](images/chat.png)

- Pick a **model** in the first list. Downloaded models, fine-tuned models and (with
  `TINYLM_ENABLE_ECHO=true`) the test model `echo` are offered.
- Pick **Data: \<collection\>** to answer from your uploaded data. The answer lists the sources
  it was given; the model is asked to cite them as [1], [2].
- The **system prompt** sets the assistant's role or tone for the conversation.
- The line under each answer shows where the time went: loading the model, searching the data,
  reading the prompt (tokens per second) and writing the answer. Use it to see what is slow.
- **New chat** clears the conversation; **Stop** ends an answer early.

The same models answer through the OpenAI- and Anthropic-compatible APIs, so any client or SDK
can use them (see [api.md](api.md)).

## Models

![Models: fine-tuned and base models](images/models.png)

- **Base models** can be downloaded with one click. GGUF models (`-gguf`) run through llama.cpp
  and are the fast choice for chatting on a CPU; the default is `gemma-3-4b-gguf`. Models with
  PyTorch weights (e.g. `qwen3-0.6b`) are slower to chat with but are the ones that can be
  fine-tuned.
- Gemma and Llama models are gated: accept their license on Hugging Face and put your token into
  `.env` as `HF_TOKEN=...`. A token also makes downloads faster.
- **Fine-tuned models** show the base model, the dataset and the final loss. Deleting one
  removes its folder; the base model stays.
- Any GGUF file or Hugging Face model folder copied into `<data dir>/models/<name>/` appears too.
- **Teachers** at the bottom of the screen are the models that write training examples; see
  [Teachers](#teachers).

## Data

![Data: a collection with files, web search and a saved website](images/data.png)

A **collection** is a set of documents that chat can answer from. Create one on the left, then:

- **Add files** or **Add folder**: PDF, DOCX, PPTX, XLSX, CSV, HTML, Markdown, text and JSON.
  The same content added twice is stored once.
- **Find on the web** (needs a [search provider](#web-search)): search for pages or videos,
  tick results and add them.

  ![Web search results with what happened to each page](images/search.png)

  - **Pages** are downloaded and indexed; each result shows whether it was added, was already
    there, was skipped (too little text, or not allowed by robots.txt) or could not be
    downloaded.
  - **Save as websites** keeps the sites of the ticked pages for crawling them further.
  - **Videos** (trailers, walkthroughs) are kept as links with their title and description, so
    answers can point to them.
- **Websites**: save a site once and crawl it again whenever you like.

  ![Adding a website](images/website-form.png)

  - *Pages* is the number of **new** pages to download per run. Pages downloaded before are
    walked through without downloading them again and are not counted, so every run continues
    where the last one stopped until the whole site is in the collection.
  - *Depth* is how many links away from the start addresses to go.
  - *Per minute* limits requests to each site. Keep it low (the default is 20) so the site is
    not overloaded and does not ban you.
  - *Browser (CDP)* loads pages in a real Chrome or Chromium, for sites that need JavaScript or
    block simple clients (see [Browser (CDP)](#browser-cdp)).
  - robots.txt is followed, and a site that answers "too many requests" is paused for as long
    as it asks.

  ![A finished crawl](images/websites.png)

- **Documents**: everything in the collection, 50 per page, with search by name or address,
  a filter for uploaded files or web pages, and sorting.

  ![Documents of a collection](images/documents.png)

## Datasets

![Datasets: imported examples](images/datasets.png)

A dataset holds examples of conversations: questions and the answers the model should learn to
give. Fine-tuning teaches a model the **style and format** of these answers (tone, structure,
language); facts are better left to collections and chat with data.

- **Import file**: JSONL, JSON or CSV. Each record is either a chat
  (`{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`)
  or a pair (`prompt`/`completion`, `question`/`answer`, `instruction`/`input`/`output`), with an
  optional `system`. Bad records are skipped and listed with their line numbers.
  [examples/game-hints.jsonl](examples/game-hints.jsonl) is a small sample in Ukrainian, Russian
  and English.
- **Add example**, **Edit** and **Delete** change examples by hand.
- **Evaluation %** holds examples out of training to measure the model on data it has not
  seen. Each example stays in its part when others are added or removed.
- Each example shows where it came from: **imported**, **generated** (with the document it was
  written from) or written by hand. The origin filter shows one kind at a time, for example to
  review generated examples before training.

### Generating examples

Writing hundreds of examples by hand is slow. **Generate** lets a larger *teacher* model write
them from the documents of a collection: it reads a chunk of text, writes question and answer
pairs that use only facts from that text, and then checks them.

![Generating examples from a collection](images/generate.png)

1. Add documents, websites or search results to a collection on the **Data** screen.
2. Add a teacher on the **Models** screen ([Teachers](#teachers)), or install a model to use as
   a local teacher.
3. Open the dataset, press **Generate**, pick the collection and the teacher, and press
   **Generate** in the panel.

Settings:

- **Chunks**: how many pieces of text the teacher reads in this run (about 1,000 characters
  each). Every run continues with chunks this dataset has not used yet, so you can generate in
  small portions and review in between. Chunks shorter than 300 characters are skipped.
- **Pairs per chunk**: question and answer pairs written from each chunk (3 by default).
- **Languages**: Ukrainian, Russian and/or English. With several, the pairs are spread over
  them; with none, each pair is in the language of its text. Pairs in another language are
  dropped.
- **Answer style**: how answers should look, e.g. "a one-line tip, then numbered steps". The
  examples you wrote or imported into the dataset are also shown to the teacher as samples of the
  style, so a few good hand-written examples go a long way.
- **System prompt of the examples**: added as the system message of every generated example.
  Use the same one later in chat.
- **Add a question the text does not answer**: one more pair per chunk where the answer honestly
  says there is no such information. This teaches the small model not to invent facts.
- **Check pairs against the text**: a second request per chunk in which the teacher drops pairs
  the text does not support. It doubles the number of requests but catches made-up answers.

The progress shows chunks read, pairs written and added, duplicates, and why pairs were dropped
(wrong language, too short or long, not supported by the text). **Stop** ends the run after the
current chunk; what was added stays. When a free provider's rate limit is reached, the run waits
and retries by itself. After three failed chunks in a row (for example, a wrong API key) the run
stops with the reason.

![Generated examples with the document they were written from](images/generated.png)

Always skim generated examples before training: delete wrong ones and fix the style where
needed. Generated examples are marked, so they are easy to find with the origin filter.

## Teachers

A teacher is the model that writes examples. Teachers are set up at the bottom of the
**Models** screen.

![Adding OpenRouter as a teacher](images/teacher-form.png)

- **OpenRouter** gives access to many models through one key, including free ones (their ids
  end with `:free`). Create a key at [openrouter.ai/keys](https://openrouter.ai/keys), press
  **Add teacher**, keep the OpenRouter provider, paste the key, press **List models** and pick a
  model; **Free only** hides the paid ones. Free models are limited to about 20 requests per
  minute (and a daily limit), so the requests-per-minute field is preset to 20.
- **Any OpenAI-compatible API**: OpenAI, Groq, or a server on your computer such as Ollama
  (`http://localhost:11434/v1`) or LM Studio (`http://localhost:1234/v1`). Choose a preset or
  **Other** and enter the base URL; servers on your computer need no key.
- **Anthropic** (Claude): choose the Anthropic provider and paste a key from
  [console.anthropic.com](https://console.anthropic.com).
- **Installed models** are offered as teachers without any setup. They are free and private,
  but a 4B model on a CPU takes minutes per chunk; use them for small runs.

**Test** sends a tiny request and shows the answer or the provider's error. **Requests per
minute** keeps the studio under the provider's limit; leave it empty for no limit.

![Saved teachers](images/teachers.png)

API keys are stored in `knowledge.db` on this computer, are sent only to the teacher's base URL,
and are never shown again: when you edit a teacher, leave the key empty to keep the saved one.
Generating sends the text of your documents to the provider, so prefer a local teacher for
private data.

## Training

![Training: a finished run and its loss](images/training.png)

1. Download a base model with PyTorch weights on the Models tab, such as `qwen3-0.6b`.
2. Name the new model, pick the base model and the dataset, and press **Start training**.
3. Follow the loss. The training loss (line) and the held-out loss (dots) should both go down;
   if the held-out loss starts going up while the training loss keeps falling, the model is
   memorizing the examples: use fewer epochs.
4. When the run finishes, the new model is in the chat's model list and in the APIs.

Settings:

| Setting | Default | What it does |
|---|---|---|
| Epochs | 3 | Passes over the training examples |
| Learning rate | 0.0002 | How fast the adapter changes; too high makes the loss jump |
| LoRA rank | 16 | Size of the adapter: higher learns more, trains slower |
| Max tokens | 1024 | Longer examples are cut; shorter is faster and needs less memory |
| Batch × Accumulate | 1 × 4 | Examples per step; the default keeps a 0.6B model at about 6 GB of RAM |
| Device | Auto | An NVIDIA GPU when PyTorch sees one, otherwise the CPU (all physical cores) |

Training runs in its own process, one run at a time. **Stop** saves a checkpoint;
**Resume** continues from it, also after the app was closed in the middle of a run. On a CPU,
LoRA on a 0.6B model takes on the order of 15 seconds per step on 4 cores (less on more
cores), so 1,000 examples × 3 epochs take hours, not days.

## Settings

Put these into `.env` (see [.env.example](../.env.example)); the full list is in
[api.md](api.md#settings).

### Proxy

`TINYLM_PROXY` sends crawler traffic, and Brave Search requests, through a proxy:

```bash
TINYLM_PROXY=http://user:password@proxy.example:3128
TINYLM_PROXY=socks5://127.0.0.1:1080     # names resolved locally
TINYLM_PROXY=socks5h://127.0.0.1:1080    # names resolved by the proxy (e.g. Tor)
```

Model downloads from Hugging Face use the standard `HTTPS_PROXY` variable instead. A local
SearXNG has its own outgoing proxy setting (`outgoing.proxies` in its `settings.yml`, see
[Web search](#web-search)).

### Browser (CDP)

Some sites build their pages with JavaScript or refuse simple HTTP clients. Tick
**Browser (CDP)** on a website (or pass `--browser` to `python -m engine crawl`) to load its pages in a real
browser that the engine controls over the Chrome DevTools Protocol:

```bash
chromium --remote-debugging-port=9222 --user-data-dir=$HOME/.cache/tinylm-browser
# through a proxy:
chromium --remote-debugging-port=9222 --user-data-dir=$HOME/.cache/tinylm-browser \
  --proxy-server=socks5://127.0.0.1:1080
```

The separate `--user-data-dir` keeps this browser apart from your everyday profile; you can
log in to sites in it, and the crawler sees them as you do. If the browser listens elsewhere,
set `TINYLM_CDP_URL` (default `http://127.0.0.1:9222`). robots.txt is still read over plain
HTTP.

### Web search

- **SearXNG** (default, free, no key): `make searxng` starts it in Docker or Podman on
  `http://127.0.0.1:8888` and `make searxng-stop` stops it. Its settings are in
  `<data dir>/searxng/settings.yml`: choose search engines there or give it an outgoing proxy,
  then run `make searxng` again. To use a SearXNG you already run, set `TINYLM_SEARXNG_URL` and
  make sure `json` is listed in its `search.formats`.
- **Brave Search API**: `TINYLM_SEARCH_PROVIDER=brave` and `TINYLM_BRAVE_API_KEY=...` (free tier
  at https://brave.com/search/api/).

### Hardware

- `TINYLM_THREADS`: CPU threads for answers (default: physical cores, at most 8; more threads
  do not help generation and slow it down on CPUs with efficiency cores).
- `TINYLM_TORCH=auto make setup-engine` installs PyTorch with GPU support, for training on an
  NVIDIA GPU; `TINYLM_LLAMA=source` with `CMAKE_ARGS="-DGGML_CUDA=on"` builds llama.cpp for
  GPU answers (see the [README](../README.md#python-environment)).

## Where data is stored

Everything lives in `~/.local/share/tinylm-studio` (`TINYLM_DATA_DIR` to change):

| Path | Contents |
|---|---|
| `models/` | Downloaded and fine-tuned models, one folder each |
| `workspace/knowledge.db` | Collections, documents, saved websites, datasets and teachers with their API keys (SQLite) |
| `workspace/training/` | Training runs: settings, progress, logs and checkpoints |
| `workspace/benchmarks/` | Reports of `make bench` |
| `embeddings/` | The embedding model used to search collections |
| `searxng/` | Settings of the local SearXNG |
| `venv/` | The engine's Python environment |

Only one engine can use the data folder at a time. If the app reports that the knowledge base
is locked, an old engine is still running: `pkill -f 'python -m engine'`.

## Command line

Everything the app does with data can also be done from a terminal (`make help` lists all
targets):

| Command | What it does |
|---|---|
| `make models` | List catalog and installed models |
| `make download MODEL=qwen3-0.6b` | Download a model |
| `make bench [MODEL=...]` | Measure answer speed of installed models; the report goes to `workspace/benchmarks/` |
| `make ingest COLLECTION=games FILES=~/docs` | Add files or folders to a collection |
| `make crawl COLLECTION=games URLS="https://..." [PAGES=50]` | Crawl websites into a collection; run again for the next new pages |
| `make search Q="elden ring review" [VIDEO=1] [ADD=games]` | Search the web, optionally adding the results |
| `make searxng` / `make searxng-stop` | Start or stop the local SearXNG |
| `make run-engine [PORT=8765] [ECHO=1]` | Run only the engine, for API clients |
| `make health` / `make info` | Check a running engine / show the paths and Python in use |

More options are available directly, e.g. `python -m engine crawl --help`
(`--max-depth`, `--per-minute`, `--browser`, `--ignore-robots`) and `python -m engine search --help`
(`--limit`, `--language`).
