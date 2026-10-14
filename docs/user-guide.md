# User guide

How to use tinylm-studio, screen by screen, and how to set it up for your network and hardware.
For the HTTP API see [api.md](api.md); for the design and roadmap see the
[implementation plan](implementation-plan.md).

- [Start](#start)
- [Chat](#chat)
- [Models](#models)
  - [Model settings: context, threads, sampling, thinking](#model-settings)
- [Data: collections, files, web search and websites](#data)
- [Datasets](#datasets)
  - [Generating examples from your documents](#generating-examples)
- [Teachers: OpenRouter, Anthropic and local models](#teachers)
- [Training](#training)
  - [Exporting a fine-tuned model to GGUF](#exporting-to-gguf)
- [Evaluation](#evaluation)
- [Settings](#settings)
  - [Chat settings](#chat-settings)
  - [Search settings: providers and site filters](#search-settings)
  - [Crawler settings: robots.txt, rate, defaults](#crawler-settings)
  - [Proxy for crawling and search](#proxy)
  - [Crawling through a real browser (CDP)](#browser-cdp)
  - [Web search: SearXNG setup](#web-search)
  - [Hardware: threads and GPU](#hardware)
- [Where data is stored](#where-data-is-stored)
- [Command line](#command-line)
- [Troubleshooting](#troubleshooting)
  - [A gated model does not download (401 / 403)](#a-gated-model-does-not-download)

## Start

```bash
make setup   # once: the engine's Python environment and the desktop app's dependencies
make run     # start the app; it starts the engine and shows "Engine ready" in the top right
```

After pulling changes that add Python packages (see `engine/pyproject.toml`), run
`make setup-engine` again: the app installs the engine only on its very first start.

Settings go into `.env` in the repository (`cp .env.example .env`); every `make` target and the
app read it.

Switching tabs stops nothing. Crawls, example generation, training, evaluation and exports run
in the engine; each tab keeps its form, selection and progress while another one is shown, and
refreshes its lists when you come back. Closing the app stops the engine, and with it a run in
progress.

The app follows the system's light or dark theme. Colors carry meaning across the tabs: blue for
section titles, table headers, the selected item and the main button of a form; green for the
assistant's answers and finished runs; red for errors and for the button that confirms a delete.

## Chat

![Chat with answers from a collection](images/chat.png)

- Pick a **model** in the first list. Downloaded models, fine-tuned models and (with
  `TINYLM_ENABLE_ECHO=true`) the test model `echo` are offered.
- Pick **Data: \<collection\>** to answer from your uploaded data. The answer lists the sources
  it was given; the model is asked to cite them as [1], [2]. A source from a website opens its
  page in the browser (the site is shown after the name); a file added with `make ingest` is
  shown in the file manager. Files uploaded in the app keep only their name, so they are not
  links. Hover a source to see its address and the text the answer got from it.
- The **system prompt** sets the assistant's role or tone for the conversation. It starts with
  the default one from [Chat settings](#chat-settings).
- The line under each answer shows where the time went: loading the model, searching the data,
  reading the prompt (tokens per second) and writing the answer. Use it to see what is slow.
- **New chat** clears the conversation; **Stop** ends an answer early.
- The conversation, the unsent text and the chosen model, data and system prompt stay when you
  switch to another tab, and an answer being written keeps going. Coming back, the chat picks
  up models and collections added on the other tabs. The conversation and the chosen model and
  data are also kept after the app is closed; both can be turned off in
  [Chat settings](#chat-settings).
- The **{ }** button in the corner of each message shows the JSON of that exchange with the
  engine. On a question it opens **Sent**: the request body sent to `/v1/chat/completions`, and
  the request as the model got it, with the system prompt and the sources found in the data.
  On an answer it opens **Received**: the HTTP status and every streamed chunk, including
  `sources` and `timings` (with streaming off, the one response). **Copy** copies what is shown;
  Esc closes it.

![The JSON sent for a question](images/chat-trace.png)

The same models answer through the OpenAI- and Anthropic-compatible APIs, so any client or SDK
can use them (see [api.md](api.md)).

## Models

![Models: installed models and a download from Hugging Face](images/models.png)

- **Installed models** lists every model in `<data dir>/models/`: its format, size on disk and
  the Hugging Face repo (and file) it came from; **loaded** marks the one in memory. Any GGUF
  file or Hugging Face model folder copied into `<data dir>/models/<name>/` appears too.
- **Delete** removes the model's folder and its settings. Deleting a base model that fine-tuned
  models need warns first: they stop working until it is downloaded again.
- **Settings** opens the model's own settings (see below); the ones changed from the defaults
  are listed under its name.
- **Find models on Hugging Face** searches all open models, most downloaded first. Pick
  **GGUF** for fast answers through llama.cpp, or **PyTorch** for models that can be fine-tuned.
  **Choose** lists a GGUF repo's files: one per quantization (Q4_K_M is a good balance, Q8_0 is
  close to the original, smaller ones lose quality) with its size; **Download** gets that one
  file (all parts of a split one). For PyTorch, **Download** gets the weights, config and
  tokenizer. The model is saved under the file or repo name, e.g. `gemma-3-4b-it-q4_k_m`, and
  the progress shows above the search.
- **Recommended models** are the tested ones, downloaded with one click. GGUF models (`-gguf`)
  run through llama.cpp and are the fast choice for chatting on a CPU; the default is
  `gemma-3-4b-gguf`. Models with PyTorch weights (e.g. `qwen3-0.6b`) are slower to chat with but
  are the ones that can be fine-tuned.
- Gemma and Llama models are gated: accept their license on Hugging Face and put your token into
  `.env` as `HF_TOKEN=...`. A token also makes downloads faster. `.env` is read by `make`, so
  start the app with `make run` (or export `HF_TOKEN` yourself). If the token is missing,
  the gated models are marked in red.
- If a gated model does not download, see
  [Troubleshooting](#a-gated-model-does-not-download).
- **Fine-tuned models** show the base model, the dataset and the final loss. Deleting one
  removes its folder; the base model stays. **Export to GGUF** turns one into a fast llama.cpp
  model, see [Exporting to GGUF](#exporting-to-gguf).

![Searching Hugging Face and choosing a GGUF file](images/models-search.png)

### Model settings

![The settings of one model](images/model-settings.png)

Each model keeps its own settings; an empty field uses the default shown in it.

| Setting | Default | What it does |
|---|---|---|
| Context size | 4096 | GGUF only. Tokens the model sees at once: prompt, sources and answer. More takes more memory. |
| GPU layers | -1 (all) | GGUF only. Layers put on the GPU when llama.cpp was built with GPU support; 0 is CPU only. |
| Batch size | 512 | GGUF only. Prompt tokens read in one step; larger reads long prompts faster. |
| CPU threads | `TINYLM_THREADS` | Threads for this model. |
| Max answer length | 512 | Tokens an answer may take. |
| Temperature | 0.7 | 0 always picks the most likely word; higher is more varied. |
| Top P | 1 | Picks only from the most likely words that make up this share. |
| Top K | runtime's (40 in llama.cpp) | Picks only from this many most likely words; 0 turns it off. |
| Min P | runtime's (0.05 in llama.cpp) | Drops words less likely than this share of the most likely one. |
| Repetition penalty | `TINYLM_REPETITION_PENALTY` (1.1) | Above 1 makes repeating words less likely. |
| Thinking | off | Lets models with a reasoning mode (Qwen3) think before they answer: better, but slower. |
| Collection | none | Optional. Requests that name no data answer from this collection, with sources. |
| Sources per answer | `TINYLM_RAG_TOP_K` (4) | How many pieces of the collection an answer gets. |

The first four are used when the model is loaded, so saving a change to them unloads it and the
next request loads it again. The answer settings apply at once, to every request that does not
set them itself: API clients that leave them out, and the chat for the fields left empty in
[Chat settings](#chat-settings). The settings are kept by the engine in
`<data dir>/models/.model-settings.json`.

**A model with its own collection.** A collection can't be put into the model's weights
reliably: fine-tuning a small model on facts teaches it the training answers, but on new
questions it still invents (compare the training and held-out loss of a fine-tuned model).
Instead, a model can be given a collection: in the chat, picking the model selects its
collection (it can still be changed, or set to **No data**), and API requests to it that name
no `collection` answer from it. An exported model can take the collection along, see
[Exporting to GGUF](#exporting-to-gguf), and **With sources in the prompt** on the generation
form teaches a model to answer from the sources it is given.

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
  - robots.txt is followed unless it is turned off in
    [Crawler settings](#crawler-settings), and a site that answers "too many requests" is paused
    for as long as it asks.
  - A new website starts with the pages, depth and requests per minute from Crawler settings.

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
  When the check reply is not JSON, the pairs are kept unchecked and the request shows the start
  of the reply.
- **With sources in the prompt** (off by default): each example is written the way the chat asks
  a model with data. Its system prompt gets four sources found in the collection for the
  question, among them the text the pair was written from, and the answer ends with that
  source's number, like `[2]`. A model trained on them learns to answer from the sources it is
  given and to cite them; use it for a model that will answer with a collection.

The progress shows chunks read, pairs written and added, duplicates, and why pairs were dropped
(wrong language, too short or long, not supported by the text). Below it, every request to
the teacher is listed, newest first (the last 10; **Show all** for the rest): the time, the
document, whether it wrote the pairs or checked them, the **model** and **provider** that
answered, and the result (pairs written, pairs supported by the text and how many of them
were added) or, in red, the error.
Model and provider are what the API reports: OpenRouter can route a request to another model
than the one asked for (`openrouter/auto`, fallbacks) and to one of several upstream providers,
and names the provider in its errors too; when an answer names none, the model asked for is
shown. **Stop** ends the run after the
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

**Test** sends a tiny request and shows the answer, with the model and provider that answered,
or the provider's error. **Requests per
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

### Exporting to GGUF

A fine-tuned model is a small LoRA adapter on top of its base model and answers through PyTorch,
which is several times slower on a CPU than the GGUF models. **Export to GGUF** on the Models
screen merges the adapter into the base model's weights and writes one GGUF file that llama.cpp
runs, as fast as any other GGUF model of that size. The result is a new model in the list
(e.g. `game-hints-0.6b-q8_0`) that does not need its base model any more.

![Exporting a fine-tuned model](images/export-form.png)

Pick the quantization and, optionally, a name and a collection to take along (**With data**).
With a collection, the new model folder also holds `collection.jsonl` (its documents, split
into pieces) and the model answers from it by default. Copied with the folder to another
machine, the engine there creates the collection when it starts (if there is none of that
name) and indexes it with its own embedding model.

| Quantization | Size of a 0.6B model | Quality |
|---|---|---|
| **Q8_0** (default) | ~0.6 GB | Practically the same as the original |
| **Q4_K_M** | ~0.4 GB | Smaller and faster, a small loss; good for 1.7B and larger |
| **F16** | ~1.2 GB | The original weights, the largest |

The export runs in the background in its own process: getting the converter, merging, converting
and quantizing. It takes a minute or two for a 0.6B model and needs about twice the model's size
in free memory and disk while it runs.

![The exported model on the Models screen](images/models-trained.png)

- The converter is llama.cpp's own `convert_hf_to_gguf.py`, taken from the source package of the
  installed llama-cpp-python, so the file matches the llama.cpp that runs it. The first export
  downloads it from PyPI (about 75 MB, of which 2 MB are kept in `<data dir>/tools/`). Offline,
  point `TINYLM_LLAMA_CPP_DIR` to a llama.cpp checkout instead.
- Compare the exported model with the adapter on the [Evaluation](#evaluation) screen: Q8_0
  should score the same; Q4_K_M a little lower.
- If the base model's tokenizer is unknown to llama.cpp, the export still finishes and says so;
  the catalog's base models are all known.
- A failed export names the error and the step it happened in; for an error without a message,
  such as a failed check in llama.cpp's converter, it shows the line of code that failed. The
  whole log is in `<data dir>/workspace/exports/<id>/worker.log`.
- Gemma 3 models are converted with the base model's SentencePiece `tokenizer.model`, which the
  export copies next to the merged model.
- "Export failed: ModuleNotFoundError while ..." means the engine's environment is older than the
  code; see [An export fails with ModuleNotFoundError](#an-export-fails-with-modulenotfounderror).

## Evaluation

![Comparing the base model with the fine-tuned one](images/evaluation.png)

Evaluation shows whether fine-tuning helped: every chosen model answers the held-out questions
of a dataset (the examples kept out of training by **Evaluation %**), and the answers are
compared with the dataset's own answers.

![Starting an evaluation](images/evaluation-form.png)

1. Open **Evaluation** → **New evaluation** and pick a dataset. The fine-tuned models of that
   dataset and their base models are selected already; add others to compare (up to four).
2. Optionally pick a **judge**: a teacher from the Models screen ([Teachers](#teachers)). Without
   one, answers are scored by similarity only.
3. **Examples** limits how many held-out questions are used; **Max answer tokens** caps each
   answer. Press **Start evaluation**.

The report has one row per model:

- **Judge score**, 1 to 5: the judge reads the question, the expected answer and every model's
  answer and scores each one (5 is as good as the expected answer, 1 is wrong or invented). The
  answers are shown to it under shuffled letters, so it does not know which model wrote what.
- **Similarity**, 0 to 100 (chrF): how much of the expected answer's text the answer repeats,
  by character sequences, so Ukrainian and Russian word forms still count. It needs no judge
  but rewards wording, not correctness: an answer in the right format with a wrong number still
  scores high. Use it to see that a model picked up the style; trust the judge for content.
- **Seconds per answer** and **answer length**.

Below, every question shows the expected answer and each model's answer with its scores and the
judge's reason, so you can see *what* changed. Models answer one after another on this
computer, the judge after all answers are written. **Stop** keeps the report of what was done;
reports are kept until you delete them.

## Settings

The **Settings** tab holds the app's own settings, saved as you change them and kept after a
restart, except the search and crawler settings, which the engine keeps. The engine's other
settings (proxy, hardware, folders) go into `.env`; see below.

### Chat settings

![Chat settings](images/settings.png)

| Setting | Default | What it does |
|---|---|---|
| Default system prompt | empty | The system prompt the chat starts with; changing it here replaces the one above the chat. |
| Max answer length | model's, 512 | Tokens an answer may take (16 to 8192). |
| Temperature | model's, 0.7 | 0 always picks the most likely word, so the same question gets the same answer; higher is more varied. |
| Top P | model's, 1 | Picks only from the most likely words that together make up this share; 1 turns it off. |
| Repetition penalty | model's, then engine's 1.1 | Above 1 makes repeated words less likely; raise it (1.2 to 1.3) when a small model loops. |
| Sources from data | engine's, 4 | How many chunks of the chosen data an answer gets. More can answer more, but the prompt is longer and slower. Empty uses `TINYLM_RAG_TOP_K`. |
| Stop sequences | none | One per line; the answer ends before any of them. |
| Show the answer as it is written | on | Off: the chat asks for one complete response (`"stream": false`) and shows it at once. |
| Show the { } button | on | The button with the JSON sent and received on each message. |
| Start with the model and data chosen last time | on | Otherwise the chat starts with the default model and no data. |
| Keep the conversation after the app is closed | on | The conversation and the system prompt come back on the next start; **New chat** still clears it. |

An empty answer setting uses the chosen model's own [settings](#model-settings) from the Models
tab, then the engine's default. **Reset to defaults** restores the table's defaults. The settings
live in the desktop app's own storage (`~/.config/tinylm-studio-desktop/`), not in `.env`, and
apply to the chat screen only; API clients pass the same values in their requests (see
[api.md](api.md)).

### Search settings

![Search settings](images/search-settings.png)

The search on the Data tab (**Find on the web**) and `make search` use the provider and site
filters chosen here. Unlike the chat settings, they are kept by the engine (in
`<data dir>/workspace/search-settings.json`), so they also apply when the engine runs in a
cluster; the `TINYLM_SEARCH_*` and `TINYLM_*_API_KEY` variables only give the defaults.

| Provider | Key | Videos | Filters sites itself | Options |
|---|---|---|---|---|
| **SearXNG** (default) | no, self-hosted (see [Web search](#web-search)) | yes | no | Address |
| **Brave Search** | [brave.com/search/api](https://brave.com/search/api/) | yes | no | |
| **Tavily** | [app.tavily.com](https://app.tavily.com/) | through YouTube and Vimeo | yes | Depth (advanced costs twice the credits), topic (general, news) |
| **Exa** | [dashboard.exa.ai](https://dashboard.exa.ai/) | through YouTube and Vimeo | yes | Search type (auto, fast, instant) |
| **Perplexity** | [docs.perplexity.ai](https://docs.perplexity.ai/) | through YouTube and Vimeo | up to 20 sites | |
| **Serper** (Google's results) | [serper.dev](https://serper.dev/) | yes | no | Country (`ua`, `us`, ...) |

- **API key**: pasted once and never shown again; the card shows **key set** or **no key**.
  **Remove** deletes it. Keys of other providers stay saved when you switch.
- **Only these sites** (empty: any site) and **Never these sites** (empty: none): one domain
  per line, e.g. `ign.com`; addresses like `https://www.ign.com/games` are reduced to the
  domain, and subdomains count too (`wiki.example` covers `en.wiki.example`).
- Patterns work too: `*` stands for any characters, dots included. `*.ua` (or `.ua`) is every
  site in the .ua zone, `*.com` every .com site, `*.google.*` Google in any country
  (`google.com`, `google.co.uk`, `maps.google.com`), `steam*.com` both `steampowered.com` and
  `steamcommunity.com`. `*` alone is not accepted: leave the list empty for any site.
- SearXNG itself has no per-search site filter (its `hostnames` plugin only removes sites for
  every search, in its own settings), and neither do Brave and Serper. With them the search asks
  for `site:<domain>` results for each listed site (up to 5; `*.ua` becomes `site:.ua`) and
  drops results from other sites in the engine, fetching up to 3 pages when the filter leaves
  too few. Tavily, Exa and Perplexity get the lists in their API (Perplexity also takes `.ua`
  for `*.ua`); the results are checked again anyway. A pattern a provider cannot take, like
  `*.google.*`, is applied by the engine alone: with one in **Only these sites** the provider
  searches everywhere and the engine keeps the matching results.
- **Test search** saves and searches for one word, to check the provider and key. Above the
  results on the Data tab, a line shows the provider, the filters and how many results were
  hidden by them.

### Crawler settings

![Crawler settings](images/crawler-settings.png)

Used when pages and websites are added on the Data tab and by `make crawl`. Like the search
settings, they are kept by the engine (in `<data dir>/workspace/crawl-settings.json`); the
variables in the table only give the defaults. Changes apply to crawls started after saving.

| Setting | Default | Meaning |
|---|---|---|
| robots.txt | Follow (`TINYLM_CRAWL_RESPECT_ROBOTS`) | **Follow**: each site's robots.txt is read first, pages it disallows are skipped and its `Crawl-delay` slows the crawl down. **Ignore**: robots.txt is not read at all, for every crawl. Ignore it only on sites you may crawl anyway, such as your own: sites can ban crawlers that do. |
| Requests per minute | 20 (`TINYLM_CRAWL_PER_MINUTE`) | To each site, 1 to 120, for new websites and for pages added from search results. |
| Pages | 50 (`TINYLM_CRAWL_MAX_PAGES`) | New pages a new website downloads per run. |
| Depth | 2 (`TINYLM_CRAWL_MAX_DEPTH`) | Link hops from the start addresses for a new website. |
| Fetching | Plain HTTP (`TINYLM_CRAWL_USE_BROWSER`) | **Browser (CDP)**: pages added from search results, and new websites, load in the browser at **Browser address**. A saved website uses its own **Browser (CDP)** box. Setting only the address changes nothing: pages go through the browser only when this or the website's box says so. |
| User agent | `tinylm-studio/0.1 (local research crawler)` (`TINYLM_USER_AGENT`) | How the crawler introduces itself; robots.txt rules are read for this name. |
| Browser address (CDP) | `http://127.0.0.1:9222` (`TINYLM_CDP_URL`) | Where the browser for [Browser (CDP)](#browser-cdp) websites listens. |

Pages, depth, requests per minute and fetching are only where a new website starts: each saved website
keeps its own values, changed in its form on the Data tab.

### Engine settings (.env)

Put these into `.env` (see [.env.example](../.env.example)); the full list is in
[api.md](api.md#settings).

### Proxy

`TINYLM_PROXY` sends crawler traffic, and requests to the hosted search providers (all but
SearXNG), through a proxy:

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
**Browser (CDP)** on a website, choose **Browser (CDP)** under Fetching in
[Crawler settings](#crawler-settings) for pages added from search, or pass `--browser` to
`python -m engine crawl`, to load pages in a real
browser that the engine controls over the Chrome DevTools Protocol.

The quickest is a headless Chrome in Docker or Podman:

```bash
make chrome                                   # on http://127.0.0.1:9222
make chrome CDP_PROXY=socks5://host:1080      # its traffic through a proxy
make chrome CDP_PORT=9333                     # another port; change Browser address to match
make chrome-stop
```

It runs `chromedp/headless-shell` and restarts with Docker. Inside the container, `127.0.0.1`
and `localhost` are the container itself: for a site on your own machine use the host's
address (with Docker on Linux usually `172.17.0.1`), and for a proxy on your machine too. The
browser keeps no logins between restarts.

To log in to sites first, run your own Chrome or Chromium with a separate profile instead:

```bash
chromium --remote-debugging-port=9222 --user-data-dir=$HOME/.cache/tinylm-browser
# through a proxy:
chromium --remote-debugging-port=9222 --user-data-dir=$HOME/.cache/tinylm-browser \
  --proxy-server=socks5://127.0.0.1:1080
```

The separate `--user-data-dir` keeps this browser apart from your everyday profile; you can
log in to sites in it, and the crawler sees them as you do. If the browser listens elsewhere,
change **Browser address** in [Crawler settings](#crawler-settings) (or `TINYLM_CDP_URL`;
default `http://127.0.0.1:9222`). robots.txt, when followed, is still read over plain HTTP.

### Web search

The provider, its key and the site filters are chosen in [Search settings](#search-settings).
For the default, SearXNG (free, no key):

- `make searxng` starts it in Docker or Podman on `http://127.0.0.1:8888` and
  `make searxng-stop` stops it. Its settings are in `<data dir>/searxng/settings.yml`: choose
  search engines there or give it an outgoing proxy, then run `make searxng` again.
- To use a SearXNG you already run, put its address into Search settings (or
  `TINYLM_SEARXNG_URL`) and make sure `json` is listed in its `search.formats`.
- From the command line: `make search Q="elden ring review" INCLUDE=ign.com,gamespot.com` or
  `EXCLUDE=pinterest.com` overrides the saved site filters for that search.

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
| `workspace/search-settings.json` | Search settings changed in the app, with the search API keys |
| `workspace/crawl-settings.json` | Crawler settings changed in the app |
| `workspace/training/` | Training runs: settings, progress, logs and checkpoints |
| `workspace/evaluations/` | Evaluation reports, one JSON file each |
| `workspace/exports/` | GGUF exports: settings, progress and logs |
| `tools/` | llama.cpp's GGUF converter, downloaded on the first export |
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
| `make crawl COLLECTION=games URLS="https://..." [PAGES=50]` | Crawl websites into a collection (pages and depth default to Crawler settings); run again for the next new pages |
| `make search Q="elden ring review" [VIDEO=1] [ADD=games]` | Search the web, optionally adding the results |
| `make searxng` / `make searxng-stop` | Start or stop the local SearXNG |
| `make chrome [CDP_PROXY=...]` / `make chrome-stop` | Start or stop a headless Chrome in Docker for websites with Browser (CDP) |
| `make run-engine [PORT=8765] [ECHO=1]` | Run only the engine, for API clients |
| `make health` / `make info` | Check a running engine / show the paths and Python in use |

More options are available directly, e.g. `python -m engine crawl --help`
(`--max-depth`, `--per-minute`, `--browser`, `--ignore-robots`; options left out come from
[Crawler settings](#crawler-settings)) and `python -m engine search --help`
(`--limit`, `--language`).

## Troubleshooting

### A gated model does not download

Gemma and Llama models are *gated*: Hugging Face serves them only to an account that has
accepted the model's license, and only when the download sends that account's token. The
Models screen shows the reason under the model. Older versions showed Hugging Face's raw error,
for example:

```text
403 Client Error. Cannot access gated repo for url
https://huggingface.co/google/gemma-3-1b-it/resolve/.../tokenizer.json.
Access to model google/gemma-3-1b-it is restricted and you are not in the authorized list.
Visit https://huggingface.co/google/gemma-3-1b-it to ask for access.
```

The HTTP status tells the two cases apart:

| Status | Shown as | Meaning |
|---|---|---|
| 401 | "... is gated and the engine has no Hugging Face token" | The download sent no token. |
| 403 | "Hugging Face refused access ... the token was sent" ("you are not in the authorized list") | The token arrived, but its account may not download this model yet. |

**401, no token:**

1. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   (type *Read*) and put it into `.env` in the repository: `HF_TOKEN=hf_...`.
   `TINYLM_HF_TOKEN` works too; a token saved by `hf auth login` is also picked up.
2. Start the app with `make run` (or the engine with `make run-engine`). Only `make` reads
   `.env`; when the app is started another way, export `HF_TOKEN` in that shell first.
3. The Models screen marks gated models in red while the engine finds no token, so you can
   check without downloading.

**403, the token was sent but access is refused:**

1. Open the model's page (the **Model page** link under the model on the Models screen),
   signed in with **the same account the token belongs to**, and accept the license
   ("Acknowledge license" / "Agree and access repository"). For Gemma access is granted at
   once; for some Llama models the request waits for approval, shown on the same page.
2. If the token is *fine-grained*, edit it at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and enable
   "Read access to contents of all public gated repos you can access". Without it the token is
   refused even after the license is accepted. A classic *Read* token needs nothing more.
3. Press **Download** again; there is no need to restart the app.

Each gated model has its own license: accepting Gemma 3 4B does not give access to Gemma 3 1B
or to Llama.

### An export fails with ModuleNotFoundError

For example, under the model on the Models screen:

```text
Export failed: ModuleNotFoundError while converting to GGUF: No module named 'sentencepiece'
```

The engine's Python environment lacks a package the code needs: the app installs the engine
only on its first start, so packages added later (here `sentencepiece`, added with GGUF export)
are not there until the environment is updated.

1. Stop the app and run `make setup-engine` in the repository. It installs what
   `engine/pyproject.toml` lists into the app's environment and keeps what is already there.
2. Start the app with `make run` and press **Export to GGUF** again.

Qwen models (all the catalog's Qwen 3 models) no longer need `sentencepiece` to export: their
tokenizer is `tokenizer.json`, and the export uses it directly. Models with a SentencePiece
tokenizer (`tokenizer.model`, e.g. Gemma) still need the package.

