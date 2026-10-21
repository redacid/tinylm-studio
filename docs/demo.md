# Walkthrough, tour and API GIFs

Three animated GIFs show the app, each recorded by a script from a scenario, with one GIF per
caption language (English and Ukrainian):

- **Walkthrough** (`make demo-gif`): from a website to a trained model.
- **Tour** (`make demo-tour`): every tab, what can be added where and what the options are for.
- **API** (`make demo-api`): the engine's public API tried in Swagger UI.

Nobody has to click through the app: the script drives the real desktop app, so the GIFs are
remade whenever the screens change.

## The walkthrough

`make demo-gif` records an animated walkthrough of the app: it crawls a website into a
collection, makes a dataset of questions and answers from it, fine-tunes a model on the dataset
and asks the new model a question in the chat. Every step has a caption, and one GIF is made
per caption language:

- `docs/images/walkthrough-en.gif` (English)
- `docs/images/walkthrough-uk.gif` (Ukrainian)

## The tour

`make demo-tour` goes through every tab: Data (collections, files, web search, websites and
their crawl options), SQL, Datasets (import, examples written by hand or by a teacher, the
held-out share), Training and what each of its options does, Evaluation, Models (Hugging Face,
recommended models, teachers), Chat (model, data, system prompt, sources, the JSON of an
exchange) and each part of Settings. A frame and the pointer show what each caption is about.

It adds a small collection (`scripts/demo/sample-notes.md`) and a dataset
(`scripts/demo/examples.jsonl`) of its own, opens the forms to show them and closes them without
saving. It needs only the base model (`make download MODEL=qwen3-0.6b`), which answers one
question in the chat, and takes a few minutes. The GIFs (about four minutes long) are
`docs/images/tour-en.gif` and `tour-uk.gif`; the scenario is
[scripts/demo/tour.yaml](../scripts/demo/tour.yaml).

## The API

`make demo-api` opens the engine's Swagger UI (`/docs`, see [api.md](api.md#openapi-schema)) in
the app's window and tries the public endpoints: `GET /health`, `GET /v1/models`,
`POST /v1/chat/completions` (once plainly, once with `collection`, so the answer comes with its
sources) and `POST /v1/messages`, ending at `/openapi.json`. The management API (`/api/...`)
that the desktop app uses is hidden on the page. When the engine has a `TINYLM_API_KEY` (from
`.env`), the scenario enters it with **Authorize** first, shown as dots.

It adds a collection from `scripts/demo/sample-notes.md` without recording it, asks
`qwen3-0.6b` (`VARS="model=<id>"` for another installed model), and takes a few minutes. Swagger
UI loads its scripts from cdn.jsdelivr.net, so it needs the internet. The GIFs are
`docs/images/api-en.gif` and `api-uk.gif`; the scenario is
[scripts/demo/api.yaml](../scripts/demo/api.yaml).

## Running it

```bash
make setup                       # once: the app and its engine
make download MODEL=qwen3-0.6b   # once: the base model the scenario trains
make demo-gif                    # minutes to half an hour on a CPU, mostly the teacher and training
make demo-tour                   # a few minutes
make demo-api                    # a few minutes
```

Requirements: `ffmpeg` (it makes the GIFs). On a computer without a display (a server, a
container), `xvfb-run` is used for the app's window (`apt install xvfb`).

The app's window opens and the script works in it: leave it alone until the script ends. The
recording does not touch your own data: the engine works in a folder of its own,
`<data dir>/demo` (by default `~/.local/share/tinylm-studio/demo`), which is emptied at the start of
each run. Installed models are linked into it, not copied, and the embedding model is shared.

Options:

| Option | Default | What it does |
|---|---|---|
| `SITE=https://…` | `https://docs.python.org/3/tutorial/` | The website to crawl |
| `TEACHER=<model id>` | the smallest installed GGUF model | The model that writes the examples; `TEACHER=` (empty) imports `scripts/demo/examples.jsonl` instead |
| `VARS="pages=4 epochs=2"` | | Any other [scenario variable](#variables) |
| `DEMO_OUT=dir` | `docs/images` | Where the GIFs go |
| `DEMO_SCENARIO=file` | `scripts/demo/scenario.yaml` | Another scenario (`make demo-tour` and `make demo-api` are `DEMO_SCENARIO=scripts/demo/tour.yaml` and `api.yaml`) |

An API teacher such as OpenRouter writes examples much faster than a local model on a CPU. Give
it in the environment; it is then used instead of an installed model:

```bash
TEACHER_URL=https://openrouter.ai/api/v1 TEACHER_MODEL=deepseek/deepseek-chat-v3.1:free \
  TEACHER_KEY=sk-or-... make demo-gif
```

`TEACHER_URL` is any OpenAI-compatible API (left out: OpenRouter).

The script itself has a few more options: `node scripts/demo/record.mjs --help` (`--lang uk` for
one language, `--width 1200`, `--keep` to keep the previous run's data).

If a step fails, the script stops, says which, and saves the screen at that moment as
`<data dir>/demo/error.png`. The frames of the last run stay in `<data dir>/demo/frames`.

## How it works

[scripts/demo/record.mjs](../scripts/demo/record.mjs) starts the desktop app with
[Playwright](https://playwright.dev/docs/api/class-electron) (`playwright-core`, no browser
download) and runs the steps of [scripts/demo/scenario.yaml](../scripts/demo/scenario.yaml):

- Every action is captured as frames, with a drawn pointer gliding to what is clicked and text
  typed a few letters at a time.
- Long waits (crawling, generating examples, training) are captured as a timelapse: the screen is
  captured all along, at most 40–80 frames are kept, and they are played back fast with a
  "Fast-forward ×N" badge. An hour of training takes a few seconds of the GIF.
- Each caption stays on screen long enough to be read.
- The captions are drawn below the frames afterwards, once per language, so all languages come
  from one recording. ffmpeg then makes each GIF with its own palette.

## The scenario

```yaml
name: walkthrough  # the GIFs are walkthrough-<lang>.gif
vars:
  site: https://docs.python.org/3/tutorial/
  pages: 6
models:            # linked from the engine's models folder; must be installed
  - ${base}
labels:
  fast_forward: { en: "Fast-forward ×{speed}", uk: "Прискорено ×{speed}" }
steps:
  - caption:
      en: "2. Add the website: start address, pages to download and link depth"
      uk: "2. Додаємо вебсайт: початкова адреса, кількість сторінок і глибина посилань"
    actions:
      - click: { button: Add website }
      - fill: { css: ".websites form > textarea", text: "${site}" }
      - fill: { label: Pages, text: "${pages}" }
```

A step has a `caption` (one text per language; a new language is just a new key) and
`actions`. `record: false` runs a step without recording it, to prepare something. `when: <variable>` runs a step only when the variable is not empty, `unless:` only when
it is; the default scenario uses them to either generate examples with a teacher or import them.

### Variables

`${name}` anywhere in a step is replaced by the variable. Those of the default scenario:

| Variable | Default | |
|---|---|---|
| `site`, `pages`, `depth`, `per_minute` | Python tutorial, 6, 1, 30 | What is crawled |
| `collection`, `dataset`, `model` | `python-tutorial`, `python-qa`, `python-helper` | Names of what is made |
| `base`, `epochs`, `max_tokens` | `qwen3-0.6b`, 1, 256 | Training; kept small so a CPU finishes in minutes |
| `teacher`, `chunks`, `pairs` | `auto`, 3, 3 | Who writes the examples, from how many chunks, how many pairs each |
| `examples` | `examples.jsonl` | Imported when there is no teacher (paths are relative to the scenario) |
| `question` | How do I create a list in Python? | Asked in the chat at the end |

The script sets `teacher` to the teacher it uses (empty when there is none) and `teacher_name`
to its name as the Generate form lists it.

### Actions

What an action works on is given by one of: `tab` (a tab in the header), `button` (its label),
`placeholder`, `label` (the text of the label around a field), `text` or `css`; `within: <css>`
narrows the search, `nth: 1` takes the second match. Only what is on screen counts, so the same
label on a hidden tab does not get in the way.

| Action | Example | What it does |
|---|---|---|
| `click` | `click: { button: Save and crawl }` | Moves the pointer there and clicks |
| `point` | `point: { label: Epochs, hold: 1 }` | Moves the pointer there and frames it, to explain it; `hold` keeps it longer |
| `open` | `open: "${engine_url}/docs"` | Opens an address in the app's window (`${engine_url}` is the engine's address, `${api_key}` its key or empty) |
| `style` | `style: ".opblock[id*=_api_] { display: none }"` | Adds CSS to the page, e.g. to hide what the scenario is not about |
| `fill` | `fill: { label: Epochs, text: "1" }` | Types the text over what the field holds; `secret: true` shows it as dots |
| `check` | `check: { label: Only under these addresses }` | Ticks a checkbox |
| `select` | `select: { label: Dataset, option: python-qa, partial: true }` | Picks an option by its text (`partial`: one that contains it), waiting for the list to have it |
| `upload` | `upload: { css: "input[type=file]", file: examples.jsonl }` | Gives a file to a file input |
| `scroll` | `scroll: { css: .documents, block: start }` | Scrolls the element into view |
| `wait` | `wait: { for: { css: ".badge.done" }, timeout: 3600, timelapse: true }` | Waits until a target is shown (`for`) or gone (`gone`), up to `timeout` seconds; `timelapse` captures the wait fast-forwarded |
| `pause` | `pause: 2` | Keeps the screen for that many seconds |
