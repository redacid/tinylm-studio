# Walkthrough GIF

`make demo-gif` records an animated walkthrough of the app: it crawls a website into a
collection, makes a dataset of questions and answers from it, fine-tunes a model on the dataset
and asks the new model a question in the chat. Every step has a caption, and one GIF is made
per caption language:

- `docs/images/walkthrough-en.gif` (English)
- `docs/images/walkthrough-uk.gif` (Ukrainian)

Nobody has to click through the app: the script drives the real desktop app, so the GIF is
remade whenever the screens change.

## Running it

```bash
make setup                       # once: the app and its engine
make download MODEL=qwen3-0.6b   # once: the base model the scenario trains
make demo-gif                    # minutes to half an hour on a CPU, mostly the teacher and training
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
| `DEMO_SCENARIO=file` | `scripts/demo/scenario.yaml` | Another scenario |

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
`actions`. `when: <variable>` runs a step only when the variable is not empty, `unless:` only when
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
| `fill` | `fill: { label: Epochs, text: "1" }` | Types the text over what the field holds |
| `check` | `check: { label: Only under these addresses }` | Ticks a checkbox |
| `select` | `select: { label: Dataset, option: python-qa, partial: true }` | Picks an option by its text (`partial`: one that contains it), waiting for the list to have it |
| `upload` | `upload: { css: "input[type=file]", file: examples.jsonl }` | Gives a file to a file input |
| `scroll` | `scroll: { css: .documents, block: start }` | Scrolls the element into view |
| `wait` | `wait: { for: { css: ".badge.done" }, timeout: 3600, timelapse: true }` | Waits until a target is shown (`for`) or gone (`gone`), up to `timeout` seconds; `timelapse` captures the wait fast-forwarded |
| `pause` | `pause: 2` | Keeps the screen for that many seconds |
