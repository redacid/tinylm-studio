# TODO

## Faster answers

Measure first: compare `prompt_ms` and `predicted_ms` in the chat stats line on a slow RAG answer,
then add a `make bench-rag` target to compare before/after.

Engine settings (cheap, low risk):

- [ ] Set `n_threads_batch` to the same value as `n_threads`. llama-cpp-python defaults it to
      `cpu_count()` (20 on an i9-12900H), so prompt processing still runs on E-cores and hyper-threads.
- [ ] Enable `flash_attn`.
- [ ] Put retrieved sources into the last user message instead of the system prompt, so the
      llama.cpp KV cache can reuse the unchanged conversation prefix between turns.
- [ ] Try `rag_top_k = 3` and/or shorter chunks to shrink the prompt.
- [ ] Prompt lookup decoding (`LlamaPromptLookupDecoding`): RAG answers often copy text from the
      sources, so draft tokens are accepted often.

Model changes:

- [ ] Benchmark Q4_0 against Q4_K_M: llama.cpp repacks Q4_0 for AVX2, which speeds up prompt processing.
- [ ] Compare answer quality of Qwen3 1.7B (~11 tok/s) with Gemma 3 4B (~8 tok/s) on the game scenario.
- [ ] Investigate trimming Gemma 3's 262k vocabulary to uk/ru/en tokens (smaller output layer; risky).
- [ ] Layer pruning only together with fine-tuning or distillation (Phase 3).

Models screen:

- [ ] Cancel a running download (huggingface_hub has no cancel; run it in a worker process).
- [ ] Load and test a model downloaded from a search right on the Models screen.
- [ ] Compare on Evaluation: a model fine-tuned with sources in the prompt and its collection
      against the same model fine-tuned on plain pairs.

## Dataset generation (Phase 4)

- [ ] Real run with OpenRouter free models on the game collections: compare a few models
      (Llama 3.3 70B, Qwen3, Gemma 3 27B, DeepSeek V3) on quality of uk/ru pairs and on how often
      the check drops pairs; pick a recommended default.
- [ ] Local teacher on the target machine: minutes per chunk with gemma-3-4b-gguf.
- [ ] Subscription teachers (proof of concept): try Claude Code, Codex CLI and Gemini CLI logged
      in on the target machine; check their error texts for a reached usage limit and a missing
      login against the ones the engine expects; measure seconds per request.
- [ ] Log each subscription tool in from Settings › Subscriptions with a real account, on this
      computer and on a RunPod pod: check the link, the code step and the account shown
      (Claude Code's `auth status` may not name the account).
- [ ] Evaluation on the target machine: qwen3-0.6b against its fine-tune on a generated
      dataset, judged by an OpenRouter free model; check that judge scores agree with a manual
      read of 20 answers.
- [ ] Evaluate a fine-tuned model right from the Training screen when a run finishes.
- [ ] Optional: a second judge, or the same judge twice, to measure how stable the scores are.
- [ ] Optional: multi-turn examples (follow-up questions) and paraphrased questions.

## Fine-tuning (Phase 3)

- [ ] Real run on the target machine: qwen3-0.6b on ~1,000 examples, 3 epochs; note time per
      step and peak RAM (cloud estimate with a random model of the same size: 15 s/step on
      4 cores, 6.3 GB).
- [ ] GPU run (CUDA PyTorch build, `TINYLM_TORCH=auto`).
- [ ] Gradient checkpointing as an option for longer examples (less memory, ~30% slower).
- [ ] Export qwen3-0.6b fine-tune to GGUF on the target machine (Q8_0 and Q4_K_M): time of the
      export, tokens/s against the adapter through PyTorch, and Evaluation scores of all three.

## Kubernetes and RunPod

The engine runs in a container in a Kubernetes cluster or a RunPod pod; see
[docs/deployment.md](docs/deployment.md).

- [x] Container image (`engine/Dockerfile`, `make image`): CPU and CUDA variants with llama.cpp
      and PyTorch; refuses to start without `TINYLM_API_KEY`.
- [x] Image tags name the engine version and git commit (`0.1.0-5a0d901-cuda`); the commit is in
      the image labels, `/health` and the app's header.
- [ ] Optional build arguments to include the embedding model and the GGUF converter.
- [x] Releases: `make release VERSION=x.y.z` sets the version, writes CHANGELOG.md and tags; a
      pushed tag runs CI and publishes the deb and rpm as a GitHub release; builds at a release
      tag are named by the tag, elsewhere by the commit ([docs/releasing.md](docs/releasing.md)).
- [x] GitHub Actions: CI on pushes and pull requests, Release on tags, engine images on demand.
- [ ] Run the workflows on github.com/redacid/tinylm-studio for the first time (first release).
- [ ] Build the images on GitHub (manual workflow) and decide where they live (ghcr.io storage,
      runner disk and time for the CUDA base).
- [x] Build and push the `cuda` image and start it on a real RunPod GPU pod (RTX 2000 Ada).
- [ ] Run a fine-tune on a real RunPod GPU pod.
- [ ] Manifests or a Helm chart: Deployment (1 replica, `Recreate`), Service, PVC
      (`ReadWriteOnce`) for `TINYLM_DATA_DIR`, Secret, ConfigMap, `/health` probes.
- [ ] Read CPU and memory limits from cgroups for the default `TINYLM_THREADS` and `/api/system`.
- [x] Desktop app: connect to a remote engine (URL and API key in Settings > Engine) instead of
      starting its own.
- [x] RunPod in Settings > Engine: create a pod from the image, start, stop, terminate, stop on quit.
- [x] RunPod: GPU, CPU, data center and network volume chosen from RunPod's catalog, with prices
      and what is free now; CPU-only pods.
- [x] RunPod: **Show request** shows the pod request before it is sent, with the secrets hidden.
- [x] Move a collection between engines (this computer and a pod): **Export** and
      **Import collection** on the Data page.
- [x] Logs: the engine and the app log to files; Settings › Logs shows both, colored and
      filtered, also the engine's in a RunPod pod.
- [x] SQL tab: SQL queries over the data, with the tables' fields, types and example
      queries; `DELETE` asks first and removes the rows with their chunks and index.
- [x] Icons on tabs, buttons, run states, the engine status and log levels, colored by what
      they do (the bundled Tabler Icons font).
- [x] The app logo next to the tinylm-studio name in the window header.

## API description (OpenAPI)

The engine serves its OpenAPI schema at `/openapi.json`, Swagger UI at `/docs`;
`make openapi` prints it. See [docs/api.md](docs/api.md#openapi-schema).

- [x] Complete schema: response models for every JSON endpoint (`engine/api/schemas.py`,
      checked against what handlers send in the tests), error responses, the media types of
      streams and files, and the `HTTPBearer`/`APIKeyHeader` security schemes for **Authorize**
      and tools that import the schema (Postman, Insomnia, Bruno).
- [x] TypeScript types for `desktop/src/renderer/api.ts` generated from the schema
      (`make api-types`, openapi-typescript) instead of written by hand.
- [ ] Optional: field descriptions in the schema (Swagger and the generated types show only
      field names; the meaning is in code comments).
- [ ] Optional: the events of streams (chat chunks, training events) in the schema; the app
      keeps their TypeScript types by hand.
