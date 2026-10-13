# Running the engine in Kubernetes

**Target (decided 2026-10-13):** the engine runs as a container in a Kubernetes cluster. The
desktop app, scripts and any OpenAI- or Anthropic-compatible client talk to it over HTTP. Running
the engine on the user's own machine, started by the desktop app, stays supported for
development and single-user use.

Every change to the engine has to keep working in a container: configured only by environment
variables, with no desktop, no browser and no home folder of a real user around it. This page
lists what that means for the code today and what is still missing.

## What already fits

| Need | How the engine meets it |
|---|---|
| Configuration | Only `TINYLM_*` environment variables (see [api.md](api.md#settings)): a ConfigMap for settings, a Secret for `TINYLM_API_KEY`, `HF_TOKEN` and the search providers' `TINYLM_*_API_KEY`. No config files to mount; search settings changed in the app are saved in the data folder and win over the variables. |
| Listening outside the pod | `TINYLM_HOST=0.0.0.0` (the default `127.0.0.1` only serves the pod itself); `TINYLM_PORT`, 8765 by default. |
| Authentication | `TINYLM_API_KEY`: every API call then needs `Authorization: Bearer <key>` or `x-api-key: <key>`. Always set it in a cluster. |
| Probes | `GET /health` answers without an API key; use it for liveness and readiness. |
| State | Everything the engine keeps is under `TINYLM_DATA_DIR` (see the table in the [user guide](user-guide.md#where-data-is-stored)): mount a PersistentVolume there. |
| Logs | The API server logs to stdout. Background jobs also write `worker.log` in their folder under `workspace/`. |
| Lifecycle | Leave `TINYLM_EXIT_WITH_PARENT` unset: it only makes sense when the desktop app starts the engine. |

## Rules the deployment must follow

1. **One engine per data folder.** `workspace/knowledge.db` is SQLite and accepts one writing
   process; models are loaded into that process's memory. Run one replica (`replicas: 1`,
   `strategy: Recreate`, so the old pod has stopped before the new one opens the database) on a
   `ReadWriteOnce` volume. More replicas need the state moved out of SQLite first; this is an
   open question.
2. **Background jobs live in the pod.** Training and GGUF export run as child processes of the
   engine; crawls, dataset generation and evaluation run as its threads. A pod restart stops all
   of them: training, export and evaluation are then shown as `interrupted`, and crawls and
   generation simply end with what they had saved. They have to be started again. Avoid
   evictions while one runs, e.g. with a PodDisruptionBudget.
3. **Size the limits for the largest job.** Answering with a 1B model needs a few GB of memory;
   fine-tuning and exporting need roughly two to three times the model's size more. An
   out-of-memory kill takes the whole engine down, not only the job.
4. **Set `TINYLM_THREADS` to the pod's CPU limit.** The engine counts the node's cores
   (`os.cpu_count()`), not the container's limit, and would start too many threads. The
   memory reported by `/api/system` is also the node's.
5. **Outgoing network.** Allow what the features in use need: Hugging Face (model and
   embedding model downloads), PyPI (the GGUF converter, downloaded on the first export, or set
   `TINYLM_LLAMA_CPP_DIR` to a copy built into the image), the teacher providers (OpenRouter,
   Anthropic and others), the hosted search provider in use (Brave, Tavily, Exa, Perplexity,
   Serper), the sites being crawled, and the proxy if `TINYLM_PROXY` is set.
6. **Helpers are separate services.** SearXNG (`TINYLM_SEARXNG_URL`) and a Chrome for sites that
   need JavaScript (`TINYLM_CDP_URL`) are not started by the engine in a cluster: run them as
   their own Deployments or as sidecars, and point the settings at them. The defaults point to
   `127.0.0.1`, which in a pod is the pod itself.
7. **Paths are the pod's paths.** `python -m engine ingest <folder>` reads folders inside the
   container, and file sources in the chat point into the container too, so they are not links
   on a user's machine. In a cluster, add files through the API (`POST
   /api/collections/{name}/documents`) or the Data tab.
8. **Secrets in the volume.** Teacher API keys are stored in `knowledge.db` and search API keys
   saved in the app in `workspace/search-settings.json`; protect the volume
   and its backups like a Secret.
9. **First start downloads.** The embedding model goes to `<data dir>/embeddings/` on first use
   and models are downloaded on request. Keep them on the volume, or build them into the image to
   start without internet access.

## Not done yet

- A container image (Dockerfile) with the engine, its CPU-only dependencies and, optionally, the
  embedding model and the GGUF converter built in.
- Kubernetes manifests or a Helm chart: Deployment, Service, PersistentVolumeClaim, Secret,
  ConfigMap, probes.
- The desktop app connecting to a remote engine (an engine URL and API key in Settings) instead
  of starting its own.
- Reading the container's CPU and memory limits (cgroups) instead of the node's, so
  `TINYLM_THREADS` does not have to be set by hand.
