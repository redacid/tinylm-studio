# Running the engine in a container: Kubernetes and RunPod

**Target (decided 2026-10-13):** the engine runs as a container in a Kubernetes cluster. The
desktop app, scripts and any OpenAI- or Anthropic-compatible client talk to it over HTTP. Running
the engine on the user's own machine, started by the desktop app, stays supported for
development and single-user use.

Every change to the engine has to keep working in a container: configured only by environment
variables, with no desktop, no browser and no home folder of a real user around it. This page
lists what that means for the code today and what is still missing. The same image also runs
on a [RunPod](#runpod) GPU or CPU pod, which the desktop app can create, start and stop itself.

## What already fits

| Need | How the engine meets it |
|---|---|
| Configuration | Only `TINYLM_*` environment variables (see [api.md](api.md#settings)): a ConfigMap for settings, a Secret for `TINYLM_API_KEY`, `HF_TOKEN` and the search providers' `TINYLM_*_API_KEY`. No config files to mount; search and crawler settings changed in the app are saved in the data folder and win over the variables. |
| Listening outside the pod | `TINYLM_HOST=0.0.0.0` (the default `127.0.0.1` only serves the pod itself); `TINYLM_PORT`, 8765 by default. |
| Authentication | `TINYLM_API_KEY`: every API call then needs `Authorization: Bearer <key>` or `x-api-key: <key>`. Always set it in a cluster. |
| Probes | `GET /health` answers without an API key; use it for liveness and readiness. Every other path, `/api/system` included, needs the key when one is set. |
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
   need JavaScript (`TINYLM_CDP_URL`, or Settings > Crawler) are not started by the engine in a cluster: run them as
   their own Deployments or as sidecars, and point the settings at them. For Chrome,
   `chromedp/headless-shell` (what `make chrome` runs) works as a sidecar listening on 9222;
   mount an `emptyDir` with `medium: Memory` on `/dev/shm`, as `make chrome` gives it `--shm-size=1g`. The defaults point to
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
10. **A model with its data.** A model exported with a collection (its folder holds
    `collection.jsonl`) can be copied into `<data dir>/models/`: at start the engine creates the
    collection if there is none of that name and indexes it in the background, so the first
    answers with sources come once that is done.

## Container image

[engine/Dockerfile](../engine/Dockerfile) builds the engine with llama.cpp and PyTorch in two
variants:

| Variant | Base | For |
|---|---|---|
| `cpu` (default) | `ubuntu:24.04` | Kubernetes nodes and servers without a GPU |
| `cuda` | `nvidia/cuda:12.6.3-runtime-ubuntu24.04` | NVIDIA GPUs (Volta to Hopper): GGUF answers through llama.cpp on the GPU, fine-tuning with CUDA PyTorch; RunPod |

```bash
make image                                   # tinylm-engine:0.1.0-5a0d901-cpu and :cpu
make image VARIANT=cuda                      # tinylm-engine:0.1.0-5a0d901-cuda and :cuda
make image-push VARIANT=cuda IMAGE=docker.io/<you>/tinylm-engine   # after docker login
TINYLM_API_KEY=<key> make run-image          # try it on 127.0.0.1:8765, data in <data dir>/container
```

- **Version and commit.** Every build is tagged `<engine version>-<commit>-<variant>`, e.g.
  `0.1.0-5a0d901-cuda`, a tag that always means the same code, and `<variant>`, which moves to
  the latest build. With uncommitted changes the commit gets `-dirty` (`0.1.0-5a0d901-dirty-cpu`),
  and `make image-push` refuses to push until they are committed (`ALLOW_DIRTY=1` pushes anyway).
  Use the fixed tag on RunPod and in a cluster, so you know what runs and a restart does not pick
  up another build.
- The version and commit are also in the image's labels (`org.opencontainers.image.version`,
  `org.opencontainers.image.revision`; `docker inspect`), in the engine's log at start
  (`tinylm-studio engine 0.1.0, commit 5a0d901`), in `GET /health` and in the app's header
  (**Engine ready · v0.1.0 (5a0d901) · …**). The engine version is the one in
  `engine/pyproject.toml`; the build fails if `ENGINE_VERSION` says otherwise.

- llama.cpp is compiled in the image without `-march=native`, so it runs on any x86-64 CPU. The CUDA
  build compiles it for six GPU generations and takes much longer than the CPU one.
- `TORCH_INDEX` (a build argument) picks where PyTorch comes from; the default is PyTorch's own
  CPU or CUDA 12.6 index.
- In the image: `TINYLM_HOST=0.0.0.0`, port 8765, data in `/workspace/tinylm-studio` (the
  Hugging Face cache too, `HF_HOME`), and a `HEALTHCHECK` on `/health`. Mount a volume on
  `/workspace`.
- The image refuses to start without `TINYLM_API_KEY`, since the port is meant to be reached
  from outside; set `TINYLM_ALLOW_NO_API_KEY=1` to run it without one on a private network.
- Pass `HF_TOKEN` for gated models. The embedding model and models are downloaded on first use,
  into the volume.

## RunPod

[RunPod](https://www.runpod.io/) rents GPU and CPU pods by the second. The engine runs in a GPU pod
from the `cuda` image or a CPU pod from the `cpu` image; the desktop app talks to it through RunPod's HTTPS proxy
(`https://<pod id>-8765.proxy.runpod.net`) with the engine's API key, as to any
[remote engine](user-guide.md#engine).

1. Build and push the image to a registry RunPod can pull from (Docker Hub, GHCR):
   `make image-push VARIANT=cuda IMAGE=docker.io/<you>/tinylm-engine`; it prints the tag to use,
   e.g. `docker.io/<you>/tinylm-engine:0.1.0-5a0d901-cuda`. A private image needs
   registry credentials added on runpod.io.
2. Create an API key on runpod.io (console > [Credentials](https://console.runpod.io/user/credentials)
   > API Keys > **Create API Key**) with **All** permissions: the app creates, starts and stops pods.
3. In the app, Settings > Engine > **RunPod**: enter the key and the image
   (`docker.io/<you>/tinylm-engine:0.1.0-5a0d901-cuda`). The app loads RunPod's catalog
   (`GET /v2/catalog/gpus`, `/v2/catalog/cpus`, `/v2/catalog/datacenters`, `/v2/network-volumes`):
   choose a GPU or CPU by price and what is free now, a data center and a network volume, and click
   **Create pod**. The app creates the pod through RunPod's REST API (`POST
   https://api.runpod.io/v2/pods`) with port `8765/http`, the volume on `/workspace`, a newly
   generated `TINYLM_API_KEY` and the `HF_TOKEN` from `.env`, waits for it to run and connects.
4. **Stop** stops the pod (billing stops, except for the volume); **Start**, or the next app start,
   starts it again; **Terminate** deletes it with its pod volume. With **Stop the pod when the app
   closes** on (the default), quitting the app stops a running pod, also after switching to another
   engine.

Notes:

- **Data.** Without a network volume a GPU pod keeps the data on its pod volume (50 GB by default):
  kept while the pod is stopped, deleted with it. A CPU pod has no pod volume: without a network
  volume its data is on the container disk and is lost when it stops. A
  [network volume](https://docs.runpod.io/storage/network-volumes) outlives the pod; it ties the
  pod to that volume's data center, so the app sets the data center from it.
- **GPU pods need CUDA 12.6 or newer** on the host, which the `cuda` image is built for: the app
  lists only GPUs offered with it and asks for it when creating the pod (`minCudaVersion`).
- **CPU pods** run in RunPod's own data centers (Secure cloud) with the `cpu` image. They answer with
  small GGUF models; fine-tuning on them is slow.
- **A stopped pod can come back without a GPU.** When the machine's GPUs are taken, starting the
  pod fails or waits; terminate it and create a new one (on a network volume nothing is lost).
- **A pod made on runpod.io** works too: give it the image, expose `8765` as an HTTP port, set
  `TINYLM_API_KEY`, and enter its id and that key under **Use a pod made on runpod.io**.
- **One engine per volume** holds here as well (rule 1 above).
- RunPod's proxy (Cloudflare) closes a connection after 100 seconds (HTTP 524). Training, export,
  crawls and generation run in the background and are only polled, so they are not affected; a
  single chat answer that takes longer is cut off.
- RunPod Serverless is not used: it scales workers to zero between requests and keeps no state,
  while the engine holds models in memory, a database and background jobs.

## Not done yet

- Building the embedding model and the GGUF converter into the image, to start without internet
  access.
- Kubernetes manifests or a Helm chart: Deployment, Service, PersistentVolumeClaim, Secret,
  ConfigMap, probes.
- Reading the container's CPU and memory limits (cgroups) instead of the node's, so
  `TINYLM_THREADS` does not have to be set by hand.
