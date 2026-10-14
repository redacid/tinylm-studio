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

## Kubernetes

The engine runs in a container in a Kubernetes cluster; see
[docs/deployment.md](docs/deployment.md).

- [ ] Container image (Dockerfile): engine with CPU-only dependencies; optional build arguments to
      include the embedding model and the GGUF converter.
- [ ] Manifests or a Helm chart: Deployment (1 replica, `Recreate`), Service, PVC
      (`ReadWriteOnce`) for `TINYLM_DATA_DIR`, Secret, ConfigMap, `/health` probes.
- [ ] Read CPU and memory limits from cgroups for the default `TINYLM_THREADS` and `/api/system`.
- [ ] Desktop app: connect to a remote engine (URL and API key in Settings) instead of starting
      its own.
