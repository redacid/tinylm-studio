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
