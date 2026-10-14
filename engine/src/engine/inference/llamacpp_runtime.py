"""llama.cpp runtime for quantized GGUF models: the fast path on CPU.

The prompt is rendered from the chat template stored in the GGUF file so that template
options such as enable_thinking=False (Qwen3) can be passed, which llama.cpp's built-in
chat formatting does not allow.
"""

import logging
import threading
from collections.abc import Iterator
from pathlib import Path

from engine.chat import ChatRequest, Finish, StopSequenceFilter, StreamEvent, TextDelta, Usage
from engine.inference.transformers_runtime import default_threads

log = logging.getLogger(__name__)

DEFAULT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


class LlamaCppRuntime:
    def __init__(
        self,
        model_path: Path,
        threads: int | None = None,
        repetition_penalty: float = 1.1,
        context_size: int = 4096,
        gpu_layers: int = -1,
        batch_size: int = 512,
    ):
        from llama_cpp import Llama

        gguf = sorted(model_path.glob("*.gguf"))[0]
        self.threads = threads or default_threads()
        self.repetition_penalty = repetition_penalty
        log.info("Loading %s with llama.cpp (%d threads)", gguf.name, self.threads)
        # gpu_layers=-1 offloads everything when llama.cpp was built with GPU support;
        # a CPU-only build ignores it.
        self.llm = Llama(
            model_path=str(gguf),
            n_ctx=context_size,
            n_threads=self.threads,
            n_gpu_layers=gpu_layers,
            n_batch=batch_size,
            verbose=False,
        )
        self.template = self._compile_template()
        self._lock = threading.Lock()

    def _compile_template(self):
        from jinja2.ext import loopcontrols
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def raise_exception(message: str):
            raise ValueError(message)

        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols]
        )
        env.globals["raise_exception"] = raise_exception
        source = self.llm.metadata.get("tokenizer.chat_template") or DEFAULT_TEMPLATE
        return env.from_string(source)

    def _token_text(self, key: str) -> str:
        token_id = self.llm.metadata.get(key)
        if token_id is None:
            return ""
        return self.llm.detokenize([int(token_id)], special=True).decode(errors="ignore")

    def render(self, request: ChatRequest) -> str:
        return self.template.render(
            messages=[{"role": m.role, "content": m.content} for m in request.messages],
            add_generation_prompt=True,
            enable_thinking=bool(request.thinking),
            bos_token=self._token_text("tokenizer.ggml.bos_token_id"),
            eos_token=self._token_text("tokenizer.ggml.eos_token_id"),
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        with self._lock:
            # The template already contains BOS where the model needs it (e.g. Gemma), so the
            # prompt is tokenized here without adding a second one.
            tokens = self.llm.tokenize(self.render(request).encode(), add_bos=False, special=True)
            input_tokens = len(tokens)
            chunks = self.llm.create_completion(
                tokens,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                repeat_penalty=request.repetition_penalty or self.repetition_penalty,
                stream=True,
                # Unset: llama.cpp's own defaults.
                **{k: v for k in ("top_k", "min_p") if (v := getattr(request, k)) is not None},
            )
            stop = StopSequenceFilter(request.stop)
            output_tokens = 0
            finish_reason = "stop"
            for chunk in chunks:
                choice = chunk["choices"][0]
                output_tokens += bool(choice["text"])
                if text := stop.feed(choice["text"]):
                    yield TextDelta(text)
                if stop.matched:
                    break
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
            if tail := stop.flush():
                yield TextDelta(tail)

            usage = Usage(input_tokens, output_tokens)
            if stop.matched:
                yield Finish("stop_sequence", usage, stop.matched)
            elif finish_reason == "length":
                yield Finish("length", usage)
            else:
                yield Finish("stop", usage)
