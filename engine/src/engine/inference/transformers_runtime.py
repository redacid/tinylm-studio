"""Hugging Face transformers runtime. Runs on CPU by default, uses CUDA/MPS when available."""

import logging
import threading
from collections.abc import Iterator
from pathlib import Path

from engine.chat import ChatRequest, Finish, StopSequenceFilter, StreamEvent, TextDelta, Usage

log = logging.getLogger(__name__)


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def physical_cores() -> int:
    import psutil

    return psutil.cpu_count(logical=False) or psutil.cpu_count() or 1


def default_threads() -> int:
    # Token generation is limited by memory bandwidth, so more than ~8 threads does not help.
    # On hybrid CPUs (P + E cores) extra threads land on slow E-cores and every step waits
    # for them: on an i9-12900H, 8 threads beat 12 and 14 by 40-60%.
    return min(physical_cores(), 8)


class TransformersRuntime:
    def __init__(
        self,
        model_path: Path,
        device: str | None = None,
        threads: int | None = None,
        repetition_penalty: float = 1.1,
        adapter: Path | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or pick_device()
        self.repetition_penalty = repetition_penalty
        if self.device == "cpu":
            torch.set_num_threads(threads or default_threads())
        self.threads = torch.get_num_threads()
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        log.info("Loading %s on %s (%d threads)", model_path, self.device, self.threads)
        self.tokenizer = AutoTokenizer.from_pretrained(adapter or model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
        if adapter:
            from peft import PeftModel

            # Merged into the weights once, so answers are as fast as with the base model.
            log.info("Applying fine-tuned adapter %s", adapter)
            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload()
        self.model.to(self.device).eval()
        # One generation at a time: a small machine has no headroom for parallel decoding.
        self._lock = threading.Lock()

    def _encode(self, request: ChatRequest):
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        # enable_thinking=False disables the Qwen3 reasoning mode; other templates ignore it.
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        return self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        from transformers import TextIteratorStreamer

        with self._lock:
            inputs = self._encode(request)
            input_tokens = int(inputs["input_ids"].shape[1])
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            sampling = request.temperature > 0
            kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=request.max_tokens,
                do_sample=sampling,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            if sampling:
                kwargs.update(temperature=request.temperature, top_p=request.top_p)

            result: dict = {}

            def run() -> None:
                result["ids"] = self.model.generate(**kwargs)

            thread = threading.Thread(target=run, daemon=True)
            thread.start()

            stop = StopSequenceFilter(request.stop)
            for piece in streamer:
                if text := stop.feed(piece):
                    yield TextDelta(text)
                if stop.matched:
                    break
            thread.join()
            if tail := stop.flush():
                yield TextDelta(tail)

            output_tokens = int(result["ids"].shape[1]) - input_tokens if "ids" in result else 0
            usage = Usage(input_tokens, output_tokens)
            if stop.matched:
                yield Finish("stop_sequence", usage, stop.matched)
            elif output_tokens >= request.max_tokens:
                yield Finish("length", usage)
            else:
                yield Finish("stop", usage)
