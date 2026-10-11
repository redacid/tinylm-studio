"""Benchmark installed models on CPU/GPU with prompts from the reference scenario.

Measures load time, time to first token and generation speed, and saves the answers so
quality in Ukrainian, Russian and English can be compared by reading the report.
"""

import platform
import time
from datetime import datetime
from pathlib import Path

import psutil
import torch

from engine import hardware
from engine.chat import ChatRequest, Finish, Message, TextDelta
from engine.inference.base import Runtime
from engine.inference.transformers_runtime import physical_cores
from engine.models.manager import ModelManager, model_format

PROMPTS = {
    "uk": "Коротко опиши гру The Witcher 3: Wild Hunt і для кого вона.",
    "ru": "Дай короткую подсказку, как победить первого босса в Dark Souls, без спойлеров.",
    "en": "Write a three-sentence review of Hollow Knight.",
}


def run_one(runtime: Runtime, prompt: str, max_tokens: int) -> dict:
    # Sampling settings recommended for small chat models; greedy decoding tends to loop.
    request = ChatRequest(
        model="bench",
        messages=[Message("user", prompt)],
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.8,
    )
    started = time.perf_counter()
    first: float | None = None
    parts: list[str] = []
    output_tokens = 0
    for event in runtime.stream(request):
        if isinstance(event, TextDelta):
            first = first or time.perf_counter()
            parts.append(event.text)
        elif isinstance(event, Finish):
            output_tokens = event.usage.output_tokens
    total = time.perf_counter() - started
    decode = total - ((first or started) - started)
    return {
        "ttft": (first or started) - started,
        "tokens": output_tokens,
        "tok_s": output_tokens / decode if decode > 0 else 0.0,
        "text": "".join(parts).strip(),
    }


def benchmark(
    models_dir: Path,
    model_ids: list[str],
    max_tokens: int,
    out_dir: Path,
    runtime_options: dict | None = None,
) -> Path:
    hw = hardware.detect()
    torch.manual_seed(0)
    manager = ModelManager(models_dir, runtime_options=runtime_options)
    threads = (runtime_options or {}).get("threads") or physical_cores()
    lines = [
        f"# Model benchmark {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Machine: {platform.platform()}, {hw.cpu_cores} cores, {hw.memory_gb} GB RAM, "
        f"accelerators: {', '.join(hw.accelerators) or 'none'}. Max tokens: {max_tokens}.",
        f"PyTorch {torch.__version__}, threads: {threads}, options: {runtime_options or {}}.",
        "Runtime: llama.cpp for GGUF models, PyTorch for Hugging Face models.",
        "",
        "| Model | Load, s | RSS, GB | "
        + " | ".join(f"{k} tok/s (TTFT s)" for k in PROMPTS)
        + " |",
        "|---|---|---|" + "---|" * len(PROMPTS),
    ]
    answers: list[str] = []
    for model_id in model_ids:
        print(f"Benchmarking {model_id}…", flush=True)
        started = time.perf_counter()
        runtime = manager.runtime(model_id)
        load = time.perf_counter() - started
        results = {lang: run_one(runtime, prompt, max_tokens) for lang, prompt in PROMPTS.items()}
        rss = psutil.Process().memory_info().rss / 2**30
        cells = " | ".join(f"{r['tok_s']:.1f} ({r['ttft']:.2f})" for r in results.values())
        name = f"{model_id} ({model_format(models_dir / model_id)})"
        lines.append(f"| {name} | {load:.1f} | {rss:.1f} | {cells} |")
        answers += [f"## {model_id}", ""]
        for lang, result in results.items():
            answers += [f"**{lang}:** {PROMPTS[lang]}", "", f"> {result['text']}", ""]
        del runtime

    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"benchmark-{datetime.now():%Y%m%d-%H%M%S}.md"
    text = "\n".join(lines + [""] + answers)
    report.write_text(text, encoding="utf-8")
    return report
