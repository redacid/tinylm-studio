"""Base models offered for download. Small, instruction-tuned, permissive or open licenses.

GGUF models run through llama.cpp and are the fast option on CPU; Hugging Face (safetensors)
models run through PyTorch and are the ones used for fine-tuning.
"""

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelSpec:
    id: str
    repo: str
    params: str
    description: str
    format: Literal["hf", "gguf"] = "hf"
    # For GGUF repos: glob of the single quantized file to download.
    file_pattern: str | None = None
    gated: bool = False  # requires accepting the license on Hugging Face and an HF token

    def to_dict(self) -> dict:
        return asdict(self)


CATALOG: list[ModelSpec] = [
    ModelSpec(
        "qwen3-1.7b-gguf",
        "Qwen/Qwen3-1.7B-GGUF",
        "1.7B",
        "Q8_0 via llama.cpp, fast; weak Ukrainian (often answers in Russian).",
        format="gguf",
        file_pattern="*Q8_0.gguf",
    ),
    ModelSpec(
        "qwen3-0.6b-gguf",
        "Qwen/Qwen3-0.6B-GGUF",
        "0.6B",
        "Q8_0 via llama.cpp, fastest.",
        format="gguf",
        file_pattern="*Q8_0.gguf",
    ),
    ModelSpec(
        "qwen3-4b-gguf",
        "Qwen/Qwen3-4B-GGUF",
        "4B",
        "Q4_K_M via llama.cpp, best quality that still runs on CPU.",
        format="gguf",
        file_pattern="*Q4_K_M.gguf",
    ),
    ModelSpec(
        "gemma-3-4b-gguf",
        "unsloth/gemma-3-4b-it-GGUF",
        "4B",
        "Default. Q4_K_M via llama.cpp, best uk/ru/en quality in our benchmarks.",
        format="gguf",
        file_pattern="*Q4_K_M.gguf",
    ),
    ModelSpec("qwen3-0.6b", "Qwen/Qwen3-0.6B", "0.6B", "PyTorch weights, base for fine-tuning."),
    ModelSpec("qwen3-1.7b", "Qwen/Qwen3-1.7B", "1.7B", "PyTorch weights, base for fine-tuning."),
    ModelSpec(
        "qwen2.5-0.5b-instruct", "Qwen/Qwen2.5-0.5B-Instruct", "0.5B", "Compact, multilingual."
    ),
    ModelSpec(
        "smollm2-360m-instruct",
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        "0.36B",
        "Very small, mostly English.",
    ),
    ModelSpec("gemma-3-1b-it", "google/gemma-3-1b-it", "1B", "Multilingual.", gated=True),
    ModelSpec(
        "llama-3.2-1b-instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "1B",
        "Multilingual.",
        gated=True,
    ),
]

DEFAULT_MODEL = "gemma-3-4b-gguf"


def find(model_id: str) -> ModelSpec | None:
    return next((m for m in CATALOG if m.id == model_id), None)
