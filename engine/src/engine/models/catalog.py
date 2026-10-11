"""Base models offered for download. Small, instruction-tuned, permissive or open licenses."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    repo: str
    params: str
    description: str
    gated: bool = False  # requires accepting the license on Hugging Face and an HF token

    def to_dict(self) -> dict:
        return asdict(self)


CATALOG: list[ModelSpec] = [
    ModelSpec(
        "qwen3-0.6b", "Qwen/Qwen3-0.6B", "0.6B", "Default. Multilingual (uk/ru/en), fast on CPU."
    ),
    ModelSpec("qwen3-1.7b", "Qwen/Qwen3-1.7B", "1.7B", "Better quality, slower on CPU."),
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

DEFAULT_MODEL = "qwen3-0.6b"


def find(model_id: str) -> ModelSpec | None:
    return next((m for m in CATALOG if m.id == model_id), None)
