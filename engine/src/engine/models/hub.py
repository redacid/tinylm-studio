"""Searching Hugging Face for open models and choosing what to download from a repo.

GGUF repos hold one file per quantization (some split into parts); a model is one of them.
Hugging Face-format repos are downloaded whole: config, tokenizer and safetensors weights.
"""

import re
from fnmatch import fnmatch
from typing import Literal

from huggingface_hub import HfApi

Kind = Literal["gguf", "hf"]

# What a Hugging Face-format download takes from a repo.
HF_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"]
# Pipelines of models that write text; image-text-to-text covers Gemma 3 and other models
# that also see images.
TEXT_PIPELINES = {"text-generation", "image-text-to-text", None}
PART = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$")
QUANT = re.compile(r"(?i)(?:^|[-_.])((?:I?Q\d+(?:_[0-9A-Z]+)*)|BF16|F16|F32)(?=[-_.]|$)")


class HubError(Exception):
    pass


def search(query: str, kind: Kind, token: str | None, limit: int = 30) -> list[dict]:
    """The most downloaded matching models, GGUF or Hugging Face format."""
    try:
        found = HfApi(token=token).list_models(
            search=query.strip() or None,
            filter="gguf" if kind == "gguf" else "safetensors",
            sort="downloads",
            # Other models (embeddings, classifiers) are dropped below, so ask for more.
            limit=limit if kind == "gguf" else limit * 3,
            expand=["downloads", "likes", "gated", "lastModified", "pipeline_tag", "gguf"]
            + ([] if kind == "gguf" else ["safetensors"]),
        )
        models = []
        for m in found:
            if kind == "hf" and m.pipeline_tag not in TEXT_PIPELINES:
                continue
            gguf = getattr(m, "gguf", None) or {}
            safetensors = getattr(m, "safetensors", None)
            models.append(
                {
                    "repo": m.id,
                    "downloads": m.downloads or 0,
                    "likes": m.likes or 0,
                    "gated": bool(m.gated),
                    "updated": m.last_modified.isoformat() if m.last_modified else None,
                    "pipeline": m.pipeline_tag,
                    "params": gguf.get("total") or (safetensors.total if safetensors else None),
                    "context_length": gguf.get("context_length"),
                }
            )
            if len(models) >= limit:
                break
        return models
    except Exception as error:
        raise HubError(f"Hugging Face search failed: {error}") from error


def files(repo: str, token: str | None) -> dict:
    """What can be downloaded from a repo: its GGUF files (a split file is one entry) and
    the size of the Hugging Face-format weights, or None when it has none."""
    try:
        info = HfApi(token=token).model_info(repo, files_metadata=True)
    except Exception as error:
        raise HubError(f"Could not read {repo} on Hugging Face: {error}") from error
    sizes = {s.rfilename: s.size or 0 for s in info.siblings or []}
    gguf: dict[str, dict] = {}
    for name, size in sorted(sizes.items()):
        base = name.rsplit("/", 1)[-1]
        # Vision projectors go with a model file, they are not a model of their own.
        if not name.endswith(".gguf") or base.lower().startswith("mmproj"):
            continue
        part = PART.match(name)
        key = part["stem"] if part else name
        entry = gguf.setdefault(
            key,
            {"file": name, "size": 0, "parts": 1, "quantization": quantization(base)},
        )
        entry["size"] += size
        if part:
            entry["parts"] = int(part["count"])
            entry["file"] = min(entry["file"], name)
    names = set(sizes)
    hf = None
    if "config.json" in names and any(n.endswith(".safetensors") for n in names):
        hf = {"size": sum(size for n, size in sizes.items() if wanted(n, HF_PATTERNS))}
    return {
        "repo": repo,
        "gated": bool(info.gated),
        "gguf": sorted(gguf.values(), key=lambda f: f["size"]),
        "hf": hf,
    }


def quantization(name: str) -> str | None:
    found = QUANT.findall(name.removesuffix(".gguf"))
    return found[-1].upper() if found else None


def gguf_patterns(file: str) -> list[str]:
    """The file to download, or all parts of a split file."""
    part = PART.match(file)
    return [f"{part['stem']}-*-of-{part['count']}.gguf"] if part else [file]


def wanted(name: str, patterns: list[str]) -> bool:
    return any(fnmatch(name, p) for p in patterns)


def model_id(repo: str, kind: Kind, file: str | None = None) -> str:
    """A folder name for a downloaded model: the GGUF file name, or the repo name."""
    if kind == "gguf" and file:
        name = file.rsplit("/", 1)[-1].removesuffix(".gguf")
        if part := PART.match(file.rsplit("/", 1)[-1]):
            name = part["stem"]
    else:
        name = repo.rsplit("/", 1)[-1]
    name = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-.")
    return name or "model"
