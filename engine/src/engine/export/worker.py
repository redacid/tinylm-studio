"""Exports a fine-tuned model (LoRA adapter) as one GGUF file that llama.cpp runs.

Run as `python -m engine.export.worker JOB_DIR`; JOB_DIR/job.json holds the settings and the
worker appends progress to JOB_DIR/events.jsonl. Steps:
1. tools: get llama.cpp's converter (see engine.export.tools);
2. merge: load the base model, merge the adapter into its weights and save the result;
3. convert: write a 16-bit GGUF file with llama.cpp's converter;
4. quantize: shrink it (Q8_0, Q4_K_M) with llama.cpp's quantizer;
5. save: a new model folder with the .gguf file and tinylm.json (and collection.jsonl when
   the export takes a collection along, see engine.models.bundles).
Temporary files stay in JOB_DIR/tmp and are deleted at the end.
"""

import ctypes
import importlib.util
import json
import logging
import os
import shutil
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

from engine.export.tools import converter_dir
from engine.training.worker import EventLog

log = logging.getLogger(__name__)

QUANTIZATIONS = ("Q8_0", "Q4_K_M", "F16")


def merge(base: Path, adapter: Path, output: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, dtype="auto", device_map="cpu")
    if model.dtype == torch.float32:
        model = model.to(torch.float16)  # the GGUF file is 16-bit anyway; halves the memory
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(output, safe_serialization=True)
    # The adapter folder has the tokenizer and chat template the model was trained with.
    AutoTokenizer.from_pretrained(adapter).save_pretrained(output)
    copy_tokenizer_model(base, output)


def copy_tokenizer_model(base: Path, output: Path) -> None:
    """A fast tokenizer saves only tokenizer.json, but llama.cpp converts some models (Gemma 3)
    from the SentencePiece tokenizer.model: without it, it reads tokenizer.json, finds Gemma 3's
    <image_soft_token> past the end of the vocabulary and fails. Take the base model's file."""
    source = base / "tokenizer.model"
    if source.is_file() and not (output / "tokenizer.model").exists():
        shutil.copy2(source, output / "tokenizer.model")


def convert(llama_cpp: Path, model: Path, output: Path, warn) -> None:
    """Runs convert_hf_to_gguf.py in this process, with its own copy of the gguf package."""
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    sys.path.insert(0, str(llama_cpp))
    for name in [m for m in sys.modules if m == "gguf" or m.startswith("gguf.")]:
        del sys.modules[name]  # an installed gguf package must not shadow the matching one
    spec = importlib.util.spec_from_file_location(
        "convert_hf_to_gguf", llama_cpp / "convert_hf_to_gguf.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    allow_unknown_tokenizer(warn)
    skip_missing_sentencepiece()
    argv = sys.argv
    sys.argv = ["convert_hf_to_gguf.py", str(model), "--outfile", str(output), "--outtype", "f16"]
    try:
        module.main()
    finally:
        sys.argv = argv


def allow_unknown_tokenizer(warn) -> None:
    """The converter refuses tokenizers whose pre-tokenizer it does not know. The supported base
    models are all known; for others, fall back to the default one and say so."""
    try:
        import conversion.base as base
    except ImportError:  # older llama.cpp layouts keep everything in convert_hf_to_gguf.py
        return
    known = base.TextModel.get_vocab_base_pre

    def get_vocab_base_pre(self, tokenizer) -> str:
        try:
            return known(self, tokenizer)
        except NotImplementedError:
            warn(
                "llama.cpp does not know this tokenizer's pre-tokenizer; the default one is "
                "used, so the GGUF model may split some words differently."
            )
            return "default"

    base.TextModel.get_vocab_base_pre = get_vocab_base_pre


def skip_missing_sentencepiece() -> None:
    """Some converters (Qwen's among them) try a SentencePiece tokenizer first and fall back to
    tokenizer.json when there is no tokenizer.model, but they import sentencepiece before looking.
    Look first, so models without tokenizer.model convert even when sentencepiece is missing."""
    try:
        import conversion.base as base
    except ImportError:
        return
    create = base.TextModel._create_vocab_sentencepiece

    def _create_vocab_sentencepiece(self):
        path = self.dir_model / "tokenizer.model"
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        if importlib.util.find_spec("sentencepiece") is None:
            raise ModuleNotFoundError(
                "This model's tokenizer needs the sentencepiece package.", name="sentencepiece"
            )
        return create(self)

    base.TextModel._create_vocab_sentencepiece = _create_vocab_sentencepiece


class StepLog(EventLog):
    """Remembers the step being run, so an error can say which one failed."""

    step: str | None = None

    def write(self, kind: str, **fields) -> None:
        if kind == "step":
            self.step = fields["step"]
        super().write(kind, **fields)


STEP_NAMES = {
    "tools": "getting the converter",
    "merge": "merging the adapter into the base model",
    "convert": "converting to GGUF",
    "quantize": "quantizing",
    "save": "saving",
}


def error_message(error: Exception, step: str | None = None) -> str:
    """What failed, in which step and, for an error without a message (a failed assert in
    llama.cpp's converter), on which line of code."""
    text = str(error)
    if not text:
        frame = traceback.extract_tb(error.__traceback__)[-1:]
        if frame:
            where = f"{Path(frame[0].filename).name}, line {frame[0].lineno}"
            text = f"{frame[0].line} ({where})" if frame[0].line else where
    message = type(error).__name__
    if step:
        message += f" while {STEP_NAMES.get(step, step)}"
    if text:
        message += f": {text}"
    if isinstance(error, ModuleNotFoundError):
        message += " The engine is missing a package; run 'make setup-engine' to update it."
    return message


def quantize(source: Path, output: Path, quantization: str, threads: int) -> None:
    import llama_cpp

    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = getattr(llama_cpp, f"LLAMA_FTYPE_MOSTLY_{quantization}")
    params.nthread = threads
    code = llama_cpp.llama_model_quantize(
        str(source).encode(), str(output).encode(), ctypes.byref(params)
    )
    if code != 0:
        raise RuntimeError(f"llama.cpp could not quantize the model (code {code})")


def export(config: dict, events: EventLog, job_dir: Path) -> None:
    models = Path(config["models_dir"])
    adapter = models / config["model"]
    info = json.loads((adapter / "tinylm.json").read_text())
    base = models / info["base"]
    output = models / config["name"]
    if output.exists():
        raise FileExistsError(f"A model named {output.name} already exists")
    quantization = config["quantization"]
    tmp = job_dir / "tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    warn = lambda message: events.write("warning", message=message)  # noqa: E731

    events.write("step", step="tools")
    override = config.get("llama_cpp_dir")
    tools = converter_dir(
        Path(config["tools_dir"]),
        Path(override) if override else None,
        progress=lambda message: events.write("note", message=message),
    )
    events.write("step", step="merge")
    merge(base, adapter, tmp / "merged")
    events.write("step", step="convert")
    f16 = tmp / "model-f16.gguf"
    convert(tools, tmp / "merged", f16, warn)
    shutil.rmtree(tmp / "merged")

    events.write("step", step="quantize")
    filename = f"{config['name']}.{quantization}.gguf"
    partial = models / f".{config['name']}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir()
    if quantization == "F16":
        f16.rename(partial / filename)
    else:
        quantize(f16, partial / filename, quantization, config["threads"])
    events.write("step", step="save")
    exported = {
        **info,
        "format": "gguf",
        "quantization": quantization,
        "exported_from": config["model"],
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if (job_dir / "collection.jsonl").is_file():
        # The collection goes with the model; engine.models.bundles reads it.
        shutil.copy2(job_dir / "collection.jsonl", partial / "collection.jsonl")
        exported["collection"] = config["collection"]
    (partial / "tinylm.json").write_text(json.dumps(exported, indent=2))
    partial.rename(output)
    shutil.rmtree(tmp, ignore_errors=True)
    size = (output / filename).stat().st_size
    events.write("done", model=config["name"], size=size)


def main() -> None:
    from engine.__main__ import exit_with_parent

    job_dir = Path(sys.argv[1])
    exit_with_parent()
    logging.basicConfig(level=logging.INFO)
    events = StepLog(job_dir / "events.jsonl")
    config = json.loads((job_dir / "job.json").read_text())["settings"]
    config.setdefault("threads", os.cpu_count() or 4)
    try:
        export(config, events, job_dir)
    except Exception as e:
        log.exception("Export failed")
        events.write("error", message=error_message(e, events.step))
        shutil.rmtree(job_dir / "tmp", ignore_errors=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
