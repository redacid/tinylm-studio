"""A LoRA fine-tuning run, started by the engine as its own process:

    python -m engine.training.worker JOB_DIR

Training needs all CPU cores and a lot of memory for minutes or hours; in a separate process it
does not slow the API down, can be stopped by killing it, and its memory is freed when it ends.

The job folder holds config.json (written by the engine), train.jsonl and eval.jsonl. The worker
appends progress to events.jsonl, saves checkpoints to checkpoints/ (so a stopped run can
resume), stops at the next step when a file named "stop" appears, and finally saves the adapter
as a new model folder.
"""

import dataclasses
import json
import math
import shutil
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from engine.training.data import Collator, encode_all


class EventLog:
    def __init__(self, path: Path):
        self.path = path

    def write(self, kind: str, **fields) -> None:
        event = {"type": kind, "time": time.time(), **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[list[dict]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def last_checkpoint(folder: Path) -> Path | None:
    checkpoints = [p for p in folder.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()]
    return max(checkpoints, key=lambda p: int(p.name.split("-")[-1]), default=None)


def train(job_dir: Path, events: EventLog) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    from engine.inference.transformers_runtime import physical_cores

    config = json.loads((job_dir / "config.json").read_text())
    stop_file = job_dir / "stop"
    output = Path(config["output_path"])
    device = pick_device(config["device"])
    if device == "cpu":
        # Unlike generating text, training is compute-bound and uses every core.
        torch.set_num_threads(config.get("threads") or physical_cores())

    tokenizer = AutoTokenizer.from_pretrained(config["base_path"])
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    max_length = config["max_length"]
    train_set, dropped = encode_all(tokenizer, read_jsonl(job_dir / "train.jsonl"), max_length)
    eval_set, _ = encode_all(tokenizer, read_jsonl(job_dir / "eval.jsonl"), max_length)
    if not train_set:
        raise ValueError("No training examples left: every answer is beyond max_length tokens.")

    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    fp16 = device == "cuda" and not bf16
    dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(config["base_path"], dtype=dtype)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    trainable, total = model.get_nb_trainable_parameters()

    batch = config["batch_size"] * config["gradient_accumulation"]
    total_steps = math.ceil(len(train_set) / batch) * config["epochs"]
    # Checkpoints and evaluations about ten times per run, so a stop loses little work.
    every = config.get("save_steps") or max(5, min(100, total_steps // 10))
    # Batches of similar lengths waste less work on padding. The option was renamed in
    # transformers 5.
    known = {f.name for f in dataclasses.fields(TrainingArguments)}
    if "train_sampling_strategy" in known:
        grouping = {"train_sampling_strategy": "group_by_length"}
    else:
        grouping = {"group_by_length": True}
    args = TrainingArguments(
        output_dir=str(job_dir / "checkpoints"),
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation"],
        lr_scheduler_type="cosine",
        warmup_steps=max(1, total_steps // 30),
        logging_steps=1,
        eval_strategy="steps" if eval_set else "no",
        eval_steps=every,
        save_strategy="steps",
        save_steps=every,
        save_total_limit=2,
        use_cpu=device == "cpu",
        bf16=bf16,
        fp16=fp16,
        seed=config["seed"],
        **grouping,
        dataloader_num_workers=0,
        report_to="none",
        disable_tqdm=True,
        remove_unused_columns=False,
    )

    class Progress(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            events.write(
                "start",
                step=state.global_step,
                total_steps=state.max_steps,
                device=device,
                train_examples=len(train_set),
                eval_examples=len(eval_set),
                dropped=dropped,
                trainable_params=trainable,
                total_params=total,
            )

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" in logs:
                events.write(
                    "step",
                    step=state.global_step,
                    epoch=round(state.epoch or 0, 3),
                    loss=logs["loss"],
                    learning_rate=logs.get("learning_rate"),
                )
            if "eval_loss" in logs:
                events.write("eval", step=state.global_step, eval_loss=logs["eval_loss"])

        def on_save(self, args, state, control, **kwargs):
            events.write("checkpoint", step=state.global_step)

        def on_step_end(self, args, state, control, **kwargs):
            if stop_file.exists():
                control.should_save = True
                control.should_training_stop = True

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=[vars(e) for e in train_set],
        eval_dataset=[vars(e) for e in eval_set] or None,
        data_collator=Collator(pad_id),
        callbacks=[Progress()],
    )
    resume = last_checkpoint(job_dir / "checkpoints") if config.get("resume") else None
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    if stop_file.exists():
        events.write("stopped", step=trainer.state.global_step)
        return

    eval_loss = None
    if eval_set:
        eval_loss = trainer.evaluate()["eval_loss"]
        events.write("eval", step=trainer.state.global_step, eval_loss=eval_loss)
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    save_model(model, tokenizer, output, config, losses[-1] if losses else None, eval_loss)
    events.write("done", step=trainer.state.global_step, model=output.name)


def save_model(model, tokenizer, output: Path, config: dict, loss, eval_loss) -> None:
    """Saves the adapter with the base model's tokenizer as a model folder the engine serves."""
    if output.exists():
        raise FileExistsError(f"A model named {output.name} already exists")
    partial = output.parent / f".{output.name}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    model.save_pretrained(partial)
    tokenizer.save_pretrained(partial)
    info = {
        "base": config["base"],
        "dataset": config["dataset"],
        "job": config["job"],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "loss": loss,
        "eval_loss": eval_loss,
        "settings": {k: config[k] for k in SETTINGS},
    }
    (partial / "tinylm.json").write_text(json.dumps(info, indent=2))
    partial.rename(output)


SETTINGS = (
    "epochs",
    "learning_rate",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "max_length",
    "batch_size",
    "gradient_accumulation",
)


def main() -> None:
    from engine.__main__ import exit_with_parent

    job_dir = Path(sys.argv[1])
    # A crashed or stopped engine takes its training run with it; the run can be resumed.
    exit_with_parent()
    events = EventLog(job_dir / "events.jsonl")
    try:
        train(job_dir, events)
    except Exception as e:
        traceback.print_exc()
        events.write("error", message=f"{type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
