"""LoRA fine-tuning: masking the prompt, training runs (start, stop, resume) and serving the
result. Runs train a tiny random model on the CPU in a few seconds."""

import json
import shutil
import time

import pytest
from tiny_model import build_tiny_model

from engine.training.data import IGNORE, answer_spans, encode

CONVERSATION = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Use Igni?"},
    {"role": "assistant", "content": "Use Igni."},
    {"role": "user", "content": "Відьмак?"},
    {"role": "assistant", "content": "Знак Ігні."},
]


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    return build_tiny_model(tmp_path_factory.mktemp("tiny") / "tiny")


@pytest.fixture
def tokenizer(tiny_model):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tiny_model)


@pytest.fixture
def trainable(client, settings, tiny_model):
    """An installed base model and a dataset of 40 examples."""
    shutil.copytree(tiny_model, settings.models_dir / "tiny")
    client.post("/api/datasets", json={"name": "hints", "eval_percent": 20})
    lines = [
        json.dumps({"prompt": f"Hint {i} for Witcher?", "completion": f"Use Igni {i} times."})
        for i in range(40)
    ]
    files = {"files": ("h.jsonl", "\n".join(lines).encode())}
    client.post("/api/datasets/hints/import", files=files)
    return client


def wait(client, job_id: str, until=lambda job: job["state"] not in ("running", "stopping")):
    deadline = time.time() + 120
    while time.time() < deadline:
        job = client.get(f"/api/training/jobs/{job_id}").json()
        if until(job):
            return job
        time.sleep(0.2)
    raise TimeoutError(job)


def test_only_answers_are_learned(tokenizer):
    text = tokenizer.apply_chat_template(CONVERSATION, tokenize=False)
    spans = answer_spans(text, CONVERSATION, "<|im_end|>")
    # The question "Use Igni?" is not mistaken for the answer "Use Igni.".
    assert [text[s:e] for s, e in spans] == ["Use Igni.<|im_end|>", "Знак Ігні.<|im_end|>"]

    encoded = encode(tokenizer, CONVERSATION, max_length=512)
    learned = [
        t for t, label in zip(encoded.input_ids, encoded.labels, strict=True) if label != IGNORE
    ]
    assert tokenizer.decode(learned) == "Use Igni.<|im_end|>Знак Ігні.<|im_end|>"
    assert encoded.labels[0] == IGNORE

    short = encode(tokenizer, CONVERSATION, max_length=8)
    assert len(short.input_ids) == 8 and short.trained_tokens == 0


def test_train_and_chat_with_the_result(trainable, settings):
    client = trainable
    body = {
        "name": "tiny-hints",
        "base": "tiny",
        "dataset": "hints",
        "epochs": 2,
        "learning_rate": 1e-3,
        "batch_size": 4,
        "gradient_accumulation": 1,
    }
    started = client.post("/api/training/jobs", json=body)
    assert started.status_code == 201, started.text
    job = wait(client, started.json()["id"])
    assert job["state"] == "done", job
    assert job["step"] == job["total_steps"] == 16  # 32 training examples, batches of 4, 2 epochs
    assert (job["train_examples"], job["eval_examples"]) == (32, 8)
    assert job["device"] == "cpu" and job["eval_loss"] is not None
    assert [p["step"] for p in job["history"]] == list(range(1, 17))

    events = client.get(f"/api/training/jobs/{job['id']}/events").text
    kinds = [json.loads(line[6:])["type"] for line in events.split("\n\n") if line]
    assert kinds[0] == "start" and "done" in kinds and kinds[-1] == "state"

    # The fine-tuned model is not offered as a base for another run.
    assert client.get("/api/models").json()["trainable"] == ["tiny"]
    trained = client.get("/api/models/trained").json()["data"]
    assert trained[0]["id"] == "tiny-hints" and trained[0]["base"] == "tiny"
    assert trained[0]["dataset"] == "hints" and trained[0]["settings"]["epochs"] == 2

    assert "tiny-hints" in [m["id"] for m in client.get("/v1/models").json()["data"]]
    answer = client.post(
        "/v1/chat/completions",
        json={
            "model": "tiny-hints",
            "messages": [{"role": "user", "content": "Hint 3?"}],
            "max_tokens": 4,
        },
    )
    assert answer.status_code == 200 and answer.json()["usage"]["completion_tokens"] <= 4

    again = client.post("/api/training/jobs", json=body)
    assert again.status_code == 422 and "already exists" in again.json()["detail"]

    # Without its base model a fine-tuned model cannot answer, and is not offered.
    shutil.rmtree(settings.models_dir / "tiny")
    assert "tiny-hints" not in [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert client.get("/api/models/trained").json()["data"][0]["base_installed"] is False
    assert client.delete("/api/models/tiny-hints").status_code == 204
    assert client.get("/api/models/trained").json()["data"] == []
    assert client.delete(f"/api/training/jobs/{job['id']}").status_code == 204


def test_stop_resume_and_learn(trainable):
    client = trainable
    body = {
        "name": "tiny-long",
        "base": "tiny",
        "dataset": "hints",
        "epochs": 60,
        "batch_size": 4,
        "gradient_accumulation": 1,
        "learning_rate": 5e-3,
        "save_steps": 5,
    }
    job_id = client.post("/api/training/jobs", json=body).json()["id"]
    other = client.post("/api/training/jobs", json={**body, "name": "tiny-other"})
    assert other.status_code == 409

    wait(client, job_id, lambda job: (job["checkpoint_step"] or 0) >= 5)
    assert client.post(f"/api/training/jobs/{job_id}/stop").json()["state"] == "stopping"
    stopped = wait(client, job_id)
    assert stopped["state"] == "stopped" and stopped["resumable"], stopped
    assert stopped["step"] < stopped["total_steps"] == 480
    assert client.delete("/api/training/jobs/0123456789ab").status_code == 404

    resumed = client.post(f"/api/training/jobs/{job_id}/resume")
    assert resumed.status_code == 200 and resumed.json()["state"] == "running"
    done = wait(client, job_id)
    assert done["state"] == "done", done
    steps = [p["step"] for p in done["history"]]
    assert steps == list(range(1, 481))

    # The run went on from its checkpoint and the model learned the answers.
    history = done["history"]
    assert history[-1]["loss"] < history[0]["loss"] / 4
    question = [{"role": "user", "content": "Hint 7 for Witcher?"}]
    answer = client.post(
        "/v1/chat/completions",
        json={"model": "tiny-long", "messages": question, "max_tokens": 6, "temperature": 0},
    )
    assert answer.json()["choices"][0]["message"]["content"].startswith("Use Igni")


def test_training_checks(trainable, settings):
    client = trainable
    (settings.models_dir / "fast-gguf").mkdir()
    (settings.models_dir / "fast-gguf" / "model.gguf").write_bytes(b"GGUF")
    start = {"name": "x", "base": "tiny", "dataset": "hints"}
    cases = [
        ({"base": "fast-gguf"}, 422, "PyTorch"),
        ({"base": "missing"}, 422, "not installed"),
        ({"dataset": "missing"}, 404, "Unknown dataset"),
        ({"name": "../evil"}, 422, "Name the model"),
        ({"name": "tiny"}, 422, "already exists"),
    ]
    for change, status, message in cases:
        response = client.post("/api/training/jobs", json={**start, **change})
        assert response.status_code == status and message in response.json()["detail"], change
    assert client.get("/api/training/jobs").json()["data"] == []
    assert client.get("/api/training/jobs/nope").status_code == 404
