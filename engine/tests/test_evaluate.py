"""Evaluation: models answer the held-out examples; answers are scored by similarity and judge."""

import json
import re
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from tiny_model import build_tiny_model

from engine.evaluation.evaluate import Evaluation, EvaluationOptions, chrf, judge_prompt


class FakeJudge(BaseHTTPRequestHandler):
    """Gives answers that echo the question 1 point, other answers 4."""

    prompts: list[str] = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][-1]["content"]
        FakeJudge.prompts.append(prompt)
        answers = re.findall(r"^--- (\w) ---\n(.*)$", prompt, re.MULTILINE)
        scores = {
            label: {"score": 1 if text.startswith("Echo:") else 4, "reason": "checked"}
            for label, text in answers
        }
        text = json.dumps({"scores": scores})
        data = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def judge_url():
    FakeJudge.prompts = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeJudge)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    return build_tiny_model(tmp_path_factory.mktemp("tiny") / "tiny")


@pytest.fixture
def hints(client):
    """A dataset with half of its 12 examples held out."""
    client.post("/api/datasets", json={"name": "hints", "eval_percent": 50})
    lines = [
        json.dumps({"prompt": f"Hint {i} for Witcher?", "completion": f"Use Igni {i} times."})
        for i in range(12)
    ]
    client.post("/api/datasets/hints/import", files={"files": ("h.jsonl", "\n".join(lines))})
    return client


def wait(client, evaluation_id: str) -> dict:
    deadline = time.time() + 60
    while time.time() < deadline:
        report = client.get(f"/api/evaluations/{evaluation_id}").json()
        if report["state"] != "running":
            return report
        time.sleep(0.05)
    raise TimeoutError(report)


def test_chrf_and_judge_prompt():
    assert chrf("Use Igni 3 times.", "Use Igni 3 times.") == 100
    assert chrf("", "Use Igni") == 0
    close, far = chrf("Use Igni 4 times", "Use Igni 3 times."), chrf("Run away", "Use Igni.")
    assert 50 < close < 100 and far < 20
    # Word forms of the same stem still overlap.
    assert chrf("Знаком Ігні", "Знак Ігні") > 40 > chrf("Тікай звідси", "Знак Ігні")
    prompt = judge_prompt([{"role": "user", "content": "Q?"}], "Ref.", [("A", "x"), ("B", "")])
    assert "--- A ---\nx" in prompt and "--- B ---\n(empty)" in prompt


def test_compare_models_with_a_judge(hints, settings, tiny_model, judge_url):
    client = hints
    shutil.copytree(tiny_model, settings.models_dir / "tiny")
    body = {"kind": "openai", "base_url": judge_url, "model": "judge"}
    judge = client.post("/api/teachers", json=body).json()["ref"]
    held_out = client.get("/api/datasets/hints/examples", params={"split": "eval"}).json()["total"]
    assert held_out > 0

    started = client.post(
        "/api/evaluations",
        json={"dataset": "hints", "models": ["echo", "tiny"], "judge": judge, "max_tokens": 16},
    )
    assert started.status_code == 202, started.json()
    report = wait(client, started.json()["id"])
    assert report["state"] == "done", report
    assert report["done"] == report["total"] == held_out * 3  # two models answer, one judging

    echo, tiny = report["summary"]["echo"], report["summary"]["tiny"]
    assert echo["answers"] == tiny["answers"] == held_out
    assert echo["score"] == 1 and tiny["score"] == 4 and echo["judged"] == held_out
    assert 0 < echo["similarity"] < 100 and echo["seconds"] >= 0

    example = report["examples"][0]
    assert example["messages"][-1]["role"] == "user"
    assert example["reference"].startswith("Use Igni")
    answer = example["answers"]["echo"]
    assert answer["text"] == f"Echo: {example['messages'][-1]['content']}"
    assert answer["reason"] == "checked"
    # The judge sees the reference and both answers, not which model wrote them.
    assert example["reference"] in FakeJudge.prompts[0] and "tiny" not in FakeJudge.prompts[0]

    listed = client.get("/api/evaluations").json()["data"]
    assert [e["id"] for e in listed] == [report["id"]] and "examples" not in listed[0]
    assert client.delete(f"/api/evaluations/{report['id']}").status_code == 204
    assert client.get(f"/api/evaluations/{report['id']}").status_code == 404


def test_similarity_only_and_limit(hints):
    client = hints
    started = client.post(
        "/api/evaluations", json={"dataset": "hints", "models": ["echo"], "limit": 2}
    )
    report = wait(client, started.json()["id"])
    assert report["state"] == "done" and report["total"] == 2
    assert report["summary"]["echo"]["score"] is None
    assert report["summary"]["echo"]["judged"] == 0


def test_an_installed_model_judges(hints):
    # The echo model repeats the judge prompt, whose example reply scores answer A with 4.
    started = hints.post(
        "/api/evaluations",
        json={"dataset": "hints", "models": ["echo"], "judge": "local:echo", "limit": 2},
    )
    assert started.status_code == 202, started.text
    report = wait(hints, started.json()["id"])
    assert report["state"] == "done"
    assert report["summary"]["echo"]["score"] == 4 and report["summary"]["echo"]["judged"] == 2


def test_evaluation_errors(hints):
    client = hints
    cases = [
        ({"dataset": "nope", "models": ["echo"]}, 404),
        ({"dataset": "hints", "models": []}, 422),
        ({"dataset": "hints", "models": ["missing"]}, 422),
        ({"dataset": "hints", "models": ["echo"], "judge": "42"}, 404),
        ({"dataset": "hints", "models": ["echo"] + [f"m{i}" for i in range(4)]}, 422),
    ]
    for body, status in cases:
        assert client.post("/api/evaluations", json=body).status_code == status, body
    client.post("/api/datasets", json={"name": "all-train", "eval_percent": 0})
    client.post(
        "/api/datasets/all-train/examples",
        json={
            "messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        },
    )
    no_eval = client.post("/api/evaluations", json={"dataset": "all-train", "models": ["echo"]})
    assert no_eval.status_code == 422 and "held-out" in no_eval.json()["detail"]
    assert client.get("/api/evaluations/0123456789ab").status_code == 404
    assert client.get("/api/evaluations/..%2Fx").status_code == 404


def test_interrupted_runs_are_marked(settings, client):
    manager = client.app.state.evaluations
    stale = Evaluation(EvaluationOptions("hints", ["echo"]))
    manager._save(stale)
    reopened = type(manager)(manager.folder, manager.datasets, manager.models, manager.teachers)
    assert reopened.get(stale.id).state == "interrupted"
