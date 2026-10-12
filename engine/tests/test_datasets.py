"""Training datasets: file formats, validation, the train/eval split and the API."""

import json

import pytest

from engine.datasets.formats import InvalidExample, parse, to_messages
from engine.datasets.store import DatasetStore

QA = [{"role": "user", "content": "Hint?"}, {"role": "assistant", "content": "Use Igni."}]


def test_record_shapes():
    assert to_messages({"messages": QA}) == QA
    assert to_messages({"prompt": " Hint? ", "completion": "Use Igni."}) == QA
    assert to_messages({"question": "Hint?", "answer": "Use Igni."}) == QA
    sharegpt = {
        "conversations": [
            {"from": "human", "value": "Hint?"},
            {"from": "gpt", "value": "Use Igni."},
        ]
    }
    assert to_messages(sharegpt) == QA
    alpaca = {
        "instruction": "Give a hint",
        "input": "Witcher 3",
        "output": "Use Igni.",
        "system": "Be brief.",
    }
    assert to_messages(alpaca) == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Give a hint\n\nWitcher 3"},
        {"role": "assistant", "content": "Use Igni."},
    ]


@pytest.mark.parametrize(
    ("record", "error"),
    [
        ({"messages": [{"role": "user", "content": "Hint?"}]}, "last message"),
        ({"messages": [{"role": "assistant", "content": "Hi"}]}, "no user message"),
        ({"messages": [{"role": "bot", "content": "Hi"}]}, "role 'bot'"),
        ({"messages": [*QA[:1], {"role": "assistant", "content": " "}]}, "no text"),
        ({"prompt": "Hint?", "completion": ""}, "no text"),
        ({"text": "free text"}, "unknown format"),
        (["not", "an", "object"], "JSON object"),
    ],
)
def test_invalid_records(record, error):
    with pytest.raises(InvalidExample, match=error):
        to_messages(record)


def test_files_report_bad_lines():
    jsonl = "\n".join([json.dumps({"messages": QA}), "{broken", "", json.dumps({"text": "x"})])
    records = list(parse("hints.jsonl", jsonl.encode()))
    assert [(r.line, r.messages is not None) for r in records] == [
        (1, True),
        (2, False),
        (4, False),
    ]

    csv = 'Question,Answer\nHint?,Use Igni.\n"Боси, мапи?","Так, є."\nNo answer,\n'
    records = list(parse("hints.csv", csv.encode("utf-8-sig")))
    assert records[1].messages[0]["content"] == "Боси, мапи?"
    assert records[2].line == 4 and "no text" in records[2].error

    array = json.dumps([{"prompt": "Hint?", "completion": "Use Igni."}])
    assert next(parse("hints.json", array.encode())).messages == QA
    with pytest.raises(InvalidExample, match=".jsonl"):
        list(parse("hints.txt", b""))


def test_split_is_stable(store):
    datasets = DatasetStore(store)
    datasets.create("hints", eval_percent=20)
    lines = [json.dumps({"prompt": f"Hint {i}?", "completion": f"Answer {i}."}) for i in range(200)]
    result = datasets.import_file("hints", "h.jsonl", "\n".join(lines).encode())
    assert result["added"] == 200

    info = datasets.dataset("hints")
    assert info["examples"] == 200 and 25 <= info["eval"] <= 55
    held_out = {e["id"] for e in datasets.examples("hints", split="eval")}
    assert len(held_out) == info["eval"] == datasets.count("hints", split="eval")

    # Other examples coming and going do not move an example between the parts.
    datasets.add("hints", [{"role": "user", "content": "New?"}, QA[1]])
    datasets.remove("hints", min(held_out))
    assert {e["id"] for e in datasets.examples("hints", split="eval")} == held_out - {min(held_out)}

    train, evaluation = datasets.split("hints")
    assert len(train) + len(evaluation) == 200 and len(evaluation) == len(held_out) - 1
    datasets.update("hints", 0)
    assert datasets.dataset("hints")["eval"] == 0


def test_datasets_api(client):
    assert client.post("/api/datasets", json={"name": "bad name"}).status_code == 422
    assert client.post("/api/datasets", json={"name": "hints"}).json()["eval_percent"] == 10

    jsonl = "\n".join(
        [
            json.dumps({"prompt": "Підказка для Відьмака?", "completion": "Знак Ігні."}),
            json.dumps({"prompt": "Подсказка?", "completion": "Знак Игни."}),
            json.dumps({"prompt": "Подсказка?", "completion": "Знак Игни."}),
            "{broken",
        ]
    )
    files = {"files": ("hints.jsonl", jsonl.encode())}
    report = client.post("/api/datasets/hints/import", files=files).json()["data"][0]
    assert (report["added"], report["duplicates"], report["invalid"]) == (2, 1, 1)
    assert report["errors"] == [{"line": 4, "error": report["errors"][0]["error"]}]
    bad_file = client.post("/api/datasets/hints/import", files={"files": ("a.txt", b"x")})
    assert "error" in bad_file.json()["data"][0]

    page = client.get("/api/datasets/hints/examples", params={"query": "ВІДЬМАК"}).json()
    assert page["total"] == 1
    example = page["data"][0]
    assert example["split"] in ("train", "eval")

    changed = [*example["messages"][:1], {"role": "assistant", "content": "Знак Аард."}]
    path = f"/api/datasets/hints/examples/{example['id']}"
    assert client.put(path, json={"messages": changed}).status_code == 204
    texts = [
        e["messages"][1]["content"]
        for e in client.get("/api/datasets/hints/examples").json()["data"]
    ]
    assert "Знак Аард." in texts
    invalid = client.put(path, json={"messages": changed[:1]})
    assert invalid.status_code == 422 and "last message" in invalid.json()["detail"]

    added = client.post("/api/datasets/hints/examples", json={"messages": QA})
    assert added.status_code == 201 and not added.json()["duplicate"]
    assert client.post("/api/datasets/hints/examples", json={"messages": QA}).json()["duplicate"]

    assert (
        client.patch("/api/datasets/hints", json={"eval_percent": 30}).json()["eval_percent"] == 30
    )
    assert client.get("/api/datasets").json()["data"][0]["examples"] == 3
    assert client.delete(path).status_code == 204
    assert client.delete(path).status_code == 404
    assert client.delete("/api/datasets/hints").status_code == 204
    assert client.get("/api/datasets/hints/examples").status_code == 404
