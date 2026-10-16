import io
import json
import tarfile
import time

import pytest
from fastapi.testclient import TestClient

from engine.config import Settings
from engine.main import create_app
from engine.models.manager import ModelManager
from engine.models.settings import ModelSettings


def test_catalog_lists_default_model(client):
    body = client.get("/api/models").json()
    assert "qwen3-0.6b" in [m["id"] for m in body["catalog"]]
    assert body["installed"] == ["echo"]


def test_unknown_catalog_model_download(client):
    assert client.post("/api/models/nope/download").status_code == 404


def test_any_folder_with_config_is_a_model(tmp_path):
    (tmp_path / "my-model").mkdir()
    (tmp_path / "my-model" / "config.json").write_text(json.dumps({}))
    assert ModelManager(tmp_path).installed() == ["my-model"]


@pytest.fixture
def tiny_model(tmp_path):
    """Builds a randomly initialized, few-kilobyte Llama model with a chat template."""
    torch = pytest.importorskip("torch")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    vocab = {w: i for i, w in enumerate(["<unk>", "<s>", "</s>", "user", "assistant", "hi", "ok"])}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
    )
    path = tmp_path / "models" / "tiny"
    LlamaForCausalLM(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return tmp_path


def test_transformers_runtime_serves_both_apis(tiny_model):
    from fastapi.testclient import TestClient

    from engine.config import Settings
    from engine.main import create_app

    client = TestClient(create_app(Settings(data_dir=tiny_model)))
    openai_reply = client.post(
        "/v1/chat/completions",
        json={
            "model": "tiny",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
            "temperature": 0,
        },
    ).json()
    assert openai_reply["usage"]["completion_tokens"] <= 5
    assert openai_reply["choices"][0]["finish_reason"] in {"stop", "length"}

    anthropic_reply = client.post(
        "/v1/messages",
        headers={"anthropic-version": "2023-06-01"},
        json={
            "model": "tiny",
            "max_tokens": 5,
            "temperature": 0,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ).json()
    assert anthropic_reply["stop_reason"] in {"end_turn", "max_tokens"}
    # Same greedy decoding through both protocols gives the same answer.
    assert anthropic_reply["content"][0]["text"] == openai_reply["choices"][0]["message"]["content"]


def test_hugging_face_token_and_gated_errors(monkeypatch, tmp_path):
    import httpx
    from huggingface_hub.errors import GatedRepoError

    from engine.config import Settings
    from engine.models.manager import download_error

    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    assert Settings(data_dir=tmp_path).hf_token == "hf_from_env"
    monkeypatch.setenv("TINYLM_HF_TOKEN", "hf_tinylm")
    assert Settings(data_dir=tmp_path).hf_token == "hf_tinylm"

    def gated(status: int) -> GatedRepoError:
        request = httpx.Request("GET", "https://huggingface.co/google/gemma-3-1b-it")
        return GatedRepoError("gated", response=httpx.Response(status, request=request))

    assert "no Hugging Face token" in download_error("google/gemma-3-1b-it", gated(401))
    refused = download_error("google/gemma-3-1b-it", gated(403))
    assert (
        "accept the license" in refused and "https://huggingface.co/google/gemma-3-1b-it" in refused
    )
    assert download_error("x/y", RuntimeError("disk full")) == "disk full"


def chat(client, **params) -> dict:
    body = {"model": "echo", "messages": [{"role": "user", "content": "one two three"}]}
    return client.post("/v1/chat/completions", json=body | params).json()


def test_model_settings_fill_in_what_a_request_leaves_out(client):
    assert client.get("/api/models/echo/settings").json() == {}
    saved = client.put("/api/models/echo/settings", json={"max_tokens": 2, "top_k": None})
    assert saved.json() == {"max_tokens": 2}
    # The model's setting limits an answer that sets no length; a request's own length wins.
    assert chat(client)["usage"]["completion_tokens"] == 2
    assert chat(client, max_tokens=50)["choices"][0]["message"]["content"] == "Echo: one two three"
    # Saving replaces all settings: left out goes back to the engine's default.
    client.put("/api/models/echo/settings", json={})
    assert chat(client)["choices"][0]["message"]["content"] == "Echo: one two three"
    assert client.put("/api/models/echo/settings", json={"temperature": 5}).status_code == 422
    assert client.get("/api/models/nope/settings").status_code == 404


def test_answer_defaults():
    from engine.chat import ChatRequest
    from engine.models.settings import with_defaults

    request = with_defaults(ChatRequest("m", [], temperature=0.2), {"temperature": 1, "top_k": 20})
    assert (request.max_tokens, request.temperature, request.top_p) == (512, 0.2, 1.0)
    assert (request.top_k, request.min_p, request.repetition_penalty) == (20, None, None)
    assert request.thinking is False


def test_hub_names():
    from engine.models import hub

    assert hub.quantization("gemma-3-4b-it-Q4_K_M.gguf") == "Q4_K_M"
    assert hub.quantization("Qwen3-4B-IQ4_XS.gguf") == "IQ4_XS"
    assert hub.quantization("model-bf16.gguf") == "BF16"
    assert hub.model_id("unsloth/gemma-3-4b-it-GGUF", "gguf", "gemma-3-4b-it-Q4_K_M.gguf") == (
        "gemma-3-4b-it-q4_k_m"
    )
    split = "Q8_0/Big-Model-Q8_0-00001-of-00002.gguf"
    assert hub.model_id("x/Big-Model-GGUF", "gguf", split) == "big-model-q8_0"
    assert hub.gguf_patterns(split) == ["Q8_0/Big-Model-Q8_0-*-of-00002.gguf"]
    assert hub.model_id("Qwen/Qwen2.5-0.5B-Instruct", "hf") == "qwen2.5-0.5b-instruct"


class FakeHub:
    """Stands in for huggingface_hub.HfApi: two repos, one GGUF and one with safetensors."""

    def __init__(self, token=None):
        pass

    def list_models(self, search=None, filter=None, **kwargs):
        from datetime import UTC, datetime
        from types import SimpleNamespace

        when = datetime(2026, 5, 1, tzinfo=UTC)
        if filter == "gguf":
            gguf = {"total": 4_000_000_000, "context_length": 131072}
            yield SimpleNamespace(
                id="unsloth/gemma-3-4b-it-GGUF",
                downloads=900,
                likes=50,
                gated=False,
                last_modified=when,
                pipeline_tag="image-text-to-text",
                gguf=gguf,
            )
            return
        for repo, pipeline in [("Qwen/Qwen3-0.6B", "text-generation"), ("x/bert", "fill-mask")]:
            yield SimpleNamespace(
                id=repo,
                downloads=10,
                likes=1,
                gated="manual" if repo == "x/bert" else False,
                last_modified=None,
                pipeline_tag=pipeline,
                safetensors=SimpleNamespace(total=600_000_000),
            )

    def model_info(self, repo, files_metadata=False):
        from types import SimpleNamespace

        names = {
            "unsloth/gemma-3-4b-it-GGUF": {
                "gemma-3-4b-it-Q4_K_M.gguf": 2500,
                "gemma-3-4b-it-Q8_0.gguf": 4100,
                "mmproj-F16.gguf": 800,
                "BF16/gemma-3-4b-it-BF16-00001-of-00002.gguf": 4000,
                "BF16/gemma-3-4b-it-BF16-00002-of-00002.gguf": 3000,
                "README.md": 10,
            },
            "Qwen/Qwen3-0.6B": {"config.json": 1, "model.safetensors": 1200, "README.md": 5},
        }[repo]
        siblings = [SimpleNamespace(rfilename=n, size=s) for n, s in names.items()]
        return SimpleNamespace(siblings=siblings, gated=False)


def test_search_hugging_face(client, monkeypatch):
    from engine.models import hub

    monkeypatch.setattr(hub, "HfApi", FakeHub)
    gguf = client.get("/api/hub/models", params={"q": "gemma", "format": "gguf"}).json()["data"]
    assert gguf[0]["repo"] == "unsloth/gemma-3-4b-it-GGUF"
    assert (gguf[0]["params"], gguf[0]["context_length"]) == (4_000_000_000, 131072)
    # Models that do not write text are left out.
    hf = client.get("/api/hub/models", params={"q": "", "format": "hf"}).json()["data"]
    assert [m["repo"] for m in hf] == ["Qwen/Qwen3-0.6B"]

    files = client.get("/api/hub/files", params={"repo": "unsloth/gemma-3-4b-it-GGUF"}).json()
    assert [(f["quantization"], f["size"], f["parts"]) for f in files["gguf"]] == [
        ("Q4_K_M", 2500, 1),
        ("Q8_0", 4100, 1),
        ("BF16", 7000, 2),
    ]
    assert files["gguf"][2]["file"] == "BF16/gemma-3-4b-it-BF16-00001-of-00002.gguf"
    assert files["hf"] is None
    assert client.get("/api/hub/files", params={"repo": "Qwen/Qwen3-0.6B"}).json()["hf"] == {
        "size": 1201
    }


def test_download_from_hugging_face_and_delete(client, settings, monkeypatch):
    import time

    import huggingface_hub

    asked = []

    def snapshot_download(repo, local_dir, token, allow_patterns):
        asked.append(allow_patterns)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "BF16").mkdir()
        (local_dir / "BF16" / "gemma-3-4b-it-BF16-00001-of-00002.gguf").write_bytes(b"x" * 40)
        (local_dir / "BF16" / "gemma-3-4b-it-BF16-00002-of-00002.gguf").write_bytes(b"x" * 30)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHub)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    body = {
        "repo": "unsloth/gemma-3-4b-it-GGUF",
        "file": "BF16/gemma-3-4b-it-BF16-00001-of-00002.gguf",
    }
    job = client.post("/api/hub/download", json=body).json()
    assert job["model_id"] == "gemma-3-4b-it-bf16"
    for _ in range(50):
        info = client.get("/api/models").json()
        if info["downloads"][0]["state"] != "downloading":
            break
        time.sleep(0.1)
    assert info["downloads"][0] | {"error": None} == {
        "model_id": "gemma-3-4b-it-bf16",
        "repo": "unsloth/gemma-3-4b-it-GGUF",
        "state": "done",
        "error": None,
        "total": 7000,
        "downloaded": 7000,
    }
    assert asked == [["BF16/gemma-3-4b-it-BF16-*-of-00002.gguf"]]
    [model] = info["models"]
    assert model["id"] == "gemma-3-4b-it-bf16" and model["format"] == "gguf"
    assert model["source"] == body and model["size"] > 70 and not model["loaded"]
    # Both parts sit at the top of the folder, where llama.cpp finds the first one.
    folder = settings.models_dir / "gemma-3-4b-it-bf16"
    assert len(list(folder.glob("*.gguf"))) == 2
    assert client.post("/api/hub/download", json=body).status_code == 409
    bad = client.post("/api/hub/download", json={"repo": "x/y", "file": "README.md"})
    assert bad.status_code == 400

    client.put("/api/models/gemma-3-4b-it-bf16/settings", json={"context_size": 8192})
    assert client.delete("/api/models/gemma-3-4b-it-bf16").status_code == 204
    assert not folder.exists()
    assert client.get("/api/models").json()["models"] == []
    assert "gemma-3-4b-it-bf16" not in (settings.models_dir / ".model-settings.json").read_text()


def test_model_collection_answers_with_sources(client):
    client.post("/api/collections", json={"name": "games"})
    client.post(
        "/api/collections/games/documents",
        files=[("files", ("witcher.md", "Утопці бояться знаку Ігні.".encode()))],
    )
    assert "sources" not in chat(client)
    bad = client.put("/api/models/echo/settings", json={"collection": "nope"})
    assert bad.status_code == 422
    client.put("/api/models/echo/settings", json={"collection": "games", "rag_top_k": 1})
    # A request that names no collection answers from the model's one.
    answer = chat(client, debug=True)
    assert answer["collection"] == "games" and len(answer["sources"]) == 1
    assert "Утопці" in answer["prompt"]["messages"][0]["content"]
    # "" asks for none.
    assert "sources" not in chat(client, collection="")


def wait_for_upload(client, upload: dict) -> dict:
    for _ in range(100):
        upload = client.get(f"/api/models/uploads/{upload['id']}").json()
        if upload["state"] not in ("receiving", "installing"):
            return upload
        time.sleep(0.05)
    raise AssertionError("the upload did not finish")


def send_model(client, name: str, data: bytes, part: int = 7) -> dict:
    upload = client.post("/api/models/uploads", json={"name": name, "size": len(data)}).json()
    for offset in range(0, len(data), part):
        path = f"/api/models/uploads/{upload['id']}?offset={offset}"
        assert client.put(path, content=data[offset : offset + part]).status_code == 200
    finished = client.post(f"/api/models/uploads/{upload['id']}/finish")
    assert finished.status_code == 202, finished.text
    return wait_for_upload(client, upload)


def test_move_a_model_between_engines(client, settings, tmp_path):
    adapter = settings.models_dir / "hints-lora"
    (adapter / "sub").mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "tinylm.json").write_text(json.dumps({"base": "qwen-base"}))
    (adapter / "sub" / "adapter_model.safetensors").write_bytes(bytes(range(256)) * 9)
    (adapter / ".cache").mkdir()
    (adapter / ".cache" / "lock").write_text("x")
    # Settings of a model whose base is not installed can only be set in the store.
    client.app.state.models.settings.set("hints-lora", ModelSettings(temperature=0.3))

    response = client.get("/api/models/hints-lora/archive")
    assert response.status_code == 200
    assert int(response.headers["content-length"]) == len(response.content)
    with tarfile.open(fileobj=io.BytesIO(response.content)) as tar:
        assert sorted(tar.getnames()) == [
            "hints-lora/adapter_config.json",
            "hints-lora/sub/adapter_model.safetensors",
            "hints-lora/tinylm-settings.json",
            "hints-lora/tinylm.json",
        ]
    assert client.get("/api/models/nope/archive").status_code == 404

    # On another engine: installed with its settings; its base model is missing there.
    other = TestClient(create_app(Settings(data_dir=tmp_path / "other", enable_echo=True)))
    upload = send_model(other, "hints-lora.tar", response.content)
    assert (upload["state"], upload["model"], upload["missing_base"]) == (
        "done",
        "hints-lora",
        "qwen-base",
    )
    copied = tmp_path / "other" / "models" / "hints-lora" / "sub" / "adapter_model.safetensors"
    assert copied.read_bytes() == bytes(range(256)) * 9
    saved = json.loads((tmp_path / "other" / "models" / ".model-settings.json").read_text())
    assert saved["hints-lora"]["temperature"] == 0.3
    trained = other.get("/api/models/trained").json()["data"]
    assert trained[0]["id"] == "hints-lora"
    assert other.get("/api/models/hints-lora/archive").status_code == 200
    # The same model again: refused, nothing changed.
    again = send_model(other, "hints-lora.tar", response.content)
    assert again["state"] == "error" and "already installed" in again["error"]

    # A single GGUF file becomes a model folder.
    upload = send_model(other, "My Model-Q4_K_M.gguf", b"GGUF" + b"\0" * 40)
    assert (upload["state"], upload["model"]) == ("done", "my-model-q4_k_m")
    assert "my-model-q4_k_m" in other.get("/api/models").json()["installed"]

    # Parts must come in order and add up to the size.
    upload = other.post("/api/models/uploads", json={"name": "a.gguf", "size": 4}).json()
    path = f"/api/models/uploads/{upload['id']}"
    assert other.put(f"{path}?offset=2", content=b"GG").status_code == 409
    assert other.put(f"{path}?offset=0", content=b"GGUF!").status_code == 409
    assert other.post(f"{path}/finish").status_code == 409
    bad = other.post("/api/models/uploads", json={"name": "a.zip", "size": 4})
    assert bad.status_code == 422

    # An archive with a path out of its folder is refused.
    evil = io.BytesIO()
    with tarfile.open(fileobj=evil, mode="w") as tar:
        info = tarfile.TarInfo("evil/../../escape.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    upload = send_model(other, "evil.tar", evil.getvalue())
    assert upload["state"] == "error"
    assert not (tmp_path / "other" / "escape.txt").exists()
