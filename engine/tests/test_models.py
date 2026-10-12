import json

import pytest

from engine.models.manager import ModelManager


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
