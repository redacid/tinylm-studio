"""End-to-end test of the llama.cpp runtime on a tiny, randomly initialized GGUF model."""

import pytest
from fastapi.testclient import TestClient

from engine.config import Settings
from engine.main import create_app

pytest.importorskip("llama_cpp")
gguf = pytest.importorskip("gguf")
np = pytest.importorskip("numpy")


def write_tiny_gguf(path) -> None:
    words = ["▁hi", "▁ok", "▁user", "▁assistant", "▁"]
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)] + words
    types = [2, 3, 3] + [6] * 256 + [1] * len(words)
    n_vocab, n_embd, n_ff, n_head = len(tokens), 16, 32, 2
    rng = np.random.default_rng(0)

    def weight(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.1

    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_context_length(256)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(1)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head)
    writer.add_rope_dimension_count(n_embd // n_head)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * n_vocab)
    writer.add_token_types(types)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(0)
    writer.add_chat_template(
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
    )
    tensors = {
        "token_embd.weight": weight(n_vocab, n_embd),
        "output_norm.weight": np.ones(n_embd, dtype=np.float32),
        "output.weight": weight(n_vocab, n_embd),
        "blk.0.attn_norm.weight": np.ones(n_embd, dtype=np.float32),
        "blk.0.attn_q.weight": weight(n_embd, n_embd),
        "blk.0.attn_k.weight": weight(n_embd, n_embd),
        "blk.0.attn_v.weight": weight(n_embd, n_embd),
        "blk.0.attn_output.weight": weight(n_embd, n_embd),
        "blk.0.ffn_norm.weight": np.ones(n_embd, dtype=np.float32),
        "blk.0.ffn_gate.weight": weight(n_ff, n_embd),
        "blk.0.ffn_up.weight": weight(n_ff, n_embd),
        "blk.0.ffn_down.weight": weight(n_embd, n_ff),
    }
    for name, value in tensors.items():
        writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture
def client(tmp_path) -> TestClient:
    model_dir = tmp_path / "models" / "tiny-gguf"
    model_dir.mkdir(parents=True)
    write_tiny_gguf(model_dir / "tiny-Q8_0.gguf")
    return TestClient(create_app(Settings(data_dir=tmp_path, threads=2)))


def test_gguf_folder_is_listed(client):
    assert client.get("/v1/models").json()["data"][0]["id"] == "tiny-gguf"


def test_llamacpp_serves_both_apis(client):
    openai_reply = client.post(
        "/v1/chat/completions",
        json={
            "model": "tiny-gguf",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
            "temperature": 0,
        },
    ).json()
    assert openai_reply["usage"]["completion_tokens"] <= 5
    assert openai_reply["usage"]["prompt_tokens"] > 0
    assert openai_reply["choices"][0]["finish_reason"] in {"stop", "length"}

    anthropic_reply = client.post(
        "/v1/messages",
        headers={"anthropic-version": "2023-06-01"},
        json={
            "model": "tiny-gguf",
            "max_tokens": 5,
            "temperature": 0,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ).json()
    assert anthropic_reply["stop_reason"] in {"end_turn", "max_tokens"}
    assert anthropic_reply["content"][0]["text"] == openai_reply["choices"][0]["message"]["content"]


def test_llamacpp_streaming(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "tiny-gguf",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "stream": True,
        },
    ) as response:
        body = response.read().decode()
    assert body.rstrip().endswith("data: [DONE]")


def test_llamacpp_model_settings(client):
    ask = {"model": "tiny-gguf", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3}
    assert client.post("/v1/chat/completions", json=ask).status_code == 200
    models = client.app.state.models
    assert models.runtime("tiny-gguf").runtime.llm.n_ctx() == 4096
    settings = {"context_size": 512, "batch_size": 64, "top_k": 5, "min_p": 0.1, "thinking": True}
    assert client.put("/api/models/tiny-gguf/settings", json=settings).status_code == 200
    # A loading setting unloads the model; it is loaded with the new one on the next request.
    assert models.loaded() is None
    assert client.post("/v1/chat/completions", json=ask).status_code == 200
    assert models.runtime("tiny-gguf").runtime.llm.n_ctx() == 512
    # Answer settings apply at once, without loading the model again.
    client.put("/api/models/tiny-gguf/settings", json=settings | {"top_k": 1})
    assert models.loaded() == "tiny-gguf"
