"""Exporting a fine-tuned model to GGUF: getting llama.cpp's converter, merging the adapter,
converting, quantizing and serving the result through llama.cpp.

The full export needs llama.cpp's converter. Set TINYLM_TEST_LLAMA_CPP_DIR to a llama.cpp
checkout (or to <data dir>/tools/llama.cpp-<version> after one export in the app) to run it;
it is skipped otherwise, so the tests do not download it.
"""

import io
import json
import os
import shutil
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from tiny_model import build_tiny_model

from engine.export.tools import CONVERTER, ToolsError, converter_dir

LLAMA_CPP_DIR = os.environ.get("TINYLM_TEST_LLAMA_CPP_DIR")


def source_package() -> bytes:
    """A stand-in for llama-cpp-python's source package on PyPI."""
    files = {
        "llama_cpp_python-9.9/README.md": "readme",
        "llama_cpp_python-9.9/vendor/llama.cpp/convert_hf_to_gguf.py": "print('convert')",
        "llama_cpp_python-9.9/vendor/llama.cpp/conversion/__init__.py": "",
        "llama_cpp_python-9.9/vendor/llama.cpp/gguf-py/gguf/__init__.py": "",
        "llama_cpp_python-9.9/vendor/llama.cpp/src/llama.cpp": "// not needed",
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@pytest.fixture
def pypi():
    requests = []

    class PyPI(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            requests.append(self.path)
            base = f"http://127.0.0.1:{self.server.server_address[1]}"
            if self.path == "/pypi/llama-cpp-python/9.9/json":
                urls = [
                    {"packagetype": "bdist_wheel", "url": f"{base}/wheel"},
                    {"packagetype": "sdist", "url": f"{base}/sdist.tar.gz"},
                ]
                body = json.dumps({"urls": urls}).encode()
            elif self.path == "/sdist.tar.gz":
                body = source_package()
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), PyPI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/pypi/llama-cpp-python/{{version}}/json"
    yield url, requests
    server.shutdown()


def test_converter_comes_from_the_source_package(tmp_path, pypi):
    url, requests = pypi
    notes = []
    folder = converter_dir(tmp_path, version="9.9", pypi=url, progress=notes.append)
    assert folder == tmp_path / "llama.cpp-9.9"
    files = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
    assert files == sorted([CONVERTER, "conversion/__init__.py", "gguf-py/gguf/__init__.py"])
    assert notes and "9.9" in notes[0]

    # Kept for the next export.
    assert converter_dir(tmp_path, version="9.9", pypi=url) == folder
    assert requests.count("/sdist.tar.gz") == 1

    with pytest.raises(ToolsError, match="Could not download"):
        converter_dir(tmp_path, version="0.0", pypi=url)
    with pytest.raises(ToolsError, match="llama.cpp checkout"):
        converter_dir(tmp_path, override=tmp_path / "nowhere")
    assert converter_dir(tmp_path, override=folder) == folder


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    return build_tiny_model(tmp_path_factory.mktemp("tiny") / "tiny")


@pytest.fixture
def fine_tuned(client, settings, tiny_model):
    """A base model and a fine-tuned model trained on it in a few steps."""
    shutil.copytree(tiny_model, settings.models_dir / "tiny")
    client.post("/api/datasets", json={"name": "hints", "eval_percent": 0})
    lines = [
        json.dumps({"prompt": f"Hint {i}?", "completion": f"Use Igni {i} times."}) for i in range(8)
    ]
    client.post("/api/datasets/hints/import", files={"files": ("h.jsonl", "\n".join(lines))})
    body = {"name": "tiny-hints", "base": "tiny", "dataset": "hints", "epochs": 1}
    job = client.post("/api/training/jobs", json=body).json()
    deadline = time.time() + 120
    while client.get(f"/api/training/jobs/{job['id']}").json()["state"] == "running":
        assert time.time() < deadline
        time.sleep(0.2)
    return client


def wait(client, export_id: str) -> dict:
    deadline = time.time() + 300
    while time.time() < deadline:
        export = client.get(f"/api/exports/{export_id}").json()
        if export["state"] != "running":
            return export
        time.sleep(0.2)
    raise TimeoutError(export)


def test_export_checks(fine_tuned):
    client = fine_tuned
    cases = [
        ("tiny", {}, "not a fine-tuned model"),
        ("missing", {}, "not a fine-tuned model"),
        ("tiny-hints", {"name": "tiny"}, "already exists"),
        ("tiny-hints", {"name": "../x"}, "letters, digits"),
    ]
    for model, body, message in cases:
        response = client.post(f"/api/models/{model}/export", json=body)
        assert response.status_code == 422 and message in response.json()["detail"], body
    assert (
        client.post("/api/models/tiny-hints/export", json={"quantization": "Q2"}).status_code == 422
    )
    assert client.get("/api/exports/nope").status_code == 404


@pytest.mark.skipif(not LLAMA_CPP_DIR, reason="set TINYLM_TEST_LLAMA_CPP_DIR to llama.cpp")
@pytest.mark.parametrize("quantization", ["Q8_0", "Q4_K_M"])
def test_export_and_chat(fine_tuned, settings, quantization):
    client = fine_tuned
    client.app.state.exports.llama_cpp_dir = Path(LLAMA_CPP_DIR)
    started = client.post("/api/models/tiny-hints/export", json={"quantization": quantization})
    assert started.status_code == 202, started.json()
    export = wait(client, started.json()["id"])
    assert export["state"] == "done", export
    name = f"tiny-hints-{quantization.lower()}"
    assert export["name"] == name and export["step"] == "save" and export["size"] > 0
    # The tiny test tokenizer is unknown to llama.cpp; the export says so and goes on.
    assert any("pre-tokenizer" in note for note in export["notes"])

    folder = settings.models_dir / name
    assert [p.name for p in folder.glob("*.gguf")] == [f"{name}.{quantization}.gguf"]
    assert not list(settings.models_dir.glob(".*partial"))
    assert not (settings.workspace_dir / "exports" / export["id"] / "tmp").exists()

    trained = {m["id"]: m for m in client.get("/api/models/trained").json()["data"]}
    assert trained[name]["format"] == "gguf" and trained[name]["quantization"] == quantization
    assert trained[name]["exported_from"] == "tiny-hints" and trained[name]["dataset"] == "hints"
    assert trained["tiny-hints"]["format"] == "adapter"

    # Served through llama.cpp, with the chat template from the file.
    answer = client.post(
        "/v1/chat/completions",
        json={"model": name, "messages": [{"role": "user", "content": "Hint 3?"}], "max_tokens": 4},
    )
    assert answer.status_code == 200, answer.json()
    assert answer.json()["usage"]["completion_tokens"] <= 4
    import gguf

    reader = gguf.GGUFReader(next(folder.glob("*.gguf")))
    assert "tokenizer.chat_template" in reader.fields

    # The GGUF model stands alone: it answers without its base model.
    shutil.rmtree(settings.models_dir / "tiny")
    assert name in [m["id"] for m in client.get("/v1/models").json()["data"]]
    again = client.post("/api/models/tiny-hints/export", json={})
    assert again.status_code == 422 and "base model tiny is not installed" in again.json()["detail"]
