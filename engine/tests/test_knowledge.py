"""Uploaded data: conversion, chunking, collections, search and grounded chat answers."""

import json
import sqlite3
import threading
import time

import anthropic
import numpy as np
import openai
import pytest

from engine.chat import ChatRequest, Message
from engine.models.settings import ModelSettings
from engine.rag.chunking import chunk_text
from engine.rag.convert import UnsupportedFile, convert, strip_markup
from engine.rag.embeddings import HashEmbedder
from engine.rag.prompt import ground
from engine.rag.store import Hit, KnowledgeStore, full_text_query
from engine.web.sites import Website, WebsiteStore

WITCHER = "The Witcher 3 is an open-world RPG. Geralt of Rivia hunts monsters for coin."
SOULS = "Dark Souls bosses punish greed. Learn the attack rhythm and roll sideways."


def test_chunks_respect_the_size_limit():
    text = "\n\n".join(["Short paragraph."] * 3 + ["Long sentence here. " * 200])
    chunks = chunk_text(text, max_chars=300)
    assert chunks[0].startswith("Short paragraph.\n\nShort paragraph.")
    assert all(len(c) <= 300 for c in chunks)
    assert "".join(chunks).count("Long sentence here.") == 200


PAGE = """<html><head><title>Hollow Knight review</title></head><body>
<nav><a href="/">Home</a> <a href="/guides/">Guides</a> <a href="/news/">News</a></nav>
<article><h1>Hollow Knight review</h1>
<p>Hollow Knight is a metroidvania about a small knight exploring the ruined kingdom of
Hallownest. Its combat is precise and every boss teaches a new pattern to learn.</p>
<p>The map is huge and rewards curiosity: shortcuts, hidden rooms and charms that change
how the knight plays. It is one of the best games of its genre.</p>
<img src="/shot.jpg" alt="screenshot"></article>
<footer>Copyright 2026. <a href="/privacy">Privacy</a></footer></body></html>"""


def test_web_pages_keep_main_content_and_title():
    page = convert("index.html", PAGE.encode())
    assert page.title == "Hollow Knight review"
    assert "ruined kingdom of" in page.text
    assert "Privacy" not in page.text and "Guides" not in page.text


def test_converts_other_formats():
    assert "| 1 | ї |" in convert("table.csv", "a,b\n1,ї\n".encode()).text
    assert convert("notes.md", "# Привіт\n".encode()).text == "# Привіт"
    assert strip_markup("See ![shot](a.png)[the guide](https://x.y/g).") == "See the guide."
    with pytest.raises(UnsupportedFile):
        convert("image.png", b"\x89PNG")


def test_ground_prepends_numbered_sources():
    request = ChatRequest("m", [Message("system", "Be brief."), Message("user", "Who is Geralt?")])
    hit = Hit(1, 1, "witcher.txt", "witcher.txt", WITCHER, 0.9)
    messages = ground(request, [hit]).messages
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].content.startswith("Be brief.")
    assert f"[1] witcher.txt\n{WITCHER}" in messages[0].content


def upload(client, collection: str, *files: tuple[str, bytes]) -> list[dict]:
    response = client.post(
        f"/api/collections/{collection}/documents",
        files=[("files", (name, data)) for name, data in files],
    )
    assert response.status_code == 200
    return response.json()["data"]


def test_collection_lifecycle(client):
    assert client.post("/api/collections", json={"name": "ігри"}).status_code == 201
    assert client.post("/api/collections", json={"name": "bad name!"}).status_code == 422
    results = upload(
        client,
        "ігри",
        ("witcher.txt", WITCHER.encode()),
        ("souls.md", SOULS.encode()),
        ("copy.txt", WITCHER.encode()),
        ("photo.png", b"\x89PNG"),
    )
    assert [r.get("duplicate") for r in results[:3]] == [False, False, True]
    assert "Unsupported" in results[3]["error"]

    documents = client.get("/api/collections/ігри/documents").json()["data"]
    assert [d["name"] for d in documents] == ["witcher.txt", "souls.md"]

    hits = client.post("/api/collections/ігри/search", json={"query": "Dark Souls bosses"})
    assert hits.json()["data"][0]["document"] == "souls.md"

    souls = documents[1]["id"]
    assert client.delete(f"/api/collections/ігри/documents/{souls}").status_code == 204
    hits = client.post("/api/collections/ігри/search", json={"query": "Dark Souls bosses"})
    assert [h["document"] for h in hits.json()["data"]] == ["witcher.txt"]

    assert client.get("/api/collections").json()["data"][0]["documents"] == 1
    assert client.delete("/api/collections/ігри").status_code == 204
    assert client.get("/api/collections/ігри/documents").status_code == 404


def wait_for_import(client, job: dict) -> dict:
    for _ in range(100):
        job = client.get(f"/api/collections/imports/{job['id']}").json()
        if job["state"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("the import did not finish")


def test_collection_export_and_import(games):
    exported = games.get("/api/collections/games/export")
    assert exported.status_code == 200
    assert "games.tinylm.jsonl" in exported.headers["content-disposition"]
    header, *documents = [json.loads(line) for line in exported.text.splitlines()]
    assert header["collection"] == "games" and header["documents"] == 2
    assert [d["name"] for d in documents] == ["witcher.txt", "souls.txt"]

    def send(data: bytes, collection: str = "") -> dict:
        response = games.post(
            "/api/collections/import",
            files={"file": ("games.tinylm.jsonl", data)},
            data={"collection": collection},
        )
        assert response.status_code == 202, response.text
        return wait_for_import(games, response.json())

    # Into a new collection, by another name: made, indexed and searchable.
    job = send(exported.content, "copy")
    assert (job["state"], job["total"], job["added"], job["duplicates"]) == ("done", 2, 2, 0)
    hits = games.post("/api/collections/copy/search", json={"query": "Dark Souls bosses"})
    assert hits.json()["data"][0]["document"] == "souls.txt"
    # Into the collection it came from (the header's name): nothing new.
    job = send(exported.content)
    assert (job["collection"], job["added"], job["duplicates"]) == ("games", 0, 2)

    # A model's collection.jsonl has no header: it needs a name.
    bundle = "\n".join(json.dumps(d) for d in documents).encode()
    response = games.post("/api/collections/import", files={"file": ("c.jsonl", bundle)})
    assert response.status_code == 422
    assert send(bundle, "bundle")["added"] == 2
    response = games.post("/api/collections/import", files={"file": ("x.txt", b"not json")})
    assert response.status_code == 422
    assert games.get("/api/collections/nope/export").status_code == 404
    assert games.get("/api/collections/imports/nope").status_code == 404


@pytest.fixture
def games(client):
    client.post("/api/collections", json={"name": "games"})
    upload(client, "games", ("witcher.txt", WITCHER.encode()), ("souls.txt", SOULS.encode()))
    return client


def test_openai_answer_carries_sources(games):
    api = openai.OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=games)
    messages = [{"role": "user", "content": "Who hunts monsters in The Witcher 3?"}]
    reply = api.chat.completions.create(
        model="echo", messages=messages, extra_body={"collection": "games"}
    )
    assert reply.sources[0]["document"] == "witcher.txt"
    assert reply.sources[0]["index"] == 1

    stream = api.chat.completions.create(
        model="echo", messages=messages, stream=True, extra_body={"collection": "games"}
    )
    first = next(iter(stream))
    assert first.sources[0]["document"] == "witcher.txt"
    assert not hasattr(first, "prompt")

    # debug: the request as the model got it, with the sources in the system prompt.
    stream = api.chat.completions.create(
        model="echo",
        messages=messages,
        stream=True,
        max_tokens=32,
        extra_body={"collection": "games", "debug": True},
    )
    prompt = next(iter(stream)).prompt
    assert prompt["model"] == "echo" and prompt["max_tokens"] == 32
    assert prompt["messages"][0]["role"] == "system"
    assert "hunts monsters for coin" in prompt["messages"][0]["content"]
    assert prompt["messages"][-1] == messages[0]
    assert prompt["repetition_penalty"] is None

    # Per-request number of sources and repetition penalty.
    body = {"model": "echo", "messages": messages, "collection": "games", "debug": True}
    reply = games.post(
        "/v1/chat/completions", json={**body, "rag_top_k": 1, "repetition_penalty": 1.3}
    ).json()
    assert len(reply["sources"]) == 1 and reply["prompt"]["repetition_penalty"] == 1.3
    assert len(games.post("/v1/chat/completions", json=body).json()["sources"]) == 2
    bad = games.post("/v1/chat/completions", json={**body, "rag_top_k": 0})
    assert bad.status_code == 422

    with pytest.raises(openai.NotFoundError):
        api.chat.completions.create(
            model="echo", messages=messages, extra_body={"collection": "missing"}
        )


def test_anthropic_answer_carries_sources(games):
    api = anthropic.Anthropic(base_url="http://testserver", api_key="unused", http_client=games)
    message = api.messages.create(
        model="echo",
        max_tokens=64,
        messages=[{"role": "user", "content": "How to beat Dark Souls bosses?"}],
        extra_body={"collection": "games"},
    )
    assert message.model_extra["sources"][0]["document"] == "souls.txt"


class BlindEmbedder:
    """Gives every text the same vector, so only the word index can tell chunks apart."""

    name = "blind"
    dim = 8

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), self.dim), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.ones(self.dim, dtype=np.float32)


def test_full_text_query_matches_word_forms():
    assert full_text_query("Хто такий Відьмак?") == '"хто" OR "такий" OR "відьм"*'
    assert full_text_query("Borderlands 3, 3!") == '"borderlan"* OR "3"'
    assert full_text_query("?!") == ""


def test_exact_names_are_found_by_words(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.db", BlindEmbedder())
    store.create_collection("games")
    store.create_collection("other")
    for name in ["witcher", "souls", "doom", "zelda", "mario"]:
        store.add_document("games", f"{name}.txt", f"A review of {name} with many details.")
    store.add_document("games", "bl3.txt", "Borderlands 3 is a looter shooter set on Pandora.")
    store.add_document("other", "bl3-other.txt", "Borderlands 3 again, in another collection.")
    store.add_document("games", "uk.txt", "Відьмака звуть Геральт, він полює на чудовиськ.")

    assert store.search("games", "Borderlands 3 setting", k=1)[0].document == "bl3.txt"
    assert store.search("games", "Хто такий відьмак?", k=1)[0].document == "uk.txt"

    bl3 = next(d for d in store.documents("games") if d["name"] == "bl3.txt")
    store.delete_document("games", bl3["id"])
    store.delete_collection("other")
    assert all("Borderlands" not in h.text for h in store.search("games", "Borderlands", k=8))


def test_existing_databases_get_a_word_index(tmp_path):
    path = tmp_path / "knowledge.db"
    store = KnowledgeStore(path, BlindEmbedder())
    store.create_collection("games")
    store.add_document("games", "doom.txt", "Doom Eternal: rip and tear.")
    store.add_document("games", "bl3.txt", "Borderlands 3 is a looter shooter.")
    store.db.close()
    old = sqlite3.connect(path)
    old.executescript("DROP TABLE chunk_text; DROP TRIGGER chunks_ai; DROP TRIGGER chunks_ad;")
    old.close()

    store = KnowledgeStore(path, BlindEmbedder())
    assert store.search("games", "Borderlands", k=1)[0].document == "bl3.txt"


def test_documents_are_paged_and_filtered(client):
    client.post("/api/collections", json={"name": "games"})
    upload(
        client,
        "games",
        *[(f"review-{i}.txt", f"Review number {i} of a game.".encode()) for i in range(5)],
        ("Відьмак.md", "Огляд гри «Відьмак 3».".encode()),
    )
    store = client.app.state.knowledge
    store.add_document("games", "Doom page", "Doom Eternal guide.", "https://doom.example/guide")

    def documents(**params) -> dict:
        return client.get("/api/collections/games/documents", params=params).json()

    everything = documents()
    assert everything["total"] == 7 and len(everything["data"]) == 7

    page = documents(limit=3, offset=3, sort="oldest")
    assert page["total"] == 7
    assert [d["name"] for d in page["data"]] == ["review-3.txt", "review-4.txt", "Відьмак.md"]
    assert documents(sort="newest", limit=1)["data"][0]["name"] == "Doom page"

    assert [d["name"] for d in documents(kind="web")["data"]] == ["Doom page"]
    assert documents(kind="file")["total"] == 6
    assert documents(query="ВІДЬМАК")["data"][0]["name"] == "Відьмак.md"
    assert documents(query="doom.example")["total"] == 1
    assert documents(query="review", limit=2)["total"] == 5


def test_parallel_reads_and_writes(tmp_path):
    """A crawl, an upload and the UI use the store from different threads, and a second
    process (python -m engine crawl) may write to the same file."""
    store = KnowledgeStore(tmp_path / "knowledge.db", HashEmbedder())
    other_process = KnowledgeStore(tmp_path / "knowledge.db", HashEmbedder())
    store.create_collection("games")
    websites = WebsiteStore(store)
    errors = []

    def run(work, times):
        for i in range(times):
            try:
                work(i)
            except Exception as e:
                errors.append(repr(e))

    jobs = [
        (lambda i: store.add_document("games", f"a{i}", f"Article {i} " * 50), 60),
        (lambda i: other_process.add_document("games", f"b{i}", f"Page {i} " * 50), 60),
        (lambda i: websites.add_website("games", Website(name=f"s{i}", urls=["http://x"])), 30),
        (lambda i: (store.documents("games", limit=50), store.search("games", "Article")), 60),
    ]
    threads = [threading.Thread(target=run, args=job) for job in jobs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert store.count_documents("games") == 120
    assert len(websites.websites("games")) == 30


def test_locked_database_explains_what_to_do(tmp_path):
    path = tmp_path / "knowledge.db"
    stuck = sqlite3.connect(path)  # an old engine that died in the middle of a write
    stuck.execute("CREATE TABLE t (x)")
    stuck.commit()
    stuck.execute("INSERT INTO t VALUES (1)")
    with pytest.raises(RuntimeError, match="locked by another process"):
        KnowledgeStore(path, HashEmbedder())
    stuck.rollback()
    KnowledgeStore(path, HashEmbedder())


def test_collection_bundled_with_a_model(client, settings, store):
    """A model folder with collection.jsonl brings its collection to another engine."""
    import json

    from engine.models.bundles import import_bundles, write_bundle

    client.post("/api/collections", json={"name": "games"})
    upload(client, "games", ("witcher.txt", WITCHER.encode()))
    folder = settings.models_dir / "hints-q8_0"
    folder.mkdir(parents=True)
    (folder / "hints-q8_0.Q8_0.gguf").write_bytes(b"GGUF")
    assert write_bundle(store, "games", folder / "collection.jsonl") == 1
    (folder / "tinylm.json").write_text(json.dumps({"base": "x", "collection": "games"}))

    # On "another machine": an empty knowledge base gets the collection, chunk for chunk.
    other = KnowledgeStore(settings.data_dir / "other.db", HashEmbedder())
    models = client.app.state.models
    assert import_bundles(models, other) == ["games"]
    assert [c["chunks"] for c in other.collections()] == [c["chunks"] for c in store.collections()]
    assert other.search("games", "Geralt", 1)[0].document == "witcher.txt"
    assert models.settings.get("hints-q8_0") == {"collection": "games"}
    # Done once; a collection of that name is kept, and a changed setting stays.
    models.settings.set("hints-q8_0", ModelSettings(collection=""))
    assert import_bundles(models, other) == []
    assert models.settings.get("hints-q8_0") == {"collection": ""}
