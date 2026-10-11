"""Uploaded data: conversion, chunking, collections, search and grounded chat answers."""

import anthropic
import openai
import pytest

from engine.chat import ChatRequest, Message
from engine.rag.chunking import chunk_text
from engine.rag.convert import UnsupportedFile, convert, strip_markup
from engine.rag.prompt import ground
from engine.rag.store import Hit

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
