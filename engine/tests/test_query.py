"""SQL queries over the data (the SQL tab)."""

from tests.test_knowledge import upload


def sql(client, query: str, limit: int | None = None, confirm: bool = False):
    body = {"sql": query, "confirm": confirm}
    if limit is not None:
        body["limit"] = limit
    return client.post("/api/query", json=body)


def test_schema_lists_tables_columns_and_examples(client):
    schema = client.get("/api/query/schema").json()
    tables = {t["name"]: t for t in schema["tables"]}
    assert {"collections", "documents", "chunks", "chunk_text"} <= set(tables)
    assert "teachers" not in tables
    chars = next(c for c in tables["documents"]["columns"] if c["name"] == "chars")
    assert chars["type"] == "INTEGER" and chars["not_null"] and chars["about"]
    assert next(c for c in tables["documents"]["columns"] if c["name"] == "id")["primary_key"]
    assert len(schema["examples"]) >= 5


def test_runs_selects_and_full_text_search(client):
    client.post("/api/collections", json={"name": "games"})
    upload(client, "games", ("witcher.md", b"# The Witcher\n\nGeralt hunts monsters."))
    upload(client, "games", ("souls.md", b"# Souls\n\nDark Souls has hard bosses."))
    result = sql(client, "SELECT collection, count(*) AS n FROM documents GROUP BY collection;")
    assert result.status_code == 200, result.text
    assert result.json()["columns"] == ["collection", "n"]
    assert result.json()["rows"] == [["games", 2]]
    found = sql(
        client,
        "SELECT d.name, snippet(chunk_text, 0, '[', ']', '…', 5) FROM chunk_text "
        "JOIN chunks c ON c.id = chunk_text.rowid JOIN documents d ON d.id = c.document_id "
        "WHERE chunk_text MATCH 'geralt' ORDER BY bm25(chunk_text)",
    ).json()
    assert found["rows"][0][0] == "witcher.md" and "[Geralt]" in found["rows"][0][1]
    cased = sql(client, "SELECT count(*) FROM chunks WHERE casefold(text) LIKE '%dark souls%'")
    assert cased.json()["rows"] == [[1]]


def test_examples_run(client):
    client.post("/api/collections", json={"name": "games"})
    upload(client, "games", ("witcher.md", b"# The Witcher\n\nGeralt hunts monsters."))
    for example in client.get("/api/query/schema").json()["examples"]:
        assert sql(client, example["sql"]).status_code == 200, example["title"]


def test_limits_rows(client):
    result = sql(
        client,
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n "
        "WHERE i < 50) SELECT i FROM n",
        limit=10,
    ).json()
    assert len(result["rows"]) == 10 and result["truncated"]
    assert not sql(client, "SELECT 1").json()["truncated"]


def test_refuses_anything_but_reading_the_data(client):
    client.post(
        "/api/teachers",
        json={"name": "t", "kind": "openai", "model": "m", "api_key": "sk-secret"},
    )
    for query in [
        "SELECT * FROM teachers",
        "SELECT api_key FROM teachers",
        "SELECT * FROM sqlite_master",
        "INSERT INTO collections VALUES ('x', 'y')",
        "DELETE FROM teachers",
        "DELETE FROM chunk_text",
        "DELETE FROM documents WHERE id IN (SELECT rowid FROM teachers)",
        "DELETE FROM documents WHERE 1; DELETE FROM collections",
        "DELETE FROM documents RETURNING *",
        "UPDATE documents SET name = 'x'",
        "DROP TABLE documents",
        "PRAGMA table_info(teachers)",
        "ATTACH DATABASE 'other.db' AS other",
        "SELECT 1; DELETE FROM documents",
        "WITH x AS (SELECT 1) DELETE FROM documents",
        "",
        "SELEC 1",
    ]:
        result = sql(client, query)
        assert result.status_code == 400, query
        assert "sk-secret" not in result.text
    assert sql(client, "SELECT count(*) FROM collections").status_code == 200
    assert sql(client, "SELECT * FROM teachers").json()["detail"].startswith("Not allowed")


def test_stops_slow_queries(client, monkeypatch):
    monkeypatch.setattr("engine.rag.query.TIME_LIMIT", 0.2)
    result = sql(
        client,
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT count(*) FROM n",
    )
    assert result.status_code == 400 and "stopped" in result.json()["detail"]


def test_deletes_after_confirming(client, store):
    client.post("/api/collections", json={"name": "games"})
    client.post("/api/collections", json={"name": "other"})
    upload(client, "games", ("witcher.md", b"# The Witcher\n\nGeralt hunts monsters."))
    upload(client, "games", ("souls.md", b"# Souls\n\nDark Souls has hard bosses."))
    upload(client, "other", ("notes.md", b"# Notes\n\nSomething about Geralt."))
    query = "DELETE FROM documents d WHERE d.collection = 'games' AND d.name LIKE 'witcher%';"
    preview = sql(client, query).json()
    assert preview["delete"] == {"table": "documents", "rows": 1, "done": False}
    assert sql(client, "SELECT count(*) FROM documents").json()["rows"] == [[3]]

    done = sql(client, query, confirm=True).json()
    assert done["delete"]["done"] and done["rows"] == [["documents", 1]]
    names = sql(client, "SELECT name FROM documents ORDER BY name").json()["rows"]
    assert names == [["notes.md"], ["souls.md"]]
    # Its chunks, their full-text entries and vectors went too.
    assert {h.document for h in store.search("games", "Geralt hunts monsters")} == {"souls.md"}
    found = sql(client, "SELECT count(*) FROM chunk_text WHERE chunk_text MATCH 'geralt'")
    assert found.json()["rows"] == [[1]]
    orphans = "SELECT count(*) FROM chunks WHERE document_id NOT IN (SELECT id FROM documents)"
    assert sql(client, orphans).json()["rows"] == [[0]]

    # A collection takes its documents with it.
    assert (
        sql(client, "DELETE FROM collections WHERE name = 'other'", confirm=True).status_code == 200
    )
    assert client.get("/api/collections/other/documents").status_code == 404
    assert sql(client, "SELECT count(*) FROM documents").json()["rows"] == [[1]]
    vectors = store.db.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0]
    chunks = store.db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    assert vectors == chunks


def test_delete_by_address_with_double_quotes(client):
    client.post("/api/collections", json={"name": "games"})
    upload(client, "games", ("witcher.md", b"# The Witcher\n\nGeralt hunts monsters."))
    client.app.state.knowledge.db.execute(
        "UPDATE documents SET source = 'https://prozorro.sale/planning/1'"
    )
    client.app.state.knowledge.db.commit()
    query = 'DELETE\nFROM documents\nwhere source LIKE "%https://prozorro.sale/planning/%"'
    assert sql(client, query).json()["delete"]["rows"] == 1
    assert sql(client, query, confirm=True).json()["rows"] == [["documents", 1]]
