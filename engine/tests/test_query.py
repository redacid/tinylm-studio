"""SQL queries over the data (the SQL tab)."""

from tests.test_knowledge import upload


def sql(client, query: str, limit: int | None = None):
    body = {"sql": query} if limit is None else {"sql": query, "limit": limit}
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
        "DELETE FROM documents",
        "UPDATE documents SET name = 'x'",
        "DROP TABLE documents",
        "PRAGMA table_info(teachers)",
        "ATTACH DATABASE 'other.db' AS other",
        "SELECT 1; DELETE FROM documents",
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
