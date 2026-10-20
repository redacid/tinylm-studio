"""The OpenAPI schema at /openapi.json, which `python -m engine openapi` writes and the desktop
app's types are made from."""

import json
import subprocess
import sys


def test_schema_describes_responses_and_the_api_key(client):
    schema = client.get("/openapi.json").json()
    datasets = schema["paths"]["/api/datasets"]["get"]["responses"]
    assert datasets["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/DatasetList"
    }
    assert datasets["401"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/OpenAIError"
    }
    missing = schema["paths"]["/api/datasets/{name}"]["delete"]["responses"]["404"]
    assert missing["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/Error"
    }
    assert set(schema["components"]["securitySchemes"]) == {"HTTPBearer", "APIKeyHeader"}
    # /health answers without the key.
    assert "security" not in schema["paths"]["/health"]["get"]
    assert schema["paths"]["/api/datasets"]["get"]["security"]


def test_every_json_endpoint_has_a_response_schema(client):
    schema = client.get("/openapi.json").json()
    untyped = []
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            for status, response in operation["responses"].items():
                content = response.get("content", {}).get("application/json")
                if status.startswith("2") and content is not None and content["schema"] == {}:
                    untyped.append(f"{method.upper()} {path}")
    assert untyped == []


def test_command_writes_the_schema(tmp_path):
    output = tmp_path / "openapi.json"
    subprocess.run([sys.executable, "-m", "engine", "openapi", "--output", str(output)], check=True)
    assert json.loads(output.read_text())["info"]["title"] == "tinylm-studio engine"
