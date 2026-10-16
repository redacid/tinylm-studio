"""Reading training examples from JSONL and CSV files.

Every example becomes a list of chat messages ({"role", "content"}) that ends with the
assistant's answer, the part the model learns to write. Accepted shapes of a record:

- {"messages": [{"role": "user", "content": ...}, {"role": "assistant", ...}]}  (OpenAI chat)
- {"conversations": [{"from": "human", "value": ...}, {"from": "gpt", ...}]}  (ShareGPT)
- {"instruction": ..., "input": ..., "output": ...}  (Alpaca)
- {"prompt": ..., "completion": ...} or {"question": ..., "answer": ...}

A "system" field adds a system message. CSV files use the same names as column headers.
"""

import csv
import io
import json
from collections.abc import Iterator
from dataclasses import dataclass

Messages = list[dict[str, str]]
ROLES = ("system", "user", "assistant")
SHAREGPT_ROLES = {"system": "system", "human": "user", "user": "user", "gpt": "assistant"}
SHAREGPT_ROLES |= {"assistant": "assistant", "model": "assistant", "bot": "assistant"}
# (question field, answer field), tried in order.
PAIRS = (("prompt", "completion"), ("question", "answer"), ("input", "output"))
SUPPORTED_SUFFIXES = (".jsonl", ".json", ".csv")
ORIGINS = ("manual", "imported", "generated")


class InvalidExample(ValueError):
    pass


@dataclass
class ParsedRecord:
    line: int  # 1-based line (JSONL) or row (CSV, counting the header) in the file
    messages: Messages | None = None
    error: str | None = None
    # Kept from a dataset export, so examples moved to another engine keep where they came from.
    origin: str | None = None
    source: str | None = None


def read(number: int, record: object) -> ParsedRecord:
    """One record of a file as messages, with its origin and source when an export wrote them."""
    parsed = ParsedRecord(number, messages=to_messages(record))
    if isinstance(record, dict) and record.get("origin") in ORIGINS:
        parsed.origin = record["origin"]
        source = record.get("source")
        parsed.source = source if isinstance(source, str) and source else None
    return parsed


def validate(messages: object) -> Messages:
    """Checks a message list and returns it cleaned up: only role and content, text trimmed."""
    if not isinstance(messages, list) or not messages:
        raise InvalidExample("messages must be a non-empty list")
    cleaned: Messages = []
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise InvalidExample(f"message {i + 1} is not an object")
        role, content = message.get("role"), message.get("content")
        if role not in ROLES:
            raise InvalidExample(
                f"message {i + 1} has role {role!r}; use system, user or assistant"
            )
        if not isinstance(content, str) or not content.strip():
            raise InvalidExample(f"message {i + 1} ({role}) has no text")
        if role == "system" and i > 0:
            raise InvalidExample("a system message can only come first")
        cleaned.append({"role": role, "content": content.strip()})
    if not any(m["role"] == "user" for m in cleaned):
        raise InvalidExample("there is no user message")
    if cleaned[-1]["role"] != "assistant":
        raise InvalidExample("the last message must be the assistant's answer")
    return cleaned


def to_messages(record: object) -> Messages:
    """Turns one record of any accepted shape into validated chat messages."""
    if not isinstance(record, dict):
        raise InvalidExample("each line must be a JSON object")
    if "messages" in record:
        return validate(record["messages"])
    if "conversations" in record:
        turns = record["conversations"]
        if not isinstance(turns, list):
            raise InvalidExample("conversations must be a list")
        messages = []
        for turn in turns:
            if not isinstance(turn, dict) or turn.get("from") not in SHAREGPT_ROLES:
                raise InvalidExample(f"unknown speaker in conversations: {turn!r:.60}")
            messages.append({"role": SHAREGPT_ROLES[turn["from"]], "content": turn.get("value")})
        return validate(messages)

    def text(field: str) -> str:
        value = record.get(field)
        return value.strip() if isinstance(value, str) else ""

    messages = [{"role": "system", "content": text("system")}] if text("system") else []
    if "instruction" in record:
        question = text("instruction")
        if text("input"):
            question += "\n\n" + text("input")
        return validate([*messages, {"role": "user", "content": question}, answer(record, text)])
    for question, reply in PAIRS:
        if question in record and reply in record:
            user = {"role": "user", "content": text(question)}
            return validate([*messages, user, {"role": "assistant", "content": text(reply)}])
    fields = ", ".join(sorted(record)) or "none"
    raise InvalidExample(
        f"unknown format (fields: {fields}); use messages, conversations, "
        "instruction/output, prompt/completion or question/answer"
    )


def answer(record: dict, text) -> dict:
    for field in ("output", "response", "answer", "completion"):
        if text(field):
            return {"role": "assistant", "content": text(field)}
    raise InvalidExample("there is no output (the answer)")


def parse(filename: str, data: bytes) -> Iterator[ParsedRecord]:
    """Reads a file of examples; a bad record is reported and does not stop the others."""
    name = filename.lower()
    if not name.endswith(SUPPORTED_SUFFIXES):
        raise InvalidExample(f"{filename}: use a .jsonl, .json or .csv file")
    text = data.decode("utf-8-sig", errors="replace")
    if name.endswith(".csv"):
        yield from parse_csv(text)
    elif name.endswith(".json") and text.lstrip().startswith("["):
        yield from parse_json_array(text)
    else:
        yield from parse_jsonl(text)


def parse_jsonl(text: str) -> Iterator[ParsedRecord]:
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            yield read(number, json.loads(line))
        except json.JSONDecodeError as e:
            yield ParsedRecord(number, error=f"not valid JSON: {e.msg}")
        except InvalidExample as e:
            yield ParsedRecord(number, error=str(e))


def parse_json_array(text: str) -> Iterator[ParsedRecord]:
    try:
        records = json.loads(text)
    except json.JSONDecodeError as e:
        yield ParsedRecord(e.lineno, error=f"not valid JSON: {e.msg}")
        return
    for number, record in enumerate(records, 1):
        try:
            yield read(number, record)
        except InvalidExample as e:
            yield ParsedRecord(number, error=str(e))


def parse_csv(text: str) -> Iterator[ParsedRecord]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames:
        reader.fieldnames = [f.strip().lower() for f in reader.fieldnames]
    for number, row in enumerate(reader, 2):
        try:
            yield read(number, row)
        except InvalidExample as e:
            yield ParsedRecord(number, error=str(e))
