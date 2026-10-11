"""Splits text into retrieval-sized chunks along paragraph and sentence boundaries."""

import re

# ~250-350 tokens for uk/ru/en text: well within the embedding model's 512-token limit.
MAX_CHARS = 1200

SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def chunk_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
            continue
        for sentence in SENTENCE_END.split(paragraph):
            pieces += [sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars)]

    chunks: list[str] = []
    current = ""
    for piece in filter(None, pieces):
        if current and len(current) + len(piece) + 2 > max_chars:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks
