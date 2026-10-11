"""Turns uploaded files into plain text (Markdown where the format has structure)."""

import io
from pathlib import Path

# Read as-is; everything else goes through markitdown.
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".json", ".jsonl", ".yaml", ".yml", ".log"}
MARKITDOWN_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".html", ".htm", ".csv", ".xml"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | MARKITDOWN_SUFFIXES


class UnsupportedFile(Exception):
    pass


def to_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return data.decode("utf-8", errors="replace").strip()
    if suffix not in MARKITDOWN_SUFFIXES:
        raise UnsupportedFile(f"Unsupported file type: {suffix or filename}")
    from markitdown import MarkItDown, StreamInfo

    result = MarkItDown().convert_stream(
        io.BytesIO(data), stream_info=StreamInfo(extension=suffix, filename=filename)
    )
    return result.markdown.strip()
