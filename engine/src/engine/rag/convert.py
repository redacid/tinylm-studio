"""Turns uploaded files into plain text (Markdown where the format has structure)."""

import io
import re
from dataclasses import dataclass
from pathlib import Path

# Read as-is; everything else goes through markitdown, except web pages (see html_to_text).
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".json", ".jsonl", ".yaml", ".yml", ".log"}
HTML_SUFFIXES = {".html", ".htm"}
MARKITDOWN_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".csv", ".xml"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | HTML_SUFFIXES | MARKITDOWN_SUFFIXES

IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


class UnsupportedFile(Exception):
    pass


@dataclass
class Converted:
    text: str
    # The document's own title (a web page's <title>), when it has one.
    title: str | None = None


def strip_markup(markdown: str) -> str:
    """Drops images and keeps only link text: URLs are noise for search and for the model."""
    return LINK.sub(r"\1", IMAGE.sub("", markdown)).strip()


def html_to_text(html: str) -> Converted:
    """Keeps a page's main content and drops menus, sidebars, footers and comments."""
    import trafilatura

    text = trafilatura.extract(
        html,
        output_format="markdown",
        include_links=False,
        include_images=False,
        include_tables=True,
        favor_recall=True,
    )
    metadata = trafilatura.extract_metadata(html)
    title = metadata.title if metadata else None
    if not text:
        text = strip_markup(markitdown(html.encode(), ".html"))
    return Converted(text.strip(), title)


def markitdown(data: bytes, suffix: str) -> str:
    from markitdown import MarkItDown, StreamInfo

    result = MarkItDown().convert_stream(io.BytesIO(data), stream_info=StreamInfo(extension=suffix))
    return result.markdown


def convert(filename: str, data: bytes) -> Converted:
    suffix = Path(filename).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return Converted(data.decode("utf-8", errors="replace").strip())
    if suffix in HTML_SUFFIXES:
        return html_to_text(data.decode("utf-8", errors="replace"))
    if suffix in MARKITDOWN_SUFFIXES:
        return Converted(strip_markup(markitdown(data, suffix)))
    raise UnsupportedFile(f"Unsupported file type: {suffix or filename}")
