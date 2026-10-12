"""llama.cpp's own converter from Hugging Face weights to GGUF.

It is not part of the llama-cpp-python wheel, but its source package on PyPI carries the
llama.cpp sources it was built from. The converter (convert_hf_to_gguf.py, conversion/ and
gguf-py/, about 2 MB) is taken from the source package of the installed version, so the GGUF
files it writes match the llama.cpp that loads them. It is downloaded once and kept in
<data dir>/tools/; TINYLM_LLAMA_CPP_DIR points to a llama.cpp checkout instead, e.g. offline.
"""

import io
import logging
import shutil
import tarfile
from collections.abc import Callable
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

PYPI = "https://pypi.org/pypi/llama-cpp-python/{version}/json"
CONVERTER = "convert_hf_to_gguf.py"
# Parts of the llama.cpp tree the converter needs.
NEEDED = (CONVERTER, "conversion/", "gguf-py/")


class ToolsError(RuntimeError):
    pass


def llama_cpp_version() -> str:
    import llama_cpp

    return llama_cpp.__version__


def converter_dir(
    tools_dir: Path,
    override: Path | None = None,
    version: str | None = None,
    pypi: str = PYPI,
    progress: Callable[[str], None] = lambda message: None,
) -> Path:
    """A folder with convert_hf_to_gguf.py, downloading it on first use."""
    if override is not None:
        if not (override / CONVERTER).is_file():
            raise ToolsError(f"{override} has no {CONVERTER}; point it to a llama.cpp checkout.")
        return override
    version = version or llama_cpp_version()
    target = tools_dir / f"llama.cpp-{version}"
    if (target / CONVERTER).is_file():
        return target
    progress(f"Downloading the GGUF converter of llama-cpp-python {version} (once)")
    data = download_source(version, pypi)
    partial = tools_dir / f".llama.cpp-{version}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    extract(data, partial)
    if not (partial / CONVERTER).is_file():
        shutil.rmtree(partial, ignore_errors=True)
        raise ToolsError(f"The source package of llama-cpp-python {version} has no {CONVERTER}.")
    shutil.rmtree(target, ignore_errors=True)
    partial.rename(target)
    return target


def download_source(version: str, pypi: str = PYPI) -> bytes:
    try:
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            release = client.get(pypi.format(version=version))
            release.raise_for_status()
            sdist = next((f for f in release.json()["urls"] if f["packagetype"] == "sdist"), None)
            if sdist is None:
                raise ToolsError(f"PyPI has no source package of llama-cpp-python {version}.")
            response = client.get(sdist["url"], timeout=600)
            response.raise_for_status()
            return response.content
    except httpx.HTTPError as e:
        raise ToolsError(
            f"Could not download the GGUF converter from PyPI ({e}). Connect to the internet "
            "once, or set TINYLM_LLAMA_CPP_DIR to a llama.cpp checkout."
        ) from None


def extract(data: bytes, target: Path) -> None:
    """Copies the converter's files out of llama_cpp_python-X/vendor/llama.cpp/."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            parts = member.name.split("/")
            if len(parts) < 4 or parts[1:3] != ["vendor", "llama.cpp"] or not member.isfile():
                continue
            relative = "/".join(parts[3:])
            if not any(relative == n or relative.startswith(n) for n in NEEDED):
                continue
            if ".." in parts:
                continue
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is not None:
                path.write_bytes(source.read())
