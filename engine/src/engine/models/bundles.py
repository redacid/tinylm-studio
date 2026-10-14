"""A collection exported together with a model.

An export can take a collection along: its documents, already split into chunks, go into the
new model folder as collection.jsonl and tinylm.json names the collection. When the engine
starts (or an export finishes) and finds such a folder, it creates the collection if there is
none of that name, indexes the chunks with its own embedding model, and makes it the model's
collection unless the model's settings already say otherwise. So a model folder copied to
another machine answers from its data there too.
"""

import json
import logging
import threading
from pathlib import Path

from engine.models.manager import ModelManager
from engine.models.settings import ModelSettings
from engine.rag.store import KnowledgeStore

log = logging.getLogger(__name__)

BUNDLE_FILE = "collection.jsonl"


def write_bundle(knowledge: KnowledgeStore, collection: str, path: Path) -> int:
    """Writes the collection's documents, one JSON line each; returns how many."""
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for document in knowledge.export_documents(collection):
            f.write(json.dumps(document, ensure_ascii=False) + "\n")
            count += 1
    return count


def bundled_collection(folder: Path) -> str | None:
    info = folder / "tinylm.json"
    if not (folder / BUNDLE_FILE).is_file() or not info.is_file():
        return None
    return json.loads(info.read_text()).get("collection")


def import_bundles(models: ModelManager, knowledge: KnowledgeStore) -> list[str]:
    """Imports the collections of model folders that have one; returns the imported names."""
    imported = []
    existing = {c["name"] for c in knowledge.collections()}
    for folder in models.models_dir.iterdir():
        if folder.name.startswith(".") or not (name := bundled_collection(folder)):
            continue
        if name not in existing:
            log.info("Importing collection %s bundled with model %s", name, folder.name)
            knowledge.create_collection(name)
            with (folder / BUNDLE_FILE).open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        knowledge.add_chunks(name, d["name"], d.get("source", ""), d["chunks"])
            existing.add(name)
            imported.append(name)
        settings = models.settings.get(folder.name)
        if "collection" not in settings:
            models.settings.set(folder.name, ModelSettings(**settings, collection=name))
    return imported


def import_bundles_in_background(models: ModelManager, knowledge: KnowledgeStore) -> None:
    def run() -> None:
        try:
            import_bundles(models, knowledge)
        except Exception:  # a broken bundle must not stop the engine
            log.exception("Importing bundled collections failed")

    threading.Thread(target=run, daemon=True).start()
