"""Management API for uploaded data: collections, documents and search."""

import json
import logging
import shutil
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from engine.api.deps import knowledge, require_api_key
from engine.rag.convert import UnsupportedFile, convert
from engine.rag.store import (
    CollectionNotFound,
    DocumentKind,
    DocumentSort,
    InvalidName,
    KnowledgeStore,
)
from engine.rag.transfer import ImportManager, export_lines, read_header

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/collections", dependencies=[Depends(require_api_key)], tags=["knowledge"]
)


class NewCollection(BaseModel):
    name: str


class SearchRequest(BaseModel):
    query: str
    k: int = 4


def not_found(name: str) -> HTTPException:
    return HTTPException(404, f"Unknown collection '{name}'")


@router.get("")
def list_collections(store: KnowledgeStore = Depends(knowledge)) -> dict:
    return {"data": store.collections()}


@router.post("", status_code=201)
def create_collection(body: NewCollection, store: KnowledgeStore = Depends(knowledge)) -> dict:
    try:
        collection = store.create_collection(body.name)
        log.info("Collection %s created", body.name)
        return collection
    except InvalidName as e:
        raise HTTPException(422, str(e)) from None


@router.delete("/{name}", status_code=204)
def delete_collection(name: str, store: KnowledgeStore = Depends(knowledge)) -> None:
    try:
        store.delete_collection(name)
        log.info("Collection %s deleted", name)
    except CollectionNotFound:
        raise not_found(name) from None


@router.get("/{name}/documents")
def list_documents(
    name: str,
    query: str = "",
    kind: DocumentKind = "all",
    sort: DocumentSort = "oldest",
    offset: int = Query(0, ge=0),
    limit: int | None = Query(None, ge=1, le=1000),
    store: KnowledgeStore = Depends(knowledge),
) -> dict:
    """One page of documents; total is the number matching the filter, for paging."""
    try:
        return {
            "data": store.documents(name, query, kind, sort, offset, limit),
            "total": store.count_documents(name, query, kind),
        }
    except CollectionNotFound:
        raise not_found(name) from None


@router.post("/{name}/documents")
def upload_documents(
    name: str, files: list[UploadFile], store: KnowledgeStore = Depends(knowledge)
) -> dict:
    """Converts and indexes each file; one bad file does not stop the others.

    The file name may be a relative path (folder uploads); it is kept as the document's source.
    """
    results = []
    for upload in files:
        filename = upload.filename or "upload"
        try:
            doc = convert(filename, upload.file.read())
            title = doc.title or Path(filename).name
            results.append(store.add_document(name, title, doc.text, filename))
        except CollectionNotFound:
            raise not_found(name) from None
        except UnsupportedFile as e:
            results.append({"name": filename, "error": str(e)})
        except Exception as e:  # a broken PDF or DOCX should not fail the whole upload
            results.append({"name": filename, "error": f"Could not read the file: {e}"})
    for r in results:
        if r.get("error"):
            log.warning("Upload to %s: %s: %s", name, r["name"], r["error"])
        elif r.get("duplicate"):
            log.info("Upload to %s: %s is already there", name, r["name"])
        else:
            chunks = r["chunks"]
            log.info(
                "Upload to %s: %s, %d chunk%s", name, r["name"], chunks, "" if chunks == 1 else "s"
            )
    return {"data": results}


@router.delete("/{name}/documents/{document_id}", status_code=204)
def delete_document(name: str, document_id: int, store: KnowledgeStore = Depends(knowledge)):
    if not store.delete_document(name, document_id):
        raise HTTPException(404, f"No document {document_id} in '{name}'")
    log.info("Document %d deleted from %s", document_id, name)


@router.get("/{name}/export")
def export_collection(name: str, store: KnowledgeStore = Depends(knowledge)) -> StreamingResponse:
    """The collection's documents and chunks as JSON lines, for POST /api/collections/import."""
    try:
        store.count_documents(name)
    except CollectionNotFound:
        raise not_found(name) from None
    return StreamingResponse(
        export_lines(store, name),
        media_type="application/x-ndjson",
        # filename* allows names such as "ігри"; headers themselves are latin-1.
        headers={"content-disposition": f"attachment; filename*=UTF-8''{quote(name)}.tinylm.jsonl"},
    )


def imports(request: Request) -> ImportManager:
    return request.app.state.imports


@router.post("/import", status_code=202)
def import_collection(
    request: Request,
    file: UploadFile,
    collection: str = Form(""),
    store: KnowledgeStore = Depends(knowledge),
    manager: ImportManager = Depends(imports),
) -> dict:
    """Starts adding an export's documents to the collection (by default the one it came from),
    made if missing; poll GET /api/collections/imports/{id}."""
    folder = request.app.state.settings.workspace_dir / "imports"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex}.jsonl"
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        name = collection.strip() or read_header(path).get("collection", "")
    except (json.JSONDecodeError, UnicodeDecodeError):
        path.unlink()
        raise HTTPException(422, "Not a collection export: expected JSON lines") from None
    if not name:
        path.unlink()
        raise HTTPException(422, "Name the collection to import into")
    try:
        store.create_collection(name)
    except InvalidName as e:
        path.unlink()
        raise HTTPException(422, str(e)) from None
    return manager.start(name, path).to_dict()


@router.get("/imports/{job_id}")
def import_status(job_id: str, manager: ImportManager = Depends(imports)) -> dict:
    if job_id not in manager.jobs:
        raise HTTPException(404, f"No import {job_id}")
    return manager.jobs[job_id].to_dict()


@router.post("/{name}/search")
def search(name: str, body: SearchRequest, store: KnowledgeStore = Depends(knowledge)) -> dict:
    try:
        return {"data": [hit.to_dict() for hit in store.search(name, body.query, body.k)]}
    except CollectionNotFound:
        raise not_found(name) from None
