"""Management API for uploaded data: collections, documents and search."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
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
        return store.create_collection(body.name)
    except InvalidName as e:
        raise HTTPException(422, str(e)) from None


@router.delete("/{name}", status_code=204)
def delete_collection(name: str, store: KnowledgeStore = Depends(knowledge)) -> None:
    try:
        store.delete_collection(name)
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
    return {"data": results}


@router.delete("/{name}/documents/{document_id}", status_code=204)
def delete_document(name: str, document_id: int, store: KnowledgeStore = Depends(knowledge)):
    if not store.delete_document(name, document_id):
        raise HTTPException(404, f"No document {document_id} in '{name}'")


@router.post("/{name}/search")
def search(name: str, body: SearchRequest, store: KnowledgeStore = Depends(knowledge)) -> dict:
    try:
        return {"data": [hit.to_dict() for hit in store.search(name, body.query, body.k)]}
    except CollectionNotFound:
        raise not_found(name) from None
