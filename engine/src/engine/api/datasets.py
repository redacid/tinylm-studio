"""Management API for training datasets: import, browse, edit and split examples."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from engine.api.deps import require_api_key
from engine.datasets.formats import InvalidExample
from engine.datasets.store import DatasetNotFound, DatasetStore, InvalidDataset, Split

router = APIRouter(
    prefix="/api/datasets", dependencies=[Depends(require_api_key)], tags=["datasets"]
)


class NewDataset(BaseModel):
    name: str
    # Share of examples held out to measure the model on data it was not trained on.
    eval_percent: int = Field(10, ge=0, le=50)


class DatasetChange(BaseModel):
    eval_percent: int = Field(ge=0, le=50)


class Message(BaseModel):
    role: str
    content: str


class Example(BaseModel):
    messages: list[Message]


def datasets(request: Request) -> DatasetStore:
    return request.app.state.datasets


def not_found(name: str) -> HTTPException:
    return HTTPException(404, f"Unknown dataset '{name}'")


def messages(body: Example) -> list[dict]:
    return [m.model_dump() for m in body.messages]


@router.get("")
def list_datasets(store: DatasetStore = Depends(datasets)) -> dict:
    return {"data": store.datasets()}


@router.post("", status_code=201)
def create_dataset(body: NewDataset, store: DatasetStore = Depends(datasets)) -> dict:
    try:
        return store.create(body.name, body.eval_percent)
    except InvalidDataset as e:
        raise HTTPException(422, str(e)) from None


@router.patch("/{name}")
def change_dataset(name: str, body: DatasetChange, store: DatasetStore = Depends(datasets)):
    try:
        return store.update(name, body.eval_percent)
    except DatasetNotFound:
        raise not_found(name) from None


@router.delete("/{name}", status_code=204)
def delete_dataset(name: str, store: DatasetStore = Depends(datasets)) -> None:
    try:
        store.delete(name)
    except DatasetNotFound:
        raise not_found(name) from None


@router.post("/{name}/import")
def import_files(name: str, files: list[UploadFile], store: DatasetStore = Depends(datasets)):
    """Adds the examples of JSONL, JSON or CSV files; bad records are reported by line."""
    results = []
    for upload in files:
        filename = upload.filename or "upload.jsonl"
        try:
            results.append(store.import_file(name, filename, upload.file.read()))
        except DatasetNotFound:
            raise not_found(name) from None
        except InvalidExample as e:
            results.append({"name": filename, "error": str(e)})
    return {"data": results}


@router.get("/{name}/examples")
def list_examples(
    name: str,
    query: str = "",
    split: Split = "all",
    offset: int = Query(0, ge=0),
    limit: int | None = Query(None, ge=1, le=1000),
    store: DatasetStore = Depends(datasets),
) -> dict:
    try:
        return {
            "data": store.examples(name, query, split, offset, limit),
            "total": store.count(name, query, split),
        }
    except DatasetNotFound:
        raise not_found(name) from None


@router.post("/{name}/examples", status_code=201)
def add_example(name: str, body: Example, store: DatasetStore = Depends(datasets)) -> dict:
    try:
        return store.add(name, messages(body))
    except DatasetNotFound:
        raise not_found(name) from None
    except InvalidExample as e:
        raise HTTPException(422, str(e)) from None


@router.put("/{name}/examples/{example_id}", status_code=204)
def edit_example(name: str, example_id: int, body: Example, store=Depends(datasets)) -> None:
    try:
        if not store.edit(name, example_id, messages(body)):
            raise HTTPException(404, f"No example {example_id} in '{name}'")
    except (InvalidExample, InvalidDataset) as e:
        raise HTTPException(422, str(e)) from None


@router.delete("/{name}/examples/{example_id}", status_code=204)
def delete_example(name: str, example_id: int, store: DatasetStore = Depends(datasets)) -> None:
    if not store.remove(name, example_id):
        raise HTTPException(404, f"No example {example_id} in '{name}'")
