"""Management API for training datasets: import, browse, edit and split examples."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from engine.api.deps import require_api_key
from engine.datasets.formats import InvalidExample
from engine.datasets.generate import LANGUAGE_NAMES, GenerationManager, GenerationOptions
from engine.datasets.store import DatasetNotFound, DatasetStore, InvalidDataset, Origin, Split
from engine.rag.store import CollectionNotFound
from engine.teachers.store import TeacherNotFound

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


class GenerateRequest(BaseModel):
    collection: str
    teacher: str  # "ref" from GET /api/teachers: a saved teacher's id or "local:<model id>"
    chunks: int = Field(20, ge=1, le=10000)
    per_chunk: int = Field(3, ge=1, le=20)
    languages: list[str] = []
    instructions: str = ""
    system_prompt: str = ""
    unanswerable: bool = True
    verify: bool = True
    # Put sources found in the collection into each example's system prompt, as the chat does
    # with data: the model learns to answer from given sources.
    with_sources: bool = False
    sources: int = Field(4, ge=1, le=20)
    min_answer_chars: int = Field(20, ge=1)
    max_answer_chars: int = Field(2000, ge=1)


def datasets(request: Request) -> DatasetStore:
    return request.app.state.datasets


def generations(request: Request) -> GenerationManager:
    return request.app.state.generations


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
    origin: Origin | None = None,
    offset: int = Query(0, ge=0),
    limit: int | None = Query(None, ge=1, le=1000),
    store: DatasetStore = Depends(datasets),
) -> dict:
    try:
        return {
            "data": store.examples(name, query, split, offset, limit, origin),
            "total": store.count(name, query, split, origin),
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


@router.post("/{name}/generate", status_code=202)
def generate(
    name: str,
    body: GenerateRequest,
    request: Request,
    store: DatasetStore = Depends(datasets),
    manager: GenerationManager = Depends(generations),
) -> dict:
    """Starts writing examples from a collection's chunks with a teacher model."""
    try:
        store.dataset(name)
        request.app.state.knowledge._require(body.collection)
        request.app.state.teachers.teacher(body.teacher)
    except DatasetNotFound:
        raise not_found(name) from None
    except CollectionNotFound:
        raise HTTPException(404, f"Unknown collection '{body.collection}'") from None
    except TeacherNotFound:
        raise HTTPException(404, f"Unknown teacher '{body.teacher}'") from None
    unknown = [code for code in body.languages if code not in LANGUAGE_NAMES]
    if unknown:
        raise HTTPException(422, f"Unknown language {unknown[0]}; use uk, ru or en")
    if manager.running():
        raise HTTPException(409, "Another generation is running; one runs at a time.")
    return manager.start(name, GenerationOptions(**body.model_dump())).to_dict()


@router.get("/{name}/generate")
def generation_source(
    name: str, collection: str, manager: GenerationManager = Depends(generations)
) -> dict:
    """How many chunks of a collection this dataset has not been generated from yet."""
    try:
        return {"unused_chunks": manager.generator.count_unused(name, collection)}
    except CollectionNotFound:
        raise HTTPException(404, f"Unknown collection '{collection}'") from None
