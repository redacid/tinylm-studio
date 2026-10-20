"""Management API for training datasets: import, browse, edit and split examples."""

from urllib.parse import quote

from fastapi import Depends, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from engine.api.deps import require_api_key
from engine.api.schemas import (
    UNAUTHORIZED,
    AddedExample,
    Dataset,
    Example,
    GenerationJob,
    ImportResult,
    Router,
    UnusedChunks,
    errors,
    file_response,
    items,
    page,
)
from engine.datasets.formats import InvalidExample
from engine.datasets.generate import LANGUAGE_NAMES, GenerationManager, GenerationOptions
from engine.datasets.store import (
    DatasetNotFound,
    DatasetStore,
    ExampleSort,
    InvalidDataset,
    Origin,
    Split,
)
from engine.rag.store import CollectionNotFound
from engine.teachers.store import TeacherNotFound

router = Router(
    prefix="/api/datasets",
    dependencies=[Depends(require_api_key)],
    tags=["datasets"],
    responses=UNAUTHORIZED,
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


class ExampleMessages(BaseModel):
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


def messages(body: ExampleMessages) -> list[dict]:
    return [m.model_dump() for m in body.messages]


@router.get("", response_model=items(Dataset))
def list_datasets(store: DatasetStore = Depends(datasets)) -> dict:
    return {"data": store.datasets()}


@router.post("", status_code=201, response_model=Dataset, responses=errors(422))
def create_dataset(body: NewDataset, store: DatasetStore = Depends(datasets)) -> dict:
    try:
        return store.create(body.name, body.eval_percent)
    except InvalidDataset as e:
        raise HTTPException(422, str(e)) from None


@router.patch("/{name}", response_model=Dataset, responses=errors(404))
def change_dataset(name: str, body: DatasetChange, store: DatasetStore = Depends(datasets)):
    try:
        return store.update(name, body.eval_percent)
    except DatasetNotFound:
        raise not_found(name) from None


@router.delete("/{name}", status_code=204, responses=errors(404))
def delete_dataset(name: str, store: DatasetStore = Depends(datasets)) -> None:
    try:
        store.delete(name)
    except DatasetNotFound:
        raise not_found(name) from None


@router.post("/{name}/import", response_model=items(ImportResult), responses=errors(404))
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


@router.get(
    "/{name}/export",
    response_class=StreamingResponse,
    responses={**file_response("application/x-ndjson", "JSON lines"), **errors(404)},
)
def export_dataset(
    name: str, split: Split = "all", store: DatasetStore = Depends(datasets)
) -> StreamingResponse:
    """The examples as JSONL in the chat format, for POST /{name}/import here or elsewhere."""
    try:
        store.dataset(name)
    except DatasetNotFound:
        raise not_found(name) from None
    filename = name if split == "all" else f"{name}-{split}"
    return StreamingResponse(
        store.export_lines(name, split),
        media_type="application/x-ndjson",
        # filename* allows names such as "підказки"; headers themselves are latin-1.
        headers={"content-disposition": f"attachment; filename*=UTF-8''{quote(filename)}.jsonl"},
    )


@router.get("/{name}/examples", response_model=page(Example), responses=errors(404))
def list_examples(
    name: str,
    query: str = "",
    split: Split = "all",
    origin: Origin | None = None,
    sort: ExampleSort = "oldest",
    offset: int = Query(0, ge=0),
    limit: int | None = Query(None, ge=1, le=1000),
    store: DatasetStore = Depends(datasets),
) -> dict:
    try:
        return {
            "data": store.examples(name, query, split, offset, limit, origin, sort),
            "total": store.count(name, query, split, origin),
        }
    except DatasetNotFound:
        raise not_found(name) from None


@router.post(
    "/{name}/examples", status_code=201, response_model=AddedExample, responses=errors(404, 422)
)
def add_example(name: str, body: ExampleMessages, store: DatasetStore = Depends(datasets)) -> dict:
    try:
        return store.add(name, messages(body))
    except DatasetNotFound:
        raise not_found(name) from None
    except InvalidExample as e:
        raise HTTPException(422, str(e)) from None


@router.put("/{name}/examples/{example_id}", status_code=204, responses=errors(404, 422))
def edit_example(
    name: str, example_id: int, body: ExampleMessages, store=Depends(datasets)
) -> None:
    try:
        if not store.edit(name, example_id, messages(body)):
            raise HTTPException(404, f"No example {example_id} in '{name}'")
    except (InvalidExample, InvalidDataset) as e:
        raise HTTPException(422, str(e)) from None


@router.delete("/{name}/examples/{example_id}", status_code=204, responses=errors(404))
def delete_example(name: str, example_id: int, store: DatasetStore = Depends(datasets)) -> None:
    if not store.remove(name, example_id):
        raise HTTPException(404, f"No example {example_id} in '{name}'")


@router.post(
    "/{name}/generate",
    status_code=202,
    response_model=GenerationJob,
    responses=errors(404, 409, 422),
)
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


@router.get("/{name}/generate", response_model=UnusedChunks, responses=errors(404))
def generation_source(
    name: str, collection: str, manager: GenerationManager = Depends(generations)
) -> dict:
    """How many chunks of a collection this dataset has not been generated from yet."""
    try:
        return {"unused_chunks": manager.generator.count_unused(name, collection)}
    except CollectionNotFound:
        raise HTTPException(404, f"Unknown collection '{collection}'") from None
