"""Management API used by the desktop app: installed models, their settings, the catalog,
searching Hugging Face and downloads."""

from typing import Literal
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from engine.api.deps import knowledge, manager, require_api_key
from engine.models import hub
from engine.models.catalog import DEFAULT_MODEL
from engine.models.manager import ModelExists, ModelManager, ModelNotFound
from engine.models.settings import ModelSettings
from engine.models.transfer import ModelUploads, Upload, UploadError, archive
from engine.rag.store import KnowledgeStore

router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)], tags=["admin"])


@router.get("/models")
def model_catalog(models: ModelManager = Depends(manager)) -> dict:
    return {
        "installed": models.installed(),
        "models": models.details(),
        "downloads": [d.to_dict() for d in models.downloads.values()],
        "trainable": models.trainable(),
        "catalog": models.catalog(),
        "default": DEFAULT_MODEL,
        "hf_token": models.has_token(),
    }


@router.post("/models/{model_id}/download", status_code=202)
def download_model(model_id: str, models: ModelManager = Depends(manager)) -> dict:
    try:
        return models.download(model_id).to_dict()
    except ModelNotFound:
        raise HTTPException(404, f"Unknown model '{model_id}'") from None


@router.get("/models/trained")
def trained_models(models: ModelManager = Depends(manager)) -> dict:
    """Fine-tuned models with their base model, dataset, final loss and training settings."""
    return {"data": models.trained()}


@router.delete("/models/{model_id}", status_code=204)
def delete_model(model_id: str, models: ModelManager = Depends(manager)) -> None:
    try:
        models.delete(model_id)
    except ModelNotFound:
        raise HTTPException(404, f"Unknown model '{model_id}'") from None


@router.get("/models/{model_id}/archive")
def model_archive(model_id: str, models: ModelManager = Depends(manager)) -> StreamingResponse:
    """The model's folder and settings as a tar, to install on another engine with
    POST /api/models/uploads."""
    # Any model folder, also a fine-tuned model whose base model is not installed here.
    if model_id.startswith(".") or "/" in model_id or not models.exists(model_id):
        raise HTTPException(404, f"Unknown model '{model_id}'")
    size, content = archive(models, model_id)
    return StreamingResponse(
        content,
        media_type="application/x-tar",
        headers={
            "content-length": str(size),
            "content-disposition": f"attachment; filename*=UTF-8''{quote(model_id)}.tar",
        },
    )


class NewUpload(BaseModel):
    name: str
    size: int = Field(gt=0)


def uploads(request: Request) -> ModelUploads:
    return request.app.state.model_uploads


def find_upload(upload_id: str, store: ModelUploads = Depends(uploads)) -> Upload:
    try:
        return store.get(upload_id)
    except KeyError:
        raise HTTPException(404, f"Unknown upload '{upload_id}'") from None


@router.post("/models/uploads", status_code=201)
def start_upload(body: NewUpload, store: ModelUploads = Depends(uploads)) -> dict:
    """Starts receiving a model file (a model archive or a .gguf file), sent in parts with PUT."""
    try:
        return store.start(body.name, body.size).to_dict()
    except UploadError as e:
        raise HTTPException(422, str(e)) from None


@router.put("/models/uploads/{upload_id}")
async def upload_part(
    request: Request,
    offset: int = Query(ge=0),
    upload: Upload = Depends(find_upload),
    store: ModelUploads = Depends(uploads),
) -> dict:
    """Appends the request body at offset, which must be the number of bytes received so far;
    parts of a few tens of MB pass RunPod's proxy."""
    pieces: list[bytes] = [piece async for piece in request.stream()]
    try:
        return (await run_in_threadpool(store.append, upload, offset, iter(pieces))).to_dict()
    except UploadError as e:
        raise HTTPException(409, str(e)) from None


@router.post("/models/uploads/{upload_id}/finish", status_code=202)
def finish_upload(upload: Upload = Depends(find_upload), store=Depends(uploads)) -> dict:
    """Installs the received model in the background; poll GET /api/models/uploads/{id}."""
    try:
        return store.finish(upload).to_dict()
    except UploadError as e:
        raise HTTPException(409, str(e)) from None


@router.get("/models/uploads/{upload_id}")
def upload_state(upload: Upload = Depends(find_upload)) -> dict:
    return upload.to_dict()


@router.get("/models/{model_id}/settings")
def model_settings(model_id: str, models: ModelManager = Depends(manager)) -> dict:
    try:
        return models.model_settings(model_id)
    except ModelNotFound:
        raise HTTPException(404, f"Unknown model '{model_id}'") from None


@router.put("/models/{model_id}/settings")
def save_model_settings(
    model_id: str,
    body: ModelSettings,
    models: ModelManager = Depends(manager),
    store: KnowledgeStore = Depends(knowledge),
) -> dict:
    """Replaces the model's settings; left-out fields go back to the engine's defaults."""
    if body.collection and body.collection not in {c["name"] for c in store.collections()}:
        raise HTTPException(422, f"The collection '{body.collection}' does not exist")
    try:
        return models.set_model_settings(model_id, body)
    except ModelNotFound:
        raise HTTPException(404, f"Unknown model '{model_id}'") from None


@router.get("/hub/models")
async def search_hub(
    q: str = "", format: Literal["gguf", "hf"] = "gguf", models: ModelManager = Depends(manager)
) -> dict:
    try:
        found = await run_in_threadpool(hub.search, q, format, models.hf_token)
    except hub.HubError as error:
        raise HTTPException(502, str(error)) from None
    return {"data": found}


@router.get("/hub/files")
async def hub_files(repo: str, models: ModelManager = Depends(manager)) -> dict:
    try:
        return await run_in_threadpool(hub.files, repo, models.hf_token)
    except hub.HubError as error:
        raise HTTPException(502, str(error)) from None


class HubDownload(BaseModel):
    repo: str
    # A .gguf file of the repo; left out: the Hugging Face-format weights.
    file: str | None = None


@router.post("/hub/download", status_code=202)
def download_from_hub(body: HubDownload, models: ModelManager = Depends(manager)) -> dict:
    try:
        return models.download_from_hub(body.repo, body.file).to_dict()
    except ModelExists as error:
        raise HTTPException(409, f"A model named '{error}' is already installed") from None
    except ValueError as error:
        raise HTTPException(400, str(error)) from None
