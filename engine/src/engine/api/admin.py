"""Management API used by the desktop app: installed models, their settings, the catalog,
searching Hugging Face and downloads."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from engine.api.deps import knowledge, manager, require_api_key
from engine.models import hub
from engine.models.catalog import DEFAULT_MODEL
from engine.models.manager import ModelExists, ModelManager, ModelNotFound
from engine.models.settings import ModelSettings
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
