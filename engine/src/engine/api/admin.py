"""Management API used by the desktop app: model catalog and downloads."""

from fastapi import APIRouter, Depends, HTTPException

from engine.api.deps import manager, require_api_key
from engine.models.catalog import DEFAULT_MODEL
from engine.models.manager import ModelManager, ModelNotFound

router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)], tags=["admin"])


@router.get("/models")
def model_catalog(models: ModelManager = Depends(manager)) -> dict:
    return {
        "installed": models.installed(),
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
