"""Management API used by the desktop app: model catalog and downloads."""

from fastapi import APIRouter, Depends, HTTPException

from engine.api.deps import manager, require_api_key
from engine.models.manager import ModelManager, ModelNotFound

router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)], tags=["admin"])


@router.get("/models")
def model_catalog(models: ModelManager = Depends(manager)) -> dict:
    return {"installed": models.installed(), "catalog": models.catalog()}


@router.post("/models/{model_id}/download", status_code=202)
def download_model(model_id: str, models: ModelManager = Depends(manager)) -> dict:
    try:
        return models.download(model_id).to_dict()
    except ModelNotFound:
        raise HTTPException(404, f"Unknown model '{model_id}'") from None
