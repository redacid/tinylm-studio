"""Management API for exporting fine-tuned models to GGUF for llama.cpp."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from engine.api.deps import require_api_key
from engine.export.jobs import ExportError, ExportManager, ExportNotFound

router = APIRouter(dependencies=[Depends(require_api_key)], tags=["export"])


class ExportRequest(BaseModel):
    quantization: Literal["Q8_0", "Q4_K_M", "F16"] = "Q8_0"
    name: str | None = None  # defaults to "<model>-<quantization>"
    # Take this collection along: the exported model answers from it, also on another machine.
    collection: str | None = None


def exports(request: Request) -> ExportManager:
    return request.app.state.exports


@router.post("/api/models/{model_id}/export", status_code=202)
def export_model(
    model_id: str, body: ExportRequest, manager: ExportManager = Depends(exports)
) -> dict:
    """Merges a fine-tuned model's adapter into its base model and writes one GGUF file."""
    try:
        return manager.start(model_id, body.quantization, body.name, body.collection).to_dict()
    except ExportError as e:
        raise HTTPException(409 if "running" in str(e) else 422, str(e)) from None


@router.get("/api/exports")
def list_exports(manager: ExportManager = Depends(exports)) -> dict:
    return {"data": [e.to_dict() for e in manager.exports()]}


@router.get("/api/exports/{export_id}")
def get_export(export_id: str, manager: ExportManager = Depends(exports)) -> dict:
    try:
        return manager.get(export_id).to_dict()
    except ExportNotFound:
        raise HTTPException(404, f"Unknown export '{export_id}'") from None
