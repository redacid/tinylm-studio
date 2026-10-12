"""Progress of dataset generation runs."""

from fastapi import APIRouter, Depends, HTTPException

from engine.api.datasets import generations
from engine.api.deps import require_api_key
from engine.datasets.generate import GenerationManager

router = APIRouter(
    prefix="/api/generations", dependencies=[Depends(require_api_key)], tags=["datasets"]
)


@router.get("")
def list_generations(manager: GenerationManager = Depends(generations)) -> dict:
    return {"data": [job.to_dict() for job in manager.jobs.values()]}


@router.get("/{job_id}")
def get_generation(job_id: str, manager: GenerationManager = Depends(generations)) -> dict:
    job = manager.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown generation '{job_id}'")
    return job.to_dict()


@router.post("/{job_id}/cancel")
def cancel_generation(job_id: str, manager: GenerationManager = Depends(generations)) -> dict:
    """Stops after the chunk in progress."""
    job = manager.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown generation '{job_id}'")
    job.cancelled.set()
    return job.to_dict()
