"""Progress of dataset generation runs."""

from fastapi import Depends, HTTPException

from engine.api.datasets import generations
from engine.api.deps import require_api_key
from engine.api.schemas import UNAUTHORIZED, GenerationJob, Router, errors, items
from engine.datasets.generate import GenerationManager

router = Router(
    prefix="/api/generations",
    dependencies=[Depends(require_api_key)],
    tags=["datasets"],
    responses=UNAUTHORIZED,
)


@router.get("", response_model=items(GenerationJob))
def list_generations(manager: GenerationManager = Depends(generations)) -> dict:
    return {"data": [job.to_dict() for job in manager.jobs.values()]}


@router.get("/{job_id}", response_model=GenerationJob, responses=errors(404))
def get_generation(job_id: str, manager: GenerationManager = Depends(generations)) -> dict:
    job = manager.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown generation '{job_id}'")
    return job.to_dict()


@router.post("/{job_id}/cancel", response_model=GenerationJob, responses=errors(404))
def cancel_generation(job_id: str, manager: GenerationManager = Depends(generations)) -> dict:
    """Stops after the chunk in progress."""
    job = manager.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown generation '{job_id}'")
    job.cancelled.set()
    return job.to_dict()
