"""Management API for fine-tuning: start, follow, stop and resume LoRA training runs."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from engine.api.deps import require_api_key
from engine.datasets.store import DatasetNotFound
from engine.training.jobs import Busy, JobNotFound, TrainingError, TrainingManager, TrainingSettings

router = APIRouter(
    prefix="/api/training", dependencies=[Depends(require_api_key)], tags=["training"]
)
# How often the event stream looks for new progress, in seconds.
POLL = 1.0


class TrainingRequest(BaseModel):
    name: str
    base: str
    dataset: str
    epochs: float = Field(3, gt=0, le=100)
    learning_rate: float = Field(2e-4, gt=0, lt=1)
    lora_rank: int = Field(16, ge=1, le=256)
    lora_alpha: int | None = Field(None, ge=1, le=1024)  # defaults to twice the rank
    lora_dropout: float = Field(0.05, ge=0, lt=1)
    max_length: int = Field(1024, ge=32, le=32768)
    batch_size: int = Field(1, ge=1, le=256)
    gradient_accumulation: int = Field(4, ge=1, le=256)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    threads: int | None = Field(None, ge=1, le=512)
    save_steps: int | None = Field(None, ge=1)


def training(request: Request) -> TrainingManager:
    return request.app.state.training


def not_found(job_id: str) -> HTTPException:
    return HTTPException(404, f"Unknown training run '{job_id}'")


@router.get("/jobs")
def list_jobs(manager: TrainingManager = Depends(training)) -> dict:
    return {"data": [job.to_dict(history=False) for job in manager.jobs()]}


@router.post("/jobs", status_code=201)
def start_job(body: TrainingRequest, manager: TrainingManager = Depends(training)) -> dict:
    fields = body.model_dump()
    fields["lora_alpha"] = body.lora_alpha or body.lora_rank * 2
    try:
        return manager.start(TrainingSettings(**fields)).to_dict()
    except Busy as e:
        raise HTTPException(409, str(e)) from None
    except DatasetNotFound:
        raise HTTPException(404, f"Unknown dataset '{body.dataset}'") from None
    except TrainingError as e:
        raise HTTPException(422, str(e)) from None


@router.get("/jobs/{job_id}")
def get_job(job_id: str, manager: TrainingManager = Depends(training)) -> dict:
    try:
        job = manager.job(job_id)
    except JobNotFound:
        raise not_found(job_id) from None
    return {**job.to_dict(), "log": manager.log(job_id) if job.state == "error" else ""}


@router.post("/jobs/{job_id}/stop")
def stop_job(job_id: str, manager: TrainingManager = Depends(training)) -> dict:
    try:
        return manager.stop(job_id).to_dict()
    except JobNotFound:
        raise not_found(job_id) from None


@router.post("/jobs/{job_id}/resume")
def resume_job(job_id: str, manager: TrainingManager = Depends(training)) -> dict:
    try:
        return manager.resume(job_id).to_dict()
    except JobNotFound:
        raise not_found(job_id) from None
    except Busy as e:
        raise HTTPException(409, str(e)) from None
    except TrainingError as e:
        raise HTTPException(422, str(e)) from None


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str, manager: TrainingManager = Depends(training)) -> None:
    try:
        manager.delete(job_id)
    except JobNotFound:
        raise not_found(job_id) from None
    except TrainingError as e:
        raise HTTPException(409, str(e)) from None


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request, manager: TrainingManager = Depends(training)):
    """Server-sent events: every worker event so far (start, step, eval, checkpoint, done,
    stopped, error), then new ones as they come, and a "state" event whenever the run's state
    changes. The stream ends when the run is no longer going."""
    try:
        manager.job(job_id)
    except JobNotFound:
        raise not_found(job_id) from None

    async def stream() -> AsyncIterator[str]:
        offset, state = 0, None
        while not await request.is_disconnected():
            events, offset = manager.events(job_id, offset)
            for event in events:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            job = manager.job(job_id)
            if job.state != state:
                state = job.state
                change = {"type": "state", "state": state, "error": job.error}
                yield f"data: {json.dumps(change)}\n\n"
            if state not in ("running", "stopping"):
                # The worker may have written its last events just before it ended.
                events, offset = manager.events(job_id, offset)
                for event in events:
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                return
            await asyncio.sleep(POLL)

    return StreamingResponse(stream(), media_type="text/event-stream")
