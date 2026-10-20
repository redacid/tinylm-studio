"""Management API for evaluations: compare models on the held-out examples of a dataset."""

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel, Field

from engine.api.deps import require_api_key
from engine.api.schemas import UNAUTHORIZED, Evaluation, Router, errors, items
from engine.datasets.store import DatasetNotFound
from engine.evaluation.evaluate import (
    EvaluationError,
    EvaluationManager,
    EvaluationNotFound,
    EvaluationOptions,
)
from engine.teachers.store import TeacherNotFound

router = Router(
    prefix="/api/evaluations",
    dependencies=[Depends(require_api_key)],
    tags=["evaluation"],
    responses=UNAUTHORIZED,
)


class EvaluationRequest(BaseModel):
    dataset: str
    models: list[str]
    judge: str | None = None  # "ref" from GET /api/teachers
    limit: int = Field(50, ge=1, le=5000)
    max_tokens: int = Field(512, ge=16, le=8192)


def evaluations(request: Request) -> EvaluationManager:
    return request.app.state.evaluations


def find(manager: EvaluationManager, evaluation_id: str):
    try:
        return manager.get(evaluation_id)
    except EvaluationNotFound:
        raise HTTPException(404, f"Unknown evaluation '{evaluation_id}'") from None


@router.get("", response_model=items(Evaluation))
def list_evaluations(manager: EvaluationManager = Depends(evaluations)) -> dict:
    """Runs with their summaries, oldest first, without the per-example answers."""
    return {"data": [e.to_dict(examples=False) for e in manager.all()]}


@router.post("", status_code=202, response_model=Evaluation, responses=errors(404, 409, 422))
def start_evaluation(
    body: EvaluationRequest, manager: EvaluationManager = Depends(evaluations)
) -> dict:
    try:
        return manager.start(EvaluationOptions(**body.model_dump())).to_dict(examples=False)
    except DatasetNotFound:
        raise HTTPException(404, f"Unknown dataset '{body.dataset}'") from None
    except TeacherNotFound:
        raise HTTPException(404, f"Unknown teacher '{body.judge}'") from None
    except EvaluationError as e:
        status = 409 if "running" in str(e) else 422
        raise HTTPException(status, str(e)) from None


@router.get("/{evaluation_id}", response_model=Evaluation, responses=errors(404))
def get_evaluation(evaluation_id: str, manager: EvaluationManager = Depends(evaluations)):
    """The report with every example's answers."""
    return find(manager, evaluation_id).to_dict()


@router.post("/{evaluation_id}/stop", response_model=Evaluation, responses=errors(404))
def stop_evaluation(evaluation_id: str, manager: EvaluationManager = Depends(evaluations)):
    """Stops after the answer in progress; the report keeps what was done."""
    find(manager, evaluation_id)
    return manager.stop(evaluation_id).to_dict(examples=False)


@router.delete("/{evaluation_id}", status_code=204, responses=errors(404, 409))
def delete_evaluation(evaluation_id: str, manager: EvaluationManager = Depends(evaluations)):
    find(manager, evaluation_id)
    try:
        manager.delete(evaluation_id)
    except EvaluationError as e:
        raise HTTPException(409, str(e)) from None
