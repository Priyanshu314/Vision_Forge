from fastapi import APIRouter, HTTPException
from ..services.training_service import train_model_task
from ..db.models import get_run
from celery.result import AsyncResult
from pathlib import Path

router = APIRouter()

@router.post("/train/{run_id}")
async def start_training(run_id: str):
    """Trigger the asynchronous training pipeline for a given run."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")
        
    ann_path = Path("data/runs") / run_id / "annotations" / "train.json"
    if not ann_path.exists():
        raise HTTPException(
            status_code=400, 
            detail="Annotations not found. Please upload annotations before training."
        )

    task = train_model_task.delay(run_id)
    return {
        "status": "training_started",
        "run_id": run_id,
        "task_id": task.id
    }

@router.get("/train/status/{task_id}")
async def get_training_status(task_id: str):
    """Check the status of a training job."""
    res = AsyncResult(task_id)
    state = res.state
    
    # Extract result or error info
    if state == "FAILURE":
        result = str(res.result)
    elif state == "SUCCESS":
        result = res.result
    else:
        # For PENDING/PROGRESS, res.info often contains custom meta (like loss)
        result = res.info
        
    return {
        "task_id": task_id,
        "status": state,
        "result": result
    }
