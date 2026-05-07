import json
from fastapi import APIRouter, HTTPException
from ..services.inference_service import run_inference_and_eval
from .upload_service import DATA_DIR

router = APIRouter()

@router.post("/infer/{run_id}")
async def start_inference(run_id: str):
    """Trigger the inference and evaluation pipeline for a run."""
    try:
        results = run_inference_and_eval(run_id)
        return {
            "status": "success",
            "run_id": run_id,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results/{run_id}")
async def get_results(run_id: str):
    """Retrieve saved inference results and metrics for a run."""
    run_dir = DATA_DIR / run_id
    metrics_path = run_dir / "metrics.json"
    pred_path = run_dir / "predictions" / "preds.json"
    
    if not metrics_path.exists() or not pred_path.exists():
        raise HTTPException(status_code=404, detail="Results not found. Did you run /infer first?")
        
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    return {
        "run_id": run_id,
        "metrics": metrics,
        "predictions_file": str(pred_path.resolve())
    }
