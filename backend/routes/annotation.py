from fastapi import APIRouter, HTTPException
from ..schemas.annotation import CocoFormat
from ..services.annotation_service import save_annotations
from ..db.models import get_run

router = APIRouter()


@router.post("/annotate/{run_id}")
async def annotate_run(run_id: str, coco_data: CocoFormat):
    """Save COCO annotations for a specific run."""
    
    # Check if run exists
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")
        
    try:
        file_path = save_annotations(run_id, coco_data)
        return {
            "status": "success",
            "run_id": run_id,
            "file_path": file_path,
            "num_annotations": len(coco_data.annotations)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
