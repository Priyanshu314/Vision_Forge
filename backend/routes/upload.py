"""Upload route – POST /upload"""

from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException

from ..services.upload_service import create_run, save_images

router = APIRouter()


@router.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    """Accept multiple image files, create a new run, and persist them to disk.

    Returns
    -------
    dict
        ``run_id``, ``created_at``, number of images saved, and their paths.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    # Validate that every file looks like an image
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    for f in files:
        ext = "." + f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {f.filename}. Allowed: {allowed_extensions}",
            )

    run = create_run()
    saved_paths = await save_images(run.run_id, files)

    return {
        "run_id": run.run_id,
        "created_at": run.created_at.isoformat(),
        "num_images": len(saved_paths),
        "saved_paths": saved_paths,
    }

@router.get("/runs")
def list_runs():
    """List all available runs."""
    from pathlib import Path
    import os
    runs_dir = Path("data/runs")
    if not runs_dir.exists():
        return {"runs": []}
    
    # Sort by creation time (newest first)
    runs = []
    for d in runs_dir.iterdir():
        if d.is_dir():
            runs.append({
                "run_id": d.name,
                "created_at": os.path.getctime(d)
            })
            
    runs.sort(key=lambda x: x["created_at"], reverse=True)
    return {"runs": [r["run_id"] for r in runs]}
