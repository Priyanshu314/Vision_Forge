"""Sampling route – GET /sample/{run_id}"""

from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..db.models import get_run
from ..services.sampling_service import sample_representative_images

router = APIRouter()

DATA_DIR = Path("data/runs")


@router.get("/sample/{run_id}")
def sample_images(run_id: str, k: int = 20):
    """Cluster images from a previous upload run and return representative samples.

    Parameters
    ----------
    run_id:
        The run identifier returned by ``POST /upload``.
    k:
        Number of representative images to select (default 20).

    Returns
    -------
    dict
        ``run_id``, ``k``, and list of ``sampled_paths``.
    """
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    image_dir = DATA_DIR / run_id / "images"
    if not image_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"No images found for run '{run_id}'.")

    # Collect all image files
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = sorted(
        str(p.resolve())
        for p in image_dir.iterdir()
        if p.suffix.lower() in allowed_extensions
    )

    if not image_paths:
        raise HTTPException(status_code=404, detail="Image directory is empty.")

    sampled = sample_representative_images(image_paths, k=k)

    return {
        "run_id": run_id,
        "total_images": len(image_paths),
        "k": min(k, len(image_paths)),
        "sampled_paths": sampled,
    }
