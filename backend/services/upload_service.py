"""Upload service – handles run creation and image persistence."""

import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import UploadFile

from ..db.models import Run, save_run

# Base directory where all run data is stored
DATA_DIR = Path("data/runs")


def create_run() -> Run:
    """Create a new Run with a unique ID and register it in the store."""
    run_id = uuid.uuid4().hex[:12]
    run = Run(run_id=run_id)
    save_run(run)
    return run


async def save_images(run_id: str, files: List[UploadFile]) -> List[str]:
    """Save uploaded images to disk under ``data/runs/{run_id}/images/``.

    Parameters
    ----------
    run_id:
        The unique run identifier.
    files:
        List of FastAPI ``UploadFile`` objects.

    Returns
    -------
    list[str]
        Absolute paths to the saved images.
    """
    image_dir = DATA_DIR / run_id / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []
    for file in files:
        # Preserve original filename; prefix with index to avoid collisions
        dest = image_dir / file.filename
        # If a file with the same name already exists, make it unique
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            dest = image_dir / f"{stem}_{uuid.uuid4().hex[:6]}{suffix}"

        with dest.open("wb") as f:
            content = await file.read()
            f.write(content)
        saved_paths.append(str(dest.resolve()))

    return saved_paths
