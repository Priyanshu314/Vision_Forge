"""Lightweight in-memory database models.

These will be replaced with SQLAlchemy + PostgreSQL once the DB layer is wired up.
For now they serve as a simple registry so the upload and sampling endpoints work
end-to-end without an external database dependency.
"""

from datetime import datetime, timezone
from typing import Dict


class Run:
    """Represents a single upload run."""

    def __init__(self, run_id: str) -> None:
        self.run_id: str = run_id
        self.created_at: datetime = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Simple in-memory store – replaced by a real DB table later
# ---------------------------------------------------------------------------
_runs: Dict[str, Run] = {}


def save_run(run: Run) -> None:
    """Persist a Run object (in-memory for now)."""
    _runs[run.run_id] = run


def get_run(run_id: str):
    """Retrieve a Run by its ID, or None if not found."""
    return _runs.get(run_id)
