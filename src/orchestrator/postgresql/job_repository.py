import logging
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

JOB_COLUMNS = "id::text AS id, video_id, status, stage, frames_total, frames_done, frames_failed, error, created_at, updated_at"

# Columns update_job may set (whitelist keeps the dynamic SET clause safe)
_UPDATABLE = ("status", "stage", "video_id", "frames_total", "frames_done", "frames_failed", "error")


def create_job(video_id: Optional[int] = None) -> str:
    """Create a queued job and return its ID (UUID string). Raises on database errors."""
    from .connection import DatabaseManager

    job_id = str(uuid.uuid4())
    with DatabaseManager() as cursor:
        cursor.execute(
            "INSERT INTO jobs (id, video_id, status, stage) VALUES (%s::uuid, %s, 'queued', 'queued')",
            (job_id, video_id),
        )
    return job_id


def update_job(job_id: str, **fields: Any) -> bool:
    """Update the given job fields (see _UPDATABLE) and bump updated_at. Raises on database errors."""
    from .connection import DatabaseManager

    unknown = set(fields) - set(_UPDATABLE)
    if unknown:
        raise ValueError(f"Unknown job fields: {sorted(unknown)}")
    if not fields:
        return False

    assignments = ", ".join(f"{col} = %({col})s" for col in fields)
    with DatabaseManager() as cursor:
        cursor.execute(
            f"UPDATE jobs SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE id = %(job_id)s::uuid",
            {**fields, "job_id": job_id},
        )
        return cursor.rowcount > 0


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Return the job as a dict, or None if it doesn't exist. Raises on database errors."""
    from .connection import DatabaseManager

    with DatabaseManager() as cursor:
        cursor.execute(f"SELECT {JOB_COLUMNS} FROM jobs WHERE id = %s::uuid", (job_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def fail_interrupted_jobs() -> int:
    """Mark jobs left in a non-terminal status (e.g. by a restart) as failed. Returns the number updated."""
    from .connection import DatabaseManager

    with DatabaseManager() as cursor:
        cursor.execute(
            """
            UPDATE jobs
            SET status = 'failed', error = 'interrupted by restart', updated_at = CURRENT_TIMESTAMP
            WHERE status NOT IN ('done', 'failed')
            """
        )
        return cursor.rowcount
