
from datetime import datetime, timedelta, timezone

from app.db.client import get_db


def _now() -> datetime:
    return datetime.now(timezone.utc)


def claim_next(worker_id: str) -> dict | None:
    # Calls the SQL function that uses FOR UPDATE SKIP LOCKED.
    res = get_db().rpc(
        "claim_processing_job",
        {"p_worker_id": worker_id}
    ).execute()

    rows = res.data or []

    if isinstance(rows, dict):
        rows = [rows]

    row = rows[0] if rows else None

    return row if row and row.get("id") else None


def _finish(job_id: str, fields: dict) -> None:
    get_db().table("processing_jobs").update(
        {
            "locked_at": None,
            "locked_by": None,
            **fields
        }
    ).eq("id", job_id).execute()


def complete(job_id: str) -> None:
    _finish(job_id, {
        "status": "completed",
        "completed_at": _now().isoformat(),
        "last_error": None
    })


def fail(job_id: str, error: str) -> None:
    _finish(job_id, {
        "status": "failed",
        "completed_at": _now().isoformat(),
        "last_error": error[:1000]
    })


def retry_later(
    job_id: str,
    error: str,
    delay_seconds: int
) -> None:

    _finish(job_id, {
        "status": "queued",
        "available_at": (
            _now() + timedelta(seconds=delay_seconds)
        ).isoformat(),
        "last_error": error[:1000],
    })


def enqueue(
    document: dict,
    job_type: str,
    idempotency_key: str | None = None
) -> dict | None:

    """
    Create one extraction job.
    Active-job and idempotency indexes are the final race guards.
    """

    db = get_db()

    active = (
        db.table("processing_jobs")
        .select("id, status")
        .eq("document_id", document["id"])
        .eq("job_type", job_type)
        .in_("status", ["queued", "processing"])
        .limit(1)
        .execute()
    )

    if active.data:
        return active.data[0]

    payload = {
        "document_id": document["id"],
        "course_id": document["course_id"],
        "user_id": document["user_id"],
        "job_type": job_type,
        "status": "queued",
        "priority": (
            60
            if job_type == "extract_syllabus_topics"
            else 70
        ),
        "idempotency_key": idempotency_key,
    }

    try:
        result = (
            db.table("processing_jobs")
            .insert(payload)
            .select("id, status")
            .single()
            .execute()
        )

        return result.data

    except Exception:

        # A concurrent enqueue may have won the unique
        # active-job/idempotency race.
        active = (
            db.table("processing_jobs")
            .select("id, status")
            .eq("document_id", document["id"])
            .eq("job_type", job_type)
            .in_("status", ["queued", "processing"])
            .limit(1)
            .execute()
        )

        if active.data:
            return active.data[0]

        raise


def find_stale(minutes: int) -> list[dict]:

    cutoff = (
        _now() - timedelta(minutes=minutes)
    ).isoformat()

    res = (
        get_db().table("processing_jobs")
        .select(
            "id, document_id, user_id, course_id, "
            "job_type, attempts, max_attempts"
        )
        .eq("status", "processing")
        .lt("locked_at", cutoff)
        .execute()
    )

    return res.data
