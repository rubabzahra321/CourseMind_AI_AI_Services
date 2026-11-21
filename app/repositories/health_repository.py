from datetime import datetime, timezone

from app.db.client import get_db


def update_worker_heartbeat(worker_id: str):
    now = datetime.now(timezone.utc).isoformat()

    get_db().table("worker_heartbeats").upsert(
        {
            "worker_id": worker_id,
            "last_seen_at": now,
            "updated_at": now,
        },
        on_conflict="worker_id",
    ).execute()


def get_latest_worker_heartbeat():
    result = (
        get_db()
        .table("worker_heartbeats")
        .select("worker_id,last_seen_at")
        .order("last_seen_at", desc=True)
        .limit(1)
        .execute()
    )

    rows = result.data or []

    return rows[0] if rows else None