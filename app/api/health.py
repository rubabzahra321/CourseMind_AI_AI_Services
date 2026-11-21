
from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.core.config import settings
from app.core.security import require_internal_key
from app.db.client import get_db
from app.repositories.health_repository import get_latest_worker_heartbeat

public = APIRouter()
internal = APIRouter(prefix="/internal", dependencies=[Depends(require_internal_key)])


@public.get("/health")
def health():
    return {"status": "ok"}


@internal.get("/db-check")
def db_check():
    get_db().table("courses").select("id").limit(1).execute()
    return {"database": "reachable"}


@internal.get("/health/full")
def full_health():
    checks = {}

    # Database check.
    try:
        get_db().table("courses").select("id").limit(1).execute()

        checks["database"] = {
            "status": "healthy",
        }

    except Exception:
        checks["database"] = {
            "status": "unhealthy",
        }

    # Worker check.
    try:
        heartbeat = get_latest_worker_heartbeat()

        if heartbeat:
            last_seen = datetime.fromisoformat(
                heartbeat["last_seen_at"].replace("Z", "+00:00")
            )

            age_seconds = (
                datetime.now(timezone.utc) - last_seen
            ).total_seconds()

            worker_healthy = age_seconds <= 60

            checks["worker"] = {
                "status": "healthy" if worker_healthy else "unhealthy",
                "worker_id": heartbeat["worker_id"],
                "last_seen_at": heartbeat["last_seen_at"],
            }

        else:
            checks["worker"] = {
                "status": "unhealthy",
                "message": "No worker heartbeat found.",
            }

    except Exception:
        checks["worker"] = {
            "status": "unhealthy",
        }

    # Provider configuration check.
    # This confirms that the API key is configured, not that
    # Gemini is reachable at this exact moment.
    checks["ai_provider"] = {
        "status": (
            "configured"
            if settings.gemini_api_key
            else "not_configured"
        ),
        "provider": "Gemini",
    }

    overall_healthy = (
        checks["database"]["status"] == "healthy"
        and checks["worker"]["status"] == "healthy"
        and checks["ai_provider"]["status"] == "configured"
    )

    return {
        "status": "healthy" if overall_healthy else "degraded",
        "checks": checks,
    }
