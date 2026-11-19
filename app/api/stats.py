from uuid import UUID
from fastapi import APIRouter, Depends
from app.core.security import require_internal_key
from app.services import stats_service

router = APIRouter(prefix="/internal", dependencies=[Depends(require_internal_key)])


@router.get("/courses/{course_id}/topic-stats")
def topic_stats(course_id: UUID, user_id: UUID):
    return stats_service.get_course_topic_stats(user_id, course_id)