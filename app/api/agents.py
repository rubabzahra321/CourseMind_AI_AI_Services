
from uuid import UUID
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException

from app.core.security import require_internal_key
from app.services import stage6_service


router = APIRouter(
    prefix="/internal/courses/{course_id}/agents",
    dependencies=[Depends(require_internal_key)],
)


class UserRequest(BaseModel):
    user_id: UUID


class CompleteItemRequest(BaseModel):
    user_id: UUID
    item_id: UUID


@router.post("/dashboard")
def agent_dashboard(course_id: UUID, payload: UserRequest):
    return stage6_service.get_agent_dashboard(
        payload.user_id, course_id
    )


@router.post("/prerequisites/{topic_id}")
def prerequisite_path(
    course_id: UUID,
    topic_id: UUID,
    payload: UserRequest,
):
    try:
        return {
            "target_topic_id": str(topic_id),
            "algorithm": "Breadth-First Search",
            "path": stage6_service.get_prerequisite_path(
                payload.user_id, course_id, topic_id
            ),
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/syllabus/{topic_id}/subtree")
def syllabus_subtree(
    course_id: UUID,
    topic_id: UUID,
    payload: UserRequest,
):
    try:
        return {
            "root_topic_id": str(topic_id),
            "algorithm": "Depth-First Search",
            "topics": stage6_service.get_syllabus_subtree(
                payload.user_id, course_id, topic_id
            ),
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/study-plan/generate")
def create_study_plan(course_id: UUID, payload: UserRequest):
    try:
        return stage6_service.generate_study_plan(
            payload.user_id, course_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/study-plan/current")
def current_study_plan(course_id: UUID, payload: UserRequest):
    return stage6_service.get_current_study_plan(
        payload.user_id, course_id
    )


@router.post("/study-plan/complete")
def complete_study_item(
    course_id: UUID,
    payload: CompleteItemRequest,
):
    try:
        return stage6_service.complete_study_plan_item(
            payload.user_id, course_id, payload.item_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
