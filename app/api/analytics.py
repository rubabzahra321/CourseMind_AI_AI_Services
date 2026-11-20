
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.security import require_internal_key
from app.services import analytics_service


router = APIRouter(
    prefix="/internal",
    dependencies=[Depends(require_internal_key)]
)


def _handle_value_error(exc):
    raise HTTPException(
        status_code=400,
        detail=str(exc)
    ) from exc


class PriorityWeightsBody(BaseModel):
    frequency: float = Field(ge=0, le=1)
    marks: float = Field(ge=0, le=1)
    syllabus_weight: float = Field(ge=0, le=1)
    mastery: float = Field(ge=0, le=1)


class MaterialLinkBody(BaseModel):
    chunk_id: UUID
    method: str = Field(default="manual", pattern="^(manual|ai)$")
    similarity: float | None = Field(default=None, ge=0, le=1)


@router.get(
    "/courses/{course_id}/topics/{topic_id}/evidence"
)
def topic_evidence(
    course_id: UUID,
    topic_id: UUID,
    user_id: UUID
):
    try:
        return analytics_service.get_topic_evidence(
            user_id,
            course_id,
            topic_id
        )
    except ValueError as exc:
        _handle_value_error(exc)


@router.get(
    "/courses/{course_id}/priority"
)
def topic_priorities(
    course_id: UUID,
    user_id: UUID
):
    return analytics_service.get_topic_priorities(
        user_id,
        course_id
    )


@router.get(
    "/courses/{course_id}/priority/weights"
)
def read_priority_weights(
    course_id: UUID,
    user_id: UUID
):
    return analytics_service.get_priority_weights(
        user_id,
        course_id
    )


@router.put(
    "/courses/{course_id}/priority/weights"
)
def change_priority_weights(
    course_id: UUID,
    user_id: UUID,
    body: PriorityWeightsBody
):
    try:
        return analytics_service.update_priority_weights(
            user_id,
            course_id,
            body.model_dump()
        )
    except ValueError as exc:
        _handle_value_error(exc)


@router.get(
    "/courses/{course_id}/topics/{topic_id}/material/suggestions"
)
def material_suggestions(
    course_id: UUID,
    topic_id: UUID,
    user_id: UUID,
    limit: int = 10
):
    if limit < 1 or limit > 50:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 50"
        )

    try:
        return analytics_service.suggest_topic_chunks(
            user_id,
            course_id,
            topic_id,
            limit
        )
    except ValueError as exc:
        _handle_value_error(exc)


@router.post(
    "/courses/{course_id}/topics/{topic_id}/material"
)
def create_material_link(
    course_id: UUID,
    topic_id: UUID,
    user_id: UUID,
    body: MaterialLinkBody
):
    try:
        return analytics_service.create_topic_chunk_link(
            user_id=user_id,
            course_id=course_id,
            topic_id=topic_id,
            chunk_id=body.chunk_id,
            method=body.method,
            similarity=body.similarity
        )
    except ValueError as exc:
        _handle_value_error(exc)


@router.get(
    "/courses/{course_id}/coverage"
)
def topic_coverage(
    course_id: UUID,
    user_id: UUID
):
    return analytics_service.get_topic_coverage(
        user_id,
        course_id
    )


@router.get(
    "/courses/{course_id}/gaps"
)
def course_gaps(
    course_id: UUID,
    user_id: UUID
):
    return analytics_service.detect_course_gaps(
        user_id,
        course_id
    )
