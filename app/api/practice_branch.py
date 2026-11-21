from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.security import require_internal_key
from app.services import practice_branch_service


router = APIRouter(
    prefix="/internal",
    dependencies=[Depends(require_internal_key)],
)


class BranchPracticeRequest(BaseModel):
    user_id: UUID

    max_questions_per_topic: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    max_total_questions: int = Field(
        default=50,
        ge=1,
        le=50,
    )


@router.post(
    "/courses/{course_id}/practice/topics/{topic_id}/branch-session"
)
def start_branch_practice(
    course_id: UUID,
    topic_id: UUID,
    body: BranchPracticeRequest,
):
    try:
        return practice_branch_service.start_branch_practice(
            user_id=body.user_id,
            course_id=course_id,
            root_topic_id=topic_id,
            max_questions_per_topic=body.max_questions_per_topic,
            max_total_questions=body.max_total_questions,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not start topic-branch practice session.",
        ) from exc