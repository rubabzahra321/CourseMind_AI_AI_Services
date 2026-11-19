from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.security import require_internal_key
from app.services.rag_service import answer_question


router = APIRouter(
    prefix="/internal",
    dependencies=[Depends(require_internal_key)]
)


class ChatRequest(BaseModel):

    user_id: UUID

    question: str = Field(
        min_length=3,
        max_length=4000
    )

    session_id: UUID | None = None


@router.post("/courses/{course_id}/chat")
def chat(course_id: UUID, body: ChatRequest):

    try:

        return answer_question(
            user_id=str(body.user_id),
            course_id=str(course_id),
            question=body.question,
            session_id=(
                str(body.session_id)
                if body.session_id
                else None
            ),
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc)
        )

    except Exception as exc:

        # Log the internal exception in production.
        raise HTTPException(
            status_code=500,
            detail="AI chat processing failed"
        ) from exc