from uuid import UUID
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.security import require_internal_key
from app.services import practice_service


router = APIRouter(
    prefix="/internal",
    dependencies=[Depends(require_internal_key)]
)


class GenerateRequest(BaseModel):
    user_id: UUID
    topic_id: UUID
    document_id: UUID
    number_of_questions: int = Field(default=5, ge=1, le=20)
    difficulty: Literal["easy", "medium", "hard"] = "medium"


class StartPracticeRequest(BaseModel):
    user_id: UUID
    topic_id: UUID | None = None
    question_ids: list[UUID] = Field(min_length=1, max_length=50)


class SubmitAnswerRequest(BaseModel):
    user_id: UUID
    question_id: UUID
    selected_option_id: UUID
    response_time_seconds: int | None = Field(default=None, ge=0)


class QuestionListRequest(BaseModel):
    user_id: UUID
    topic_id: UUID | None = None
    limit: int = Field(default=20, ge=1, le=100)


@router.post("/courses/{course_id}/practice/generate")
def generate(course_id: UUID, body: GenerateRequest):

    try:
        return practice_service.generate_questions(
            user_id=str(body.user_id),
            course_id=str(course_id),
            topic_id=str(body.topic_id),
            document_id=str(body.document_id),
            number_of_questions=body.number_of_questions,
            difficulty=body.difficulty,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Question generation failed"
        ) from exc


@router.post("/courses/{course_id}/practice/questions")
def list_questions(course_id: UUID, body: QuestionListRequest):

    return practice_service.get_questions(
        user_id=str(body.user_id),
        course_id=str(course_id),
        topic_id=str(body.topic_id) if body.topic_id else None,
        limit=body.limit,
    )


@router.post("/courses/{course_id}/practice/sessions")
def start_session(course_id: UUID, body: StartPracticeRequest):

    try:
        return practice_service.start_practice(
            user_id=str(body.user_id),
            course_id=str(course_id),
            topic_id=str(body.topic_id) if body.topic_id else None,
            question_ids=[str(qid) for qid in body.question_ids],
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not start practice session"
        ) from exc


@router.post("/courses/{course_id}/practice/sessions/{session_id}/attempts")
def submit_attempt(
    course_id: UUID,
    session_id: UUID,
    body: SubmitAnswerRequest,
):

    try:
        return practice_service.submit_answer(
            user_id=str(body.user_id),
            course_id=str(course_id),
            session_id=str(session_id),
            question_id=str(body.question_id),
            selected_option_id=str(body.selected_option_id),
            response_time_seconds=body.response_time_seconds,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not submit practice answer"
        ) from exc