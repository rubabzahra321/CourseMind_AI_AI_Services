from uuid import UUID
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.security import require_internal_key
from app.services import practice_formats_service


router = APIRouter(
    prefix="/internal",
    dependencies=[Depends(require_internal_key)],
)


class GenerateFormatRequest(BaseModel):
    user_id: UUID
    topic_id: UUID
    document_id: UUID
    question_type: Literal[
        "mcq",
        "true_false",
        "short_answer",
        "long_answer",
    ]
    number_of_questions: int = Field(default=5, ge=1, le=20)
    difficulty: Literal["easy", "medium", "hard"] = "medium"


class StartFormatPracticeRequest(BaseModel):
    user_id: UUID
    topic_id: UUID | None = None
    question_ids: list[UUID] = Field(min_length=1, max_length=50)


class SubmitFormatAnswerRequest(BaseModel):
    user_id: UUID
    question_id: UUID
    selected_option_id: UUID | None = None
    response_text: str | None = Field(default=None, max_length=20000)
    response_time_seconds: int | None = Field(default=None, ge=0)


@router.post("/courses/{course_id}/practice/formats/generate")
def generate_questions(
    course_id: UUID,
    body: GenerateFormatRequest,
):
    try:
        return practice_formats_service.generate_questions(
            user_id=str(body.user_id),
            course_id=str(course_id),
            topic_id=str(body.topic_id),
            document_id=str(body.document_id),
            question_type=body.question_type,
            number_of_questions=body.number_of_questions,
            difficulty=body.difficulty,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Question generation failed",
        ) from exc


@router.post("/courses/{course_id}/practice/formats/sessions")
def start_practice(
    course_id: UUID,
    body: StartFormatPracticeRequest,
):
    try:
        return practice_formats_service.start_practice(
            user_id=str(body.user_id),
            course_id=str(course_id),
            topic_id=str(body.topic_id) if body.topic_id else None,
            question_ids=[str(item) for item in body.question_ids],
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not start practice session",
        ) from exc


@router.post(
    "/courses/{course_id}/practice/formats/sessions/{session_id}/attempts"
)
def submit_answer(
    course_id: UUID,
    session_id: UUID,
    body: SubmitFormatAnswerRequest,
):
    try:
        return practice_formats_service.submit_answer(
            user_id=str(body.user_id),
            course_id=str(course_id),
            session_id=str(session_id),
            question_id=str(body.question_id),
            selected_option_id=(
                str(body.selected_option_id)
                if body.selected_option_id else None
            ),
            response_text=body.response_text,
            response_time_seconds=body.response_time_seconds,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not submit practice answer",
        ) from exc