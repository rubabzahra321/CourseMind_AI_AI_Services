
from uuid import UUID
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.security import require_internal_key
from app.services import practice_service
from app.services import past_paper_practice_service


router = APIRouter(
    prefix="/internal",
    dependencies=[Depends(require_internal_key)]
)


# =========================================================
# EXISTING GENERATED QUESTION REQUESTS
# =========================================================

class GenerateRequest(BaseModel):
    user_id: UUID
    topic_id: UUID
    document_id: UUID
    idempotency_key: UUID | None = None
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


class SessionReportRequest(BaseModel):
    user_id: UUID


# =========================================================
# TASK 11.3: PAST-PAPER PRACTICE REQUESTS
# =========================================================

class PastPaperQuestionListRequest(BaseModel):
    user_id: UUID
    topic_id: UUID | None = None
    limit: int = Field(default=20, ge=1, le=100)


class StartPastPaperPracticeRequest(BaseModel):
    user_id: UUID
    topic_id: UUID | None = None
    question_ids: list[UUID] = Field(min_length=1, max_length=50)


class SubmitPastPaperAnswerRequest(BaseModel):
    user_id: UUID

    question_id: UUID

    response_text: str = Field(
        min_length=1,
        max_length=20000
    )

    response_time_seconds: int | None = Field(
        default=None,
        ge=0
    )


# =========================================================
# EXISTING: GENERATE QUESTIONS
# =========================================================

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
            idempotency_key=(
                str(body.idempotency_key)
                if body.idempotency_key
                else None
            ),
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Question generation failed"
        ) from exc


# =========================================================
# EXISTING: LIST GENERATED QUESTIONS
# =========================================================

@router.post("/courses/{course_id}/practice/questions")
def list_questions(course_id: UUID, body: QuestionListRequest):

    return practice_service.get_questions(
        user_id=str(body.user_id),
        course_id=str(course_id),
        topic_id=str(body.topic_id) if body.topic_id else None,
        limit=body.limit,
    )


# =========================================================
# EXISTING: START GENERATED MCQ PRACTICE
# =========================================================

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


# =========================================================
# EXISTING: SUBMIT GENERATED MCQ ANSWER
# =========================================================

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


# =========================================================
# TASK 11.3: LIST REVIEWED PAST-PAPER QUESTIONS
# =========================================================

@router.post("/courses/{course_id}/practice/past-paper/questions")
def list_past_paper_questions(
    course_id: UUID,
    body: PastPaperQuestionListRequest,
):

    try:
        return past_paper_practice_service.list_past_paper_questions(
            user_id=str(body.user_id),
            course_id=str(course_id),
            topic_id=str(body.topic_id) if body.topic_id else None,
            limit=body.limit,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve past-paper questions"
        ) from exc


# =========================================================
# TASK 11.3: START PAST-PAPER PRACTICE
# =========================================================

@router.post("/courses/{course_id}/practice/past-paper/sessions")
def start_past_paper_session(
    course_id: UUID,
    body: StartPastPaperPracticeRequest,
):

    try:
        return past_paper_practice_service.start_past_paper_practice(
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
            detail="Could not start past-paper practice"
        ) from exc


# =========================================================
# TASK 11.3: SUBMIT WRITTEN PAST-PAPER ANSWER
# =========================================================

@router.post(
    "/courses/{course_id}/practice/sessions/{session_id}/past-paper-attempts"
)
def submit_past_paper_attempt(
    course_id: UUID,
    session_id: UUID,
    body: SubmitPastPaperAnswerRequest,
):

    try:
        return past_paper_practice_service.submit_past_paper_answer(
            user_id=str(body.user_id),
            course_id=str(course_id),
            session_id=str(session_id),
            question_id=str(body.question_id),
            response_text=body.response_text,
            response_time_seconds=body.response_time_seconds,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not save past-paper answer"
        ) from exc


@router.post(
    "/courses/{course_id}/practice/sessions/{session_id}/report"
)
def get_session_report(
    course_id: UUID,
    session_id: UUID,
    body: SessionReportRequest,
):
    try:
        return practice_service.get_session_report(
            user_id=str(body.user_id),
            course_id=str(course_id),
            session_id=str(session_id),
        )

    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not generate practice session report"
        ) from exc
