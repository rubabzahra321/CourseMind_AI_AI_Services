
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.security import require_internal_key
from app.repositories import chat_repository
from app.services import phase4_service


router = APIRouter(
    prefix="/internal",
    dependencies=[Depends(require_internal_key)],
)


class StartPastPaperRequest(BaseModel):
    user_id: UUID
    topic_id: UUID | None = None
    limit: int = Field(default=20, ge=1, le=50)


class SubmitPastPaperRequest(BaseModel):
    user_id: UUID
    question_id: UUID
    text_answer: str = Field(min_length=1, max_length=12000)
    response_time_seconds: int | None = Field(default=None, ge=0)


# ==========================================
# Chat history
# ==========================================

@router.get("/courses/{course_id}/chat/sessions")
def chat_sessions(
    course_id: UUID,
    user_id: UUID,
    limit: int = Query(default=30, ge=1, le=100),
):
    return chat_repository.list_sessions(
        str(user_id),
        str(course_id),
        limit,
    )


@router.get(
    "/courses/{course_id}/chat/sessions/{session_id}/messages"
)
def chat_messages(
    course_id: UUID,
    session_id: UUID,
    user_id: UUID,
    limit: int = Query(default=100, ge=1, le=200),
):
    try:
        return chat_repository.list_messages(
            str(session_id),
            str(user_id),
            str(course_id),
            limit,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


# ==========================================
# Practice attempt history
# ==========================================

@router.get("/courses/{course_id}/practice/attempts")
def attempt_history(
    course_id: UUID,
    user_id: UUID,
    limit: int = Query(default=50, ge=1, le=100),
):
    return phase4_service.get_attempt_history(
        str(user_id),
        str(course_id),
        limit,
    )


# ==========================================
# Incorrect questions
# ==========================================

@router.get("/courses/{course_id}/practice/incorrect")
def incorrect_questions(
    course_id: UUID,
    user_id: UUID,
    limit: int = Query(default=100, ge=1, le=200),
):
    return phase4_service.get_incorrect_questions(
        str(user_id),
        str(course_id),
        limit,
    )


# ==========================================
# Reviewed past-paper practice
# ==========================================

@router.post("/courses/{course_id}/practice/past-paper/sessions")
def start_past_paper_session(
    course_id: UUID,
    body: StartPastPaperRequest,
):
    try:
        return phase4_service.start_past_paper_practice(
            user_id=str(body.user_id),
            course_id=str(course_id),
            topic_id=str(body.topic_id) if body.topic_id else None,
            limit=body.limit,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not start past-paper practice",
        ) from exc


@router.post(
    "/courses/{course_id}/practice/sessions/{session_id}/past-paper-answers"
)
def submit_past_paper_answer(
    course_id: UUID,
    session_id: UUID,
    body: SubmitPastPaperRequest,
):
    try:
        return phase4_service.submit_past_paper_answer(
            user_id=str(body.user_id),
            course_id=str(course_id),
            session_id=str(session_id),
            question_id=str(body.question_id),
            text_answer=body.text_answer,
            response_time_seconds=body.response_time_seconds,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not save past-paper answer",
        ) from exc


# ==========================================
# Per-session performance report
# ==========================================

@router.get(
    "/courses/{course_id}/practice/sessions/{session_id}/report"
)
def session_report(
    course_id: UUID,
    session_id: UUID,
    user_id: UUID,
):
    try:
        return phase4_service.get_session_report(
            str(user_id),
            str(course_id),
            str(session_id),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not load practice report",
        ) from exc
