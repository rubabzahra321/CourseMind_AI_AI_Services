
from uuid import UUID
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.security import require_internal_key
from app.services.rag_service import answer_question
from app.repositories import chat_repository


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

    depth: Literal[
        "short",
        "detailed",
        "step_by_step"
    ] = "detailed"


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
            depth=body.depth,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc)
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="AI chat processing failed"
        ) from exc


@router.get("/courses/{course_id}/chat/sessions")
def list_chat_sessions(
    course_id: UUID,
    user_id: UUID = Query(...),
    limit: int = Query(default=30, ge=1, le=100),
):
    try:
        sessions = chat_repository.list_sessions(
            user_id=str(user_id),
            course_id=str(course_id),
            limit=limit,
        )

        return {
            "sessions": sessions
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve chat sessions"
        ) from exc


@router.get("/courses/{course_id}/chat/sessions/{session_id}")
def get_chat_session(
    course_id: UUID,
    session_id: UUID,
    user_id: UUID = Query(...),
):
    try:
        # Verify that the session belongs to this user
        # and the requested course.
        session = chat_repository.get_session(
            str(session_id),
            str(user_id),
            str(course_id),
        )

        if not session:
            raise HTTPException(
                status_code=404,
                detail="Chat session not found"
            )

        messages = chat_repository.get_session_messages(
            str(session_id),
            str(user_id),
        )

        return {
            "session": session,
            "messages": messages,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve chat history"
        ) from exc
