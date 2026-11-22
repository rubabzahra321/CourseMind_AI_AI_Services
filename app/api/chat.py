"""Chat (RAG) endpoints.

The Node backend calls these:
    POST /internal/courses/{course_id}/chat
    GET  /internal/courses/{course_id}/chat/sessions
    GET  /internal/courses/{course_id}/chat/sessions/{session_id}
"""

import logging
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.security import require_internal_key
from app.repositories import chat_repository
from app.services import rag_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/internal",
    dependencies=[Depends(require_internal_key)],
)


class ChatRequest(BaseModel):
    user_id: UUID
    question: str = Field(min_length=3, max_length=4000)
    session_id: UUID | None = None
    depth: Literal["short", "detailed", "step_by_step"] = "detailed"
    simplify: bool = False
    include_examples: bool = False


# ==========================================
# Ask a question
# ==========================================

@router.post("/courses/{course_id}/chat")
def chat(course_id: UUID, body: ChatRequest):
    try:
        return rag_service.answer_question(
            user_id=str(body.user_id),
            course_id=str(course_id),
            question=body.question,
            session_id=str(body.session_id) if body.session_id else None,
            depth=body.depth,
            simplify=body.simplify,
            include_examples=body.include_examples,
        )

    except ValueError as exc:
        # e.g. "Chat session not found" (unknown, foreign or other-course session)
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    except Exception as exc:  # noqa: BLE001 - never leak provider/DB details
        logger.exception("Chat processing failed")
        raise HTTPException(
            status_code=500,
            detail="AI chat processing failed",
        ) from exc


# ==========================================
# Chat history
# ==========================================

@router.get("/courses/{course_id}/chat/sessions")
def list_chat_sessions(
    course_id: UUID,
    user_id: UUID,
    limit: int = Query(default=30, ge=1, le=100),
):
    return chat_repository.list_sessions(
        str(user_id),
        str(course_id),
        limit,
    )


@router.get("/courses/{course_id}/chat/sessions/{session_id}")
def get_chat_session(
    course_id: UUID,
    session_id: UUID,
    user_id: UUID,
):
    session = chat_repository.get_session(
        str(session_id),
        str(user_id),
        str(course_id),
    )

    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    return {
        "session": session,
        "messages": chat_repository.get_session_messages(
            str(session_id),
            str(user_id),
        ),
    }