
from uuid import UUID
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException

from app.core.security import require_internal_key
from app.services import quiz_duel_service


router = APIRouter(
    prefix="/internal/courses/{course_id}/quiz-duel",
    dependencies=[Depends(require_internal_key)],
)


class StartGameRequest(BaseModel):
    user_id: UUID
    topic_id: UUID


class AnswerRequest(BaseModel):
    user_id: UUID
    question_id: UUID
    selected_option_id: UUID
    position: int = Field(ge=0, le=8)


@router.post("/start")
def start_game(course_id: UUID, payload: StartGameRequest):
    try:
        return quiz_duel_service.start_game(
            payload.user_id, course_id, payload.topic_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/{game_id}/answer")
def submit_answer(
    course_id: UUID,
    game_id: UUID,
    payload: AnswerRequest,
):
    try:
        return quiz_duel_service.submit_game_answer(
            payload.user_id,
            course_id,
            game_id,
            payload.question_id,
            payload.selected_option_id,
            payload.position,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
