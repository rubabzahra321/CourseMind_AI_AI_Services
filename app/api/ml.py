from typing import Literal, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.security import require_internal_key
from app.services import difficulty_service, ml_service, performance_service


router = APIRouter(
    prefix="/internal",
    dependencies=[Depends(require_internal_key)]
)


class TrainBody(BaseModel):
    user_id: UUID

    # Also learn from passages the AI linked to topics (default: hand-made
    # links only, so the model never learns from its own guesses).
    include_ai_links: bool = False

    # Activate the new version even if it scores lower than the current one.
    force: bool = False


class ClassifyBody(BaseModel):
    user_id: UUID

    # Specific questions; by default every question that has no topic yet.
    question_ids: Optional[list[UUID]] = Field(default=None, max_length=200)

    # Save confident suggestions as `ai_suggested` (pending questions only).
    save: bool = False

    min_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    limit: Optional[int] = Field(default=None, ge=1, le=200)


@router.post("/courses/{course_id}/ml/topic-classifier/train")
def train_topic_classifier(course_id: UUID, body: TrainBody):
    try:
        return ml_service.train_topic_classifier(
            body.user_id,
            course_id,
            include_ai_links=body.include_ai_links,
            force=body.force,
        )
    except ml_service.InsufficientTrainingData as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/courses/{course_id}/ml/status")
def ml_status(course_id: UUID, user_id: UUID):
    return ml_service.get_status(user_id, course_id)


@router.post("/courses/{course_id}/ml/classify-questions")
def classify_questions(course_id: UUID, body: ClassifyBody):
    return ml_service.classify_questions(
        body.user_id,
        course_id,
        question_ids=body.question_ids,
        save=body.save,
        min_confidence=body.min_confidence,
        limit=body.limit,
    )


# ---------------------------------------------------------------- difficulty
class DifficultyTrainBody(BaseModel):
    user_id: UUID

    # Activate the new version even if it scores lower than the current one.
    force: bool = False


class DifficultyPredictBody(BaseModel):
    user_id: UUID

    # Specific past-paper questions; by default those without a prediction.
    question_ids: Optional[list[UUID]] = Field(default=None, max_length=200)

    # Save confident predictions on the questions.
    save: bool = False

    min_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    limit: Optional[int] = Field(default=None, ge=1, le=200)


@router.post("/courses/{course_id}/ml/difficulty/train")
def train_difficulty(course_id: UUID, body: DifficultyTrainBody):
    try:
        return difficulty_service.train_difficulty_model(
            body.user_id,
            course_id,
            force=body.force,
        )
    except ml_service.InsufficientTrainingData as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/courses/{course_id}/ml/difficulty/status")
def difficulty_status(course_id: UUID, user_id: UUID):
    return difficulty_service.get_status(user_id, course_id)


@router.post("/courses/{course_id}/ml/difficulty/predict")
def predict_difficulty(course_id: UUID, body: DifficultyPredictBody):
    return difficulty_service.predict_difficulty(
        body.user_id,
        course_id,
        question_ids=body.question_ids,
        save=body.save,
        min_confidence=body.min_confidence,
        limit=body.limit,
    )


# ---------------------------------------------------------------- performance
class PerformanceTrainBody(BaseModel):
    # Activate the new version even if it scores lower than the current one.
    force: bool = False


@router.post("/ml/performance/train")
def train_performance(body: PerformanceTrainBody):
    # Trains ONE global model from every student's pooled, anonymised answers.
    try:
        return performance_service.train_performance_model(force=body.force)
    except ml_service.InsufficientTrainingData as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/ml/performance/status")
def performance_status():
    return performance_service.get_status()


@router.get("/courses/{course_id}/ml/readiness")
def readiness(
    course_id: UUID,
    user_id: UUID,
    difficulty: Literal["easy", "medium", "hard"] = "medium",
):
    return performance_service.student_readiness(
        user_id, course_id, difficulty)
