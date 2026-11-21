
import hashlib
import json

from datetime import datetime, timezone

from app.core.config import settings
from app.db.client import get_db
from app.db.helpers import data_of

from app.ai.question_generator import generate_mcqs

from app.repositories import (
    question_repository,
    practice_repository,
)


# =========================================================
# IDEMPOTENCY HELPERS
# =========================================================

class IdempotencyConflict(ValueError):
    """The same generation request is already being processed (HTTP 409)."""

def _get_idempotency_request(user_id, course_id, idempotency_key):
    result = (
        get_db()
        .table("ai_idempotency_requests")
        .select("*")
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .eq("idempotency_key", idempotency_key)
        .maybe_single()
        .execute()
    )

    return data_of(result)


def _create_idempotency_request(
    user_id,
    course_id,
    idempotency_key,
    request_hash,
):
    get_db().table("ai_idempotency_requests").insert({
        "user_id": user_id,
        "course_id": course_id,
        "idempotency_key": idempotency_key,
        "request_hash": request_hash,
        "status": "processing",
    }).execute()


def _update_idempotency_request(
    user_id,
    course_id,
    idempotency_key,
    values,
):
    (
        get_db()
        .table("ai_idempotency_requests")
        .update(values)
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .eq("idempotency_key", idempotency_key)
        .execute()
    )


# =========================================================
# GENERATE QUESTIONS WITH IDEMPOTENCY
# =========================================================

def generate_questions(
    user_id,
    course_id,
    topic_id,
    document_id,
    number_of_questions,
    difficulty,
    idempotency_key=None,
):
    request_hash = hashlib.sha256(
        json.dumps({
            "user_id": user_id,
            "course_id": course_id,
            "topic_id": topic_id,
            "document_id": document_id,
            "number_of_questions": number_of_questions,
            "difficulty": difficulty,
        }, sort_keys=True).encode()
    ).hexdigest()

    if idempotency_key:
        existing = _get_idempotency_request(
            user_id,
            course_id,
            idempotency_key,
        )

        if existing:
            if existing["request_hash"] != request_hash:
                raise ValueError(
                    "Idempotency key was already used for a different request"
                )

            if existing["status"] == "completed":
                return existing["response"]

            if existing["status"] == "processing":
                raise IdempotencyConflict(
                    "This question generation request is already processing"
                )

            _update_idempotency_request(
                user_id,
                course_id,
                idempotency_key,
                {"status": "processing", "response": None},
            )

        else:
            try:
                _create_idempotency_request(
                    user_id,
                    course_id,
                    idempotency_key,
                    request_hash,
                )
            except Exception:
                # Two identical requests arrived together and the other one
                # inserted the key first (unique violation). Treat it as
                # "already processing" instead of crashing with a 500.
                if _get_idempotency_request(
                    user_id, course_id, idempotency_key
                ):
                    raise IdempotencyConflict(
                        "This question generation request is already processing"
                    )
                raise

    saved_ids = []

    try:
        topic = question_repository.get_topic(
            user_id,
            course_id,
            topic_id,
        )

        if not topic:
            raise ValueError("Topic not found in this course")

        document = question_repository.get_document(
            user_id,
            course_id,
            document_id,
        )

        if not document:
            raise ValueError("Ready source document not found")

        chunks = question_repository.get_document_chunks(
            user_id,
            course_id,
            document_id,
        )

        if not chunks:
            raise ValueError("Source document has no extracted text")

        material = "\n\n".join(
            f"Page {chunk['page_start']}:\n{chunk['content']}"
            for chunk in chunks
        )

        material = material[:settings.mcq_context_max_chars]

        page_numbers = [
            chunk["page_start"]
            for chunk in chunks
            if chunk.get("page_start") is not None
        ]

        source_page_start = min(page_numbers) if page_numbers else None

        source_page_end_values = [
            chunk.get("page_end") or chunk.get("page_start")
            for chunk in chunks
            if chunk.get("page_end") is not None
            or chunk.get("page_start") is not None
        ]

        source_page_end = (
            max(source_page_end_values)
            if source_page_end_values
            else None
        )

        generated = generate_mcqs(
            topic_title=topic["title"],
            difficulty=difficulty,
            number_of_questions=number_of_questions,
            material=material,
        )

        for question in generated:
            question_id = question_repository.save_generated_question(
                user_id=user_id,
                course_id=course_id,
                topic_id=topic_id,
                document_id=document_id,
                question=question,
                difficulty=difficulty,
                model_name=settings.gemini_chat_model,
                source_page_start=source_page_start,
                source_page_end=source_page_end,
            )

            saved_ids.append(question_id)

        response = {
            "topic_id": topic_id,
            "document_id": document_id,
            "source_page_start": source_page_start,
            "source_page_end": source_page_end,
            "generated_count": len(saved_ids),
            "question_ids": saved_ids,
        }

        if idempotency_key:
            _update_idempotency_request(
                user_id,
                course_id,
                idempotency_key,
                {
                    "status": "completed",
                    "response": response,
                },
            )

        return response

    except Exception:
        # Remove questions saved before the failure so a retry cannot
        # leave duplicates behind (task 3: no duplicate questions).
        if saved_ids:
            try:
                question_repository.delete_generated_questions(
                    user_id, course_id, saved_ids
                )
            except Exception:
                pass

        if idempotency_key:
            _update_idempotency_request(
                user_id,
                course_id,
                idempotency_key,
                {"status": "failed"},
            )

        raise


# =========================================================
# GET QUESTIONS
# =========================================================

def get_questions(
    user_id,
    course_id,
    topic_id=None,
    limit=20,
):
    return question_repository.list_questions(
        user_id,
        course_id,
        topic_id,
        limit,
    )


# =========================================================
# START PRACTICE
# =========================================================

def start_practice(
    user_id,
    course_id,
    topic_id,
    question_ids,
):
    if not question_ids:
        raise ValueError("Select at least one question")

    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Duplicate questions are not allowed")

    if len(question_ids) > 50:
        raise ValueError("A session cannot exceed 50 questions")

    db_questions = []

    for question_id in question_ids:
        question = practice_repository.get_question(
            question_id,
            user_id,
            course_id,
        )

        if not question:
            raise ValueError("A selected question is unavailable")

        if topic_id and question["topic_id"] != topic_id:
            raise ValueError("A question does not belong to this topic")

        db_questions.append(question)

    session = practice_repository.create_session(
        user_id,
        course_id,
        topic_id,
        len(db_questions),
    )

    rows = [
        {
            "session_id": session["id"],
            "question_id": question["id"],
            "user_id": user_id,
            "question_order": index,
        }
        for index, question in enumerate(db_questions, start=1)
    ]

    get_db().table("practice_session_questions").insert(rows).execute()

    result_questions = []

    for question in db_questions:
        options = (
            get_db()
            .table("question_options")
            .select("id, option_text, option_order")
            .eq("question_id", question["id"])
            .order("option_order")
            .execute()
        )

        source = question_repository.get_source_reference(
            user_id=user_id,
            course_id=course_id,
            document_id=question.get("source_document_id"),
            page_start=question.get("source_page_start"),
            page_end=question.get("source_page_end"),
        )

        result_questions.append({
            "id": question["id"],
            "question_text": question["question_text"],
            "difficulty": question["difficulty"],
            "options": options.data or [],
            "source": source,
        })

    return {
        "session_id": session["id"],
        "total_questions": len(result_questions),
        "questions": result_questions,
    }


# =========================================================
# SUBMIT ANSWER
# =========================================================

def submit_answer(
    user_id,
    course_id,
    session_id,
    question_id,
    selected_option_id,
    response_time_seconds,
):
    session = practice_repository.get_session(
        session_id,
        user_id,
        course_id,
    )

    if not session:
        raise ValueError("Practice session not found")

    if session["status"] != "in_progress":
        raise ValueError("Practice session is not active")

    assigned = (
        get_db()
        .table("practice_session_questions")
        .select("id")
        .eq("session_id", session_id)
        .eq("question_id", question_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )

    if not data_of(assigned):
        raise ValueError("Question does not belong to this session")

    if practice_repository.has_attempt(session_id, question_id):
        raise ValueError("This question has already been answered")

    question = practice_repository.get_question(
        question_id,
        user_id,
        course_id,
    )

    if not question:
        raise ValueError("Question not found")

    option = practice_repository.get_option(
        selected_option_id,
        question_id,
    )

    if not option:
        raise ValueError("Selected option does not belong to this question")

    is_correct = bool(option["is_correct"])

    correct_option = (
        option
        if is_correct
        else practice_repository.get_correct_option(question_id)
    )

    attempt = practice_repository.save_attempt(
        session_id=session_id,
        question_id=question_id,
        selected_option_id=selected_option_id,
        user_id=user_id,
        is_correct=is_correct,
        response_time_seconds=response_time_seconds,
    )

    if question["topic_id"]:
        practice_repository.refresh_topic_progress(
            user_id,
            course_id,
            question["topic_id"],
        )
    else:
        practice_repository.refresh_topic_progress(
            user_id,
            course_id,
            None,
        )

    attempts = practice_repository.get_session_attempts(
        session_id,
        user_id,
    )

    total_answered = len(attempts)

    correct_count = sum(
        1 for item in attempts if item["is_correct"]
    )

    accuracy = round(
        (correct_count / total_answered) * 100,
        2
    ) if total_answered else 0

    completed = total_answered >= session["total_questions"]

    practice_repository.update_session(
        session_id,
        {
            "correct_answers": correct_count,
            "accuracy_percentage": accuracy,
            "status": "completed" if completed else "in_progress",
            "completed_at": (
                attempt["answered_at"] if completed else None
            ),
        }
    )

    source = question_repository.get_source_reference(
        user_id=user_id,
        course_id=course_id,
        document_id=question.get("source_document_id"),
        page_start=question.get("source_page_start"),
        page_end=question.get("source_page_end"),
    )

    # Give a useful next step after a wrong answer.
    mistake_follow_up = None

    if not is_correct:
        mistake_follow_up = {
            "topic_id": question.get("topic_id"),
            "action": "revise_topic",
            "title": "Revise this topic",
            "description": (
                "Review the linked source material, then practise "
                "another question from this topic."
            ),
            "recommended_next_step": (
                "Read the source pages and attempt another question "
                "from the same topic."
            ),
        }

    return {
        "attempt_id": attempt["id"],
        "is_correct": is_correct,
        "correct_option": (
            {
                "id": correct_option["id"],
                "option_text": correct_option["option_text"],
            }
            if correct_option else None
        ),
        "explanation": question["explanation"],
        "source": source,
        "mistake_follow_up": mistake_follow_up,
        "progress": {
            "answered": total_answered,
            "total": session["total_questions"],
            "correct": correct_count,
            "accuracy_percentage": accuracy,
            "completed": completed,
        },
    }


# =========================================================
# GET SESSION REPORT
# =========================================================

def get_session_report(user_id, course_id, session_id):
    session = practice_repository.get_session(
        session_id,
        user_id,
        course_id,
    )

    if not session:
        raise ValueError("Practice session not found")

    attempts = practice_repository.get_session_attempts(
        session_id,
        user_id,
    )

    attempt_question_ids = [
        attempt["question_id"]
        for attempt in attempts
    ]

    question_map = {}

    if attempt_question_ids:
        question_result = (
            get_db()
            .table("generated_questions")
            .select(
                "id, topic_id, question_text, explanation, "
                "source_document_id, source_page_start, source_page_end"
            )
            .eq("user_id", user_id)
            .eq("course_id", course_id)
            .in_("id", attempt_question_ids)
            .execute()
        )

        question_map = {
            question["id"]: question
            for question in (question_result.data or [])
        }

    topic_ids = list({
        question["topic_id"]
        for question in question_map.values()
        if question.get("topic_id")
    })

    topic_map = {}

    if topic_ids:
        topic_result = (
            get_db()
            .table("syllabus_topics")
            .select("id, title")
            .eq("user_id", user_id)
            .eq("course_id", course_id)
            .in_("id", topic_ids)
            .execute()
        )

        topic_map = {
            topic["id"]: topic["title"]
            for topic in (topic_result.data or [])
        }

    correct_count = sum(
        1 for attempt in attempts if attempt["is_correct"]
    )

    incorrect_count = len(attempts) - correct_count

    accuracy = round(
        (correct_count / len(attempts)) * 100,
        2
    ) if attempts else 0

    response_times = [
        attempt["response_time_seconds"]
        for attempt in attempts
        if attempt.get("response_time_seconds") is not None
    ]

    average_response_time = (
        round(sum(response_times) / len(response_times), 2)
        if response_times else None
    )

    topic_breakdown = {}

    incorrect_questions = []

    for attempt in attempts:
        question = question_map.get(attempt["question_id"])

        if not question:
            continue

        topic_id = question.get("topic_id")
        topic_title = topic_map.get(topic_id, "Unassigned topic")

        if topic_id not in topic_breakdown:
            topic_breakdown[topic_id] = {
                "topic_id": topic_id,
                "topic_title": topic_title,
                "answered": 0,
                "correct": 0,
                "incorrect": 0,
            }

        topic_breakdown[topic_id]["answered"] += 1

        if attempt["is_correct"]:
            topic_breakdown[topic_id]["correct"] += 1
        else:
            topic_breakdown[topic_id]["incorrect"] += 1

            source = question_repository.get_source_reference(
                user_id=user_id,
                course_id=course_id,
                document_id=question.get("source_document_id"),
                page_start=question.get("source_page_start"),
                page_end=question.get("source_page_end"),
            )

            incorrect_questions.append({
                "question_id": question["id"],
                "question_text": question["question_text"],
                "explanation": question["explanation"],
                "topic_id": topic_id,
                "topic_title": topic_title,
                "source": source,
                "follow_up": {
                    "action": "revise_topic",
                    "title": f"Revise {topic_title}",
                    "description": (
                        "Review the source material and practise "
                        "another question from this topic."
                    ),
                },
            })

    for topic in topic_breakdown.values():
        topic["accuracy_percentage"] = round(
            (topic["correct"] / topic["answered"]) * 100,
            2
        ) if topic["answered"] else 0

    follow_up_tasks = []

    for topic in topic_breakdown.values():
        if topic["incorrect"] > 0:
            follow_up_tasks.append({
                "topic_id": topic["topic_id"],
                "topic_title": topic["topic_title"],
                "title": f"Revise {topic['topic_title']}",
                "description": (
                    f"You answered {topic['incorrect']} question(s) "
                    "incorrectly in this topic. Review the source "
                    "material and practise this topic again."
                ),
                "reason": "incorrect_answers",
            })

    return {
        "session": {
            "id": session["id"],
            "status": session["status"],
            "topic_id": session.get("topic_id"),
            "total_questions": session["total_questions"],
            "started_at": session["started_at"],
            "completed_at": session.get("completed_at"),
        },
        "score": {
            "answered": len(attempts),
            "total_questions": session["total_questions"],
            "correct": correct_count,
            "incorrect": incorrect_count,
            "unanswered": max(
                session["total_questions"] - len(attempts),
                0
            ),
            "accuracy_percentage": accuracy,
            "average_response_time_seconds": average_response_time,
        },
        "topic_breakdown": list(topic_breakdown.values()),
        "incorrect_questions": incorrect_questions,
        "follow_up_tasks": follow_up_tasks,
    }
