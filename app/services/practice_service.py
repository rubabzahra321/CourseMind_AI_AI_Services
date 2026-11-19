
from app.core.config import settings
from app.db.helpers import data_of

from app.ai.question_generator import generate_mcqs

from app.repositories import (
    question_repository,
    practice_repository,
)


def generate_questions(
    user_id,
    course_id,
    topic_id,
    document_id,
    number_of_questions,
    difficulty,
):
    topic = question_repository.get_topic(
        user_id,
        course_id,
        topic_id
    )

    if not topic:
        raise ValueError("Topic not found in this course")

    document = question_repository.get_document(
        user_id,
        course_id,
        document_id
    )

    if not document:
        raise ValueError("Ready source document not found")

    chunks = question_repository.get_document_chunks(
        user_id,
        course_id,
        document_id
    )

    if not chunks:
        raise ValueError("Source document has no extracted text")

    material = "\n\n".join(
        f"Page {chunk['page_start']}:\n{chunk['content']}"
        for chunk in chunks
    )

    material = material[:settings.mcq_context_max_chars]

    generated = generate_mcqs(
        topic_title=topic["title"],
        difficulty=difficulty,
        number_of_questions=number_of_questions,
        material=material,
    )

    saved_ids = []

    for question in generated:
        question_id = question_repository.save_generated_question(
            user_id=user_id,
            course_id=course_id,
            topic_id=topic_id,
            document_id=document_id,
            question=question,
            difficulty=difficulty,
            model_name=settings.gemini_chat_model,
        )

        saved_ids.append(question_id)

    return {
        "topic_id": topic_id,
        "document_id": document_id,
        "generated_count": len(saved_ids),
        "question_ids": saved_ids,
    }


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


def start_practice(
    user_id,
    course_id,
    topic_id,
    question_ids,
):
    if not question_ids:
        raise ValueError("Select at least one question")

    # Reject duplicate IDs before processing questions.
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

    # Persist the exact question set and order.
    from app.db.client import get_db

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

    # Return question options without answer keys.
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

        result_questions.append({
            "id": question["id"],
            "question_text": question["question_text"],
            "difficulty": question["difficulty"],
            "options": options.data or [],
        })

    return {
        "session_id": session["id"],
        "total_questions": len(result_questions),
        "questions": result_questions,
    }


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

    # Verify that this question was assigned to this session.
    from app.db.client import get_db

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

    # Show the student the REAL answer, even when they picked a wrong option.
    correct_option = option if is_correct else practice_repository.get_correct_option(question_id)

    attempt = practice_repository.save_attempt(
        session_id=session_id,
        question_id=question_id,
        selected_option_id=selected_option_id,
        user_id=user_id,
        is_correct=is_correct,
        response_time_seconds=response_time_seconds,
    )

    # Refresh progress immediately after saving the attempt.
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

    # Recalculate session totals from persisted attempts.
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

    return {
        "attempt_id": attempt["id"],
        "is_correct": is_correct,
        "correct_option": (
            {"id": correct_option["id"], "option_text": correct_option["option_text"]}
            if correct_option else None
        ),
        "explanation": question["explanation"],
        "progress": {
            "answered": total_answered,
            "total": session["total_questions"],
            "correct": correct_count,
            "accuracy_percentage": accuracy,
            "completed": completed,
        },
    }
