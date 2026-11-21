
from app.core.config import settings
from app.db.client import get_db
from app.db.helpers import data_of

from app.ai.question_formats_generator import (
    generate_practice_questions,
)

from app.repositories import (
    question_repository,
    practice_repository,
)


SUPPORTED_TYPES = {
    "mcq",
    "true_false",
    "short_answer",
    "long_answer",
}


def generate_questions(
    user_id,
    course_id,
    topic_id,
    document_id,
    question_type,
    number_of_questions,
    difficulty,
):
    if question_type not in SUPPORTED_TYPES:
        raise ValueError("Unsupported question type")

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
        f"Page {chunk.get('page_start') or 'Unknown'}:\n"
        f"{chunk['content']}"
        for chunk in chunks
    )

    material = material[:settings.mcq_context_max_chars]

    generated = generate_practice_questions(
        topic_title=topic["title"],
        question_type=question_type,
        difficulty=difficulty,
        number_of_questions=number_of_questions,
        material=material,
    )

    db = get_db()
    saved_ids = []

    for question in generated:
        result = (
            db.table("generated_questions")
            .insert({
                "user_id": user_id,
                "course_id": course_id,
                "topic_id": topic_id,
                "source_document_id": document_id,
                "question_text": question["question_text"],
                "explanation": question["explanation"],
                "difficulty": difficulty,
                "question_type": question_type,
                "expected_answer": question.get("expected_answer"),
                "answer_rubric": question.get("answer_rubric"),
                "max_marks": question.get("max_marks"),
                "generation_model": settings.gemini_chat_model,
                "status": "active",
            })
            .select("id")
            .single()
            .execute()
        )

        question_id = result.data["id"]

        options = question.get("options") or []

        if options:
            option_rows = []

            for index, option in enumerate(options, start=1):
                option_rows.append({
                    "question_id": question_id,
                    "option_text": option["option_text"],
                    "option_order": index,
                    "is_correct": option["is_correct"],
                })

            db.table("question_options").insert(option_rows).execute()

        saved_ids.append(question_id)

    return {
        "topic_id": topic_id,
        "document_id": document_id,
        "question_type": question_type,
        "difficulty": difficulty,
        "generated_count": len(saved_ids),
        "question_ids": saved_ids,
    }


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

        if topic_id and question.get("topic_id") != topic_id:
            raise ValueError("A question does not belong to this topic")

        db_questions.append(question)

    session = practice_repository.create_session(
        user_id,
        course_id,
        topic_id,
        len(db_questions),
    )

    rows = []

    for index, question in enumerate(db_questions, start=1):
        rows.append({
            "session_id": session["id"],
            "question_id": question["id"],
            "user_id": user_id,
            "question_order": index,
        })

    db = get_db()
    db.table("practice_session_questions").insert(rows).execute()

    result_questions = []

    for question in db_questions:
        options_result = (
            db.table("question_options")
            .select("id, option_text, option_order")
            .eq("question_id", question["id"])
            .order("option_order")
            .execute()
        )

        # Do not expose correct answers or written model answers
        # before the student submits a response.
        result_questions.append({
            "id": question["id"],
            "question_text": question["question_text"],
            "question_type": question.get("question_type") or "mcq",
            "difficulty": question["difficulty"],
            "max_marks": question.get("max_marks"),
            "options": options_result.data or [],
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
    selected_option_id=None,
    response_text=None,
    response_time_seconds=None,
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

    db = get_db()

    assigned = (
        db.table("practice_session_questions")
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

    question_type = question.get("question_type") or "mcq"

    if question_type in {"mcq", "true_false"}:
        if not selected_option_id:
            raise ValueError("Select an answer option")

        if response_text and response_text.strip():
            raise ValueError(
                "Do not send written text for an objective question"
            )

        option = practice_repository.get_option(
            selected_option_id,
            question_id,
        )

        if not option:
            raise ValueError(
                "Selected option does not belong to this question"
            )

        is_correct = bool(option["is_correct"])
        grading_status = "graded"
        correct_option = (
            option
            if is_correct
            else practice_repository.get_correct_option(question_id)
        )

        saved_response = option["option_text"]

    elif question_type in {"short_answer", "long_answer"}:
        if selected_option_id:
            raise ValueError(
                "Written questions do not accept option IDs"
            )

        if not response_text or not response_text.strip():
            raise ValueError("Write an answer before submitting")

        is_correct = None
        grading_status = "pending_review"
        correct_option = None
        saved_response = response_text.strip()

    else:
        raise ValueError("Unsupported question type")

    attempt_result = (
        db.table("practice_attempts")
        .insert({
            "session_id": session_id,
            "question_id": question_id,
            "selected_option_id": selected_option_id,
            "user_id": user_id,
            "is_correct": is_correct,
            "response_text": saved_response,
            "grading_status": grading_status,
            "response_time_seconds": response_time_seconds,
        })
        .select("*")
        .single()
        .execute()
    )

    attempt = attempt_result.data

    # Refresh topic progress after the attempt is inserted.
    if question.get("topic_id"):
        practice_repository.refresh_topic_progress(
            user_id,
            course_id,
            question["topic_id"],
        )

    attempts = practice_repository.get_session_attempts(
        session_id,
        user_id,
    )

    total_answered = len(attempts)

    graded_attempts = [
        item
        for item in attempts
        if item.get("is_correct") is not None
    ]

    correct_count = sum(
        1 for item in graded_attempts if item["is_correct"]
    )

    accuracy = round(
        (correct_count / len(graded_attempts)) * 100,
        2,
    ) if graded_attempts else None

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
        },
    )

    result = {
        "attempt_id": attempt["id"],
        "question_type": question_type,
        "grading_status": grading_status,
        "is_correct": is_correct,
        "progress": {
            "answered": total_answered,
            "total": session["total_questions"],
            "graded_answers": len(graded_attempts),
            "correct": correct_count,
            "accuracy_percentage": accuracy,
            "completed": completed,
        },
    }

    if question_type in {"mcq", "true_false"}:
        result["correct_option"] = (
            {
                "id": correct_option["id"],
                "option_text": correct_option["option_text"],
            }
            if correct_option else None
        )

        result["explanation"] = question["explanation"]

    else:
        # Show the model answer and rubric only after submission.
        result["response_text"] = saved_response
        result["expected_answer"] = question.get("expected_answer")
        result["answer_rubric"] = question.get("answer_rubric")
        result["max_marks"] = question.get("max_marks")
        result["message"] = (
            "Your written answer has been saved and is awaiting review."
        )

    return result
