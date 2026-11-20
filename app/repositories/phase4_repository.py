
from app.db.client import get_db
from app.db.helpers import data_of


def list_practice_sessions(user_id, course_id, limit=50):
    result = (
        get_db()
        .table("practice_sessions")
        .select("*")
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .order("started_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def get_practice_session(session_id, user_id, course_id):
    result = (
        get_db()
        .table("practice_sessions")
        .select("*")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .maybe_single()
        .execute()
    )
    return data_of(result)


def get_session_attempts(session_id, user_id):
    result = (
        get_db()
        .table("practice_attempts")
        .select("*")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .order("answered_at")
        .execute()
    )
    return result.data or []


def get_generated_question(question_id, user_id, course_id):
    result = (
        get_db()
        .table("generated_questions")
        .select("*")
        .eq("id", question_id)
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .maybe_single()
        .execute()
    )
    return data_of(result)


def get_past_paper_question(question_id, user_id, course_id):
    result = (
        get_db()
        .table("past_paper_questions")
        .select("*")
        .eq("id", question_id)
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .eq("review_status", "reviewed")
        .maybe_single()
        .execute()
    )
    return data_of(result)


def list_reviewed_paper_questions(user_id, course_id, topic_id=None, limit=50):
    query = (
        get_db()
        .table("past_paper_questions")
        .select("*")
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .eq("review_status", "reviewed")
    )

    result = (
        query
        .order("paper_year", desc=True)
        .limit(limit)
        .execute()
    )

    questions = result.data or []

    if topic_id:
        mapped_ids = (
            get_db()
            .table("question_topic_mappings")
            .select("past_paper_question_id")
            .eq("user_id", user_id)
            .eq("topic_id", topic_id)
            .execute()
        )

        allowed = {
            row["past_paper_question_id"]
            for row in (mapped_ids.data or [])
        }

        questions = [
            question for question in questions
            if question["id"] in allowed
        ]

    for question in questions:
        document_result = (
            get_db()
            .table("documents")
            .select("id, title")
            .eq("id", question["document_id"])
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )

        document = data_of(document_result)
        question["source_document"] = document

        mapping_result = (
            get_db()
            .table("question_topic_mappings")
            .select("topic_id, mapping_method, confidence")
            .eq("past_paper_question_id", question["id"])
            .eq("user_id", user_id)
            .execute()
        )
        question["topic_mappings"] = mapping_result.data or []

    return questions


def get_paper_topic_ids(question_id, user_id):
    result = (
        get_db()
        .table("question_topic_mappings")
        .select("topic_id")
        .eq("past_paper_question_id", question_id)
        .eq("user_id", user_id)
        .execute()
    )
    return [row["topic_id"] for row in (result.data or [])]


def create_practice_session(
    user_id,
    course_id,
    topic_id,
    total_questions,
):
    result = (
        get_db()
        .table("practice_sessions")
        .insert({
            "user_id": user_id,
            "course_id": course_id,
            "topic_id": topic_id,
            "total_questions": total_questions,
            "correct_answers": 0,
            "status": "in_progress",
        })
        .select("*")
        .single()
        .execute()
    )
    return result.data


def add_session_questions(rows):
    if rows:
        get_db().table("practice_session_questions").insert(rows).execute()


def is_paper_question_assigned(session_id, question_id, user_id):
    result = (
        get_db()
        .table("practice_session_questions")
        .select("id")
        .eq("session_id", session_id)
        .eq("past_paper_question_id", question_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    return data_of(result) is not None


def is_generated_question_assigned(session_id, question_id, user_id):
    result = (
        get_db()
        .table("practice_session_questions")
        .select("id")
        .eq("session_id", session_id)
        .eq("question_id", question_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    return data_of(result) is not None


def has_paper_attempt(session_id, question_id):
    result = (
        get_db()
        .table("practice_attempts")
        .select("id")
        .eq("session_id", session_id)
        .eq("past_paper_question_id", question_id)
        .maybe_single()
        .execute()
    )
    return data_of(result) is not None


def save_paper_attempt(
    session_id,
    question_id,
    user_id,
    text_answer,
    response_time_seconds,
):
    result = (
        get_db()
        .table("practice_attempts")
        .insert({
            "session_id": session_id,
            "past_paper_question_id": question_id,
            "question_id": None,
            "selected_option_id": None,
            "user_id": user_id,
            "is_correct": None,
            "text_answer": text_answer,
            "response_time_seconds": response_time_seconds,
            "evaluation_feedback": (
                "This is a reviewed past-paper question. "
                "It has been saved for review; no official answer "
                "key is stored for automatic marking."
            ),
        })
        .select("*")
        .single()
        .execute()
    )
    return result.data


def create_follow_up_task(
    user_id,
    course_id,
    topic_id,
    session_id,
    attempt_id,
    title,
    description,
    task_type="revisit_mistake",
):
    result = (
        get_db()
        .table("practice_follow_up_tasks")
        .insert({
            "user_id": user_id,
            "course_id": course_id,
            "topic_id": topic_id,
            "practice_session_id": session_id,
            "attempt_id": attempt_id,
            "title": title,
            "description": description,
            "task_type": task_type,
        })
        .select("*")
        .single()
        .execute()
    )
    return result.data


def list_incorrect_attempts(user_id, course_id, limit=100):
    result = (
        get_db()
        .table("practice_attempts")
        .select(
            "id, session_id, question_id, past_paper_question_id, "
            "selected_option_id, is_correct, text_answer, "
            "evaluation_feedback, response_time_seconds, answered_at"
        )
        .eq("user_id", user_id)
        .eq("is_correct", False)
        .order("answered_at", desc=True)
        .limit(limit)
        .execute()
    )

    attempts = result.data or []
    output = []

    for attempt in attempts:
        session = get_practice_session(
            attempt["session_id"],
            user_id,
            course_id,
        )
        if not session:
            continue

        question = None
        if attempt.get("question_id"):
            question = get_generated_question(
                attempt["question_id"],
                user_id,
                course_id,
            )

        output.append({
            "attempt": attempt,
            "session": session,
            "question": question,
        })

    return output


def get_topic_title(topic_id, user_id, course_id):
    if not topic_id:
        return None

    result = (
        get_db()
        .table("syllabus_topics")
        .select("id, title")
        .eq("id", topic_id)
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .maybe_single()
        .execute()
    )
    return data_of(result)


def get_follow_up_tasks(session_id, user_id):
    result = (
        get_db()
        .table("practice_follow_up_tasks")
        .select("*")
        .eq("practice_session_id", session_id)
        .eq("user_id", user_id)
        .order("created_at")
        .execute()
    )
    return result.data or []


def list_all_follow_up_tasks(user_id, course_id, session_id=None):
    query = (
        get_db()
        .table("practice_follow_up_tasks")
        .select("*")
        .eq("user_id", user_id)
        .eq("course_id", course_id)
    )

    if session_id:
        query = query.eq("practice_session_id", session_id)

    result = query.order("created_at", desc=True).execute()
    return result.data or []


def get_document(document_id, user_id, course_id):
    result = (
        get_db()
        .table("documents")
        .select("id, title, document_type")
        .eq("id", document_id)
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .maybe_single()
        .execute()
    )
    return data_of(result)

def save_typed_question(
    user_id,
    course_id,
    topic_id,
    document_id,
    question_text,
    question_type,
    difficulty,
    expected_answer,
    answer_rubric,
    max_marks,
    source_page_start=None,
    source_page_end=None,
):
    result = (
        get_db()
        .table("generated_questions")
        .insert({
            "user_id": user_id,
            "course_id": course_id,
            "topic_id": topic_id,
            "source_document_id": document_id,
            "question_text": question_text,
            "question_type": question_type,
            "difficulty": difficulty,
            "expected_answer": expected_answer,
            "answer_rubric": answer_rubric,
            "max_marks": max_marks,
            "source_page_start": source_page_start,
            "source_page_end": source_page_end,
            "explanation": answer_rubric or expected_answer,
            "generation_model": "gemini",
            "status": "active",
        })
        .select("*")
        .single()
        .execute()
    )
    return result.data


def save_typed_options(question_id, options):
    rows = []

    for index, option in enumerate(options, start=1):
        rows.append({
            "question_id": question_id,
            "option_text": option["option_text"],
            "option_order": index,
            "is_correct": bool(option["is_correct"]),
        })

    if rows:
        get_db().table("question_options").insert(rows).execute()


def get_generated_options(question_id):
    result = (
        get_db()
        .table("question_options")
        .select("id, option_text, option_order")
        .eq("question_id", question_id)
        .order("option_order")
        .execute()
    )
    return result.data or []


def save_text_attempt(
    session_id,
    question_id,
    user_id,
    text_answer,
    response_time_seconds,
    is_correct,
    awarded_marks,
    max_marks,
    evaluation_feedback,
):
    result = (
        get_db()
        .table("practice_attempts")
        .insert({
            "session_id": session_id,
            "question_id": question_id,
            "past_paper_question_id": None,
            "selected_option_id": None,
            "user_id": user_id,
            "is_correct": is_correct,
            "text_answer": text_answer,
            "response_time_seconds": response_time_seconds,
            "awarded_marks": awarded_marks,
            "max_marks": max_marks,
            "evaluation_feedback": evaluation_feedback,
        })
        .select("*")
        .single()
        .execute()
    )
    return result.data


def get_question_source_chunks(document_id, user_id, course_id, page_start=None):
    query = (
        get_db()
        .table("document_chunks")
        .select("id, document_id, content, page_start, page_end")
        .eq("document_id", document_id)
        .eq("user_id", user_id)
        .eq("course_id", course_id)
    )

    if page_start is not None:
        query = query.eq("page_start", page_start)

    result = query.order("chunk_index").limit(5).execute()
    return result.data or []


def update_practice_session(session_id, fields):
    return (
        get_db()
        .table("practice_sessions")
        .update(fields)
        .eq("id", session_id)
        .execute()
    )


def get_session_question_ids(session_id, user_id):
    result = (
        get_db()
        .table("practice_session_questions")
        .select("question_id, past_paper_question_id")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data or []
