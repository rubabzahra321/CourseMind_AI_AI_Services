from app.db.client import get_db


def create_session(
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


def get_session(session_id, user_id, course_id):

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

    return result.data


def get_question(question_id, user_id, course_id):

    result = (
        get_db()
        .table("generated_questions")
        .select("*")
        .eq("id", question_id)
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .eq("status", "active")
        .maybe_single()
        .execute()
    )

    return result.data


def get_option(option_id, question_id):

    result = (
        get_db()
        .table("question_options")
        .select("id, question_id, option_text, is_correct")
        .eq("id", option_id)
        .eq("question_id", question_id)
        .maybe_single()
        .execute()
    )

    return result.data


def has_attempt(session_id, question_id):

    result = (
        get_db()
        .table("practice_attempts")
        .select("id")
        .eq("session_id", session_id)
        .eq("question_id", question_id)
        .maybe_single()
        .execute()
    )

    return result.data is not None


def save_attempt(
    session_id,
    question_id,
    selected_option_id,
    user_id,
    is_correct,
    response_time_seconds,
):

    result = (
        get_db()
        .table("practice_attempts")
        .insert({
            "session_id": session_id,
            "question_id": question_id,
            "selected_option_id": selected_option_id,
            "user_id": user_id,
            "is_correct": is_correct,
            "response_time_seconds": response_time_seconds,
        })
        .select("*")
        .single()
        .execute()
    )

    return result.data


def refresh_topic_progress(user_id, course_id, topic_id):

    result = get_db().rpc(
        "refresh_topic_progress",
        {
            "p_user_id": user_id,
            "p_course_id": course_id,
            "p_topic_id": topic_id,
        }
    ).execute()

    return result.data


def update_session(session_id, fields):

    return (
        get_db()
        .table("practice_sessions")
        .update(fields)
        .eq("id", session_id)
        .execute()
    )


def get_session_attempts(session_id, user_id):

    result = (
        get_db()
        .table("practice_attempts")
        .select(
            "id, question_id, selected_option_id, is_correct, "
            "response_time_seconds, answered_at"
        )
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .order("answered_at")
        .execute()
    )

    return result.data or []