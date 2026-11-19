from app.db.client import get_db
from app.db.helpers import data_of


def get_topic(user_id, course_id, topic_id):

    result = (
        get_db()
        .table("syllabus_topics")
        .select("id, title, course_id, user_id")
        .eq("id", topic_id)
        .eq("course_id", course_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )

    return data_of(result)


def get_document(user_id, course_id, document_id):

    result = (
        get_db()
        .table("documents")
        .select("id, title, status, course_id, user_id")
        .eq("id", document_id)
        .eq("course_id", course_id)
        .eq("user_id", user_id)
        .eq("status", "ready")
        .maybe_single()
        .execute()
    )

    return data_of(result)


def get_document_chunks(user_id, course_id, document_id, limit=40):

    result = (
        get_db()
        .table("document_chunks")
        .select("content, page_start, page_end")
        .eq("document_id", document_id)
        .eq("course_id", course_id)
        .eq("user_id", user_id)
        .order("chunk_index")
        .limit(limit)
        .execute()
    )

    return result.data or []


def save_generated_question(
    user_id,
    course_id,
    topic_id,
    document_id,
    question,
    difficulty,
    model_name,
):

    db = get_db()

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
            "generation_model": model_name,
            "status": "active",
        })
        .select("id")
        .single()
        .execute()
    )

    question_id = result.data["id"]

    options = []

    for index, option in enumerate(question["options"], start=1):

        options.append({
            "question_id": question_id,
            "option_text": option["option_text"],
            "option_order": index,
            "is_correct": option["is_correct"],
        })

    db.table("question_options").insert(options).execute()

    return question_id


def list_questions(user_id, course_id, topic_id=None, limit=20):

    query = (
        get_db()
        .table("generated_questions")
        .select(
            "id, topic_id, source_document_id, question_text, "
            "explanation, difficulty, created_at"
        )
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .eq("status", "active")
    )

    if topic_id:
        query = query.eq("topic_id", topic_id)

    result = query.order("created_at", desc=True).limit(limit).execute()

    questions = result.data or []

    for question in questions:

        option_result = (
            get_db()
            .table("question_options")
            .select("id, option_text, option_order")
            .eq("question_id", question["id"])
            .order("option_order")
            .execute()
        )

        question["options"] = option_result.data or []

    return questions