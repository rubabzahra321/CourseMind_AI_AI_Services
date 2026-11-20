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
    source_page_start=None,
    source_page_end=None,
):
    db = get_db()

    result = (
        db.table("generated_questions")
        .insert({
            "user_id": user_id,
            "course_id": course_id,
            "topic_id": topic_id,
            "source_document_id": document_id,
            "source_page_start": source_page_start,
            "source_page_end": source_page_end,
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
            "id, topic_id, source_document_id, source_page_start, "
            "source_page_end, question_text, explanation, difficulty, created_at"
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

        question["source"] = get_source_reference(
            user_id=user_id,
            course_id=course_id,
            document_id=question.get("source_document_id"),
            page_start=question.get("source_page_start"),
            page_end=question.get("source_page_end"),
        )

    return questions


def get_source_reference(
    user_id,
    course_id,
    document_id,
    page_start=None,
    page_end=None,
):
    if not document_id:
        return None

    document_result = (
        get_db()
        .table("documents")
        .select("id, title, original_filename")
        .eq("id", document_id)
        .eq("course_id", course_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )

    document = data_of(document_result)

    if not document:
        return None

    source = {
        "document_id": document["id"],
        "document_title": (
            document.get("title")
            or document.get("original_filename")
            or "Source document"
        ),
        "page_start": page_start,
        "page_end": page_end,
    }

    # Retrieve a small source excerpt from the referenced page range.
    if page_start is not None:
        chunk_query = (
            get_db()
            .table("document_chunks")
            .select("content, page_start, page_end")
            .eq("document_id", document_id)
            .eq("course_id", course_id)
            .eq("user_id", user_id)
            .lte("page_start", page_end or page_start)
            .gte("page_end", page_start)
            .order("chunk_index")
            .limit(2)
        )

        chunk_result = chunk_query.execute()
        chunks = chunk_result.data or []

        source["excerpt"] = "\n\n".join(
            chunk.get("content", "")
            for chunk in chunks
            if chunk.get("content")
        )[:1200]

    return source