
from app.db.client import get_db
from app.db.helpers import data_of


# --------------------------------------------------
# GET TOPIC
# --------------------------------------------------

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


# --------------------------------------------------
# GET DOCUMENT
# --------------------------------------------------

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


# --------------------------------------------------
# GET DOCUMENT CHUNKS
# --------------------------------------------------

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


# --------------------------------------------------
# SAVE GENERATED QUESTION
# --------------------------------------------------

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

    # Get the actual document chunks to validate page references.
    chunks = get_document_chunks(
        user_id,
        course_id,
        document_id
    )

    # Collect valid starting and ending pages separately.
    available_start_pages = {
        chunk.get("page_start")
        for chunk in chunks
        if chunk.get("page_start") is not None
    }

    available_end_pages = {
        chunk.get("page_end")
        for chunk in chunks
        if chunk.get("page_end") is not None
    }

    # Do not save page references that do not exist in the document.
    if source_page_start not in available_start_pages:
        source_page_start = None

    if source_page_end not in available_end_pages:
        source_page_end = None

    # Ensure the ending page is not before the starting page.
    if (
        source_page_start is not None
        and source_page_end is not None
        and source_page_end < source_page_start
    ):
        source_page_start = None
        source_page_end = None

    # Insert the generated question.
    result = (
        db.table("generated_questions")
        .insert({
            "user_id": user_id,
            "course_id": course_id,
            "topic_id": topic_id,
            "source_document_id": document_id,

            "question_text": question["question_text"],
            "explanation": question.get("explanation") or "",

            "expected_answer": question.get("expected_answer"),
            "answer_rubric": question.get("answer_rubric"),

            "difficulty": difficulty,
            "question_type": question.get("question_type", "mcq"),
            "max_marks": question.get("max_marks", 1),

            "source_page_start": source_page_start,
            "source_page_end": source_page_end,

            "generation_model": model_name,
            "status": "active",
        })
        .select("id")
        .single()
        .execute()
    )

    question_id = result.data["id"]

    # Save options when the generated question has options.
    options = []

    for index, option in enumerate(
        question.get("options", []),
        start=1
    ):

        options.append({
            "question_id": question_id,
            "option_text": option["option_text"],
            "option_order": index,
            "is_correct": option["is_correct"],
        })

    if options:
        db.table("question_options").insert(options).execute()

    return question_id


# --------------------------------------------------
# LIST GENERATED QUESTIONS
# --------------------------------------------------

def list_questions(
    user_id,
    course_id,
    topic_id=None,
    limit=20
):

    query = (
        get_db()
        .table("generated_questions")
        .select(
            "id, topic_id, source_document_id, "
            "question_text, explanation, difficulty, "
            "question_type, max_marks, "
            "expected_answer, answer_rubric, "
            "source_page_start, source_page_end, "
            "created_at"
        )
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .eq("status", "active")
    )

    if topic_id:
        query = query.eq("topic_id", topic_id)

    result = (
        query
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )

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

        # Never expose the correct option before submission.
        question["options"] = option_result.data or []

    return questions
