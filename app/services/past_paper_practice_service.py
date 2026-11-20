
from app.db.client import get_db
from app.db.helpers import data_of
from datetime import datetime, timezone


def list_past_paper_questions(
    user_id,
    course_id,
    topic_id=None,
    limit=20,
):
    """
    Return reviewed past-paper questions only.

    Questions must belong to the current user and course.
    If topic_id is supplied, only mapped questions for that
    topic are returned.
    """

    query = (
        get_db()
        .table("past_paper_questions")
        .select(
            "id, course_id, user_id, document_id, question_number, "
            "question_text, question_type, marks, paper_year, "
            "page_number, review_status"
        )
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .eq("review_status", "reviewed")
    )

    result = query.order(
        "created_at",
        desc=True
    ).limit(limit).execute()

    questions = result.data or []

    # Filter by the selected topic using the existing mapping table.
    if topic_id:
        mappings = (
            get_db()
            .table("question_topic_mappings")
            .select("past_paper_question_id")
            .eq("user_id", user_id)
            .eq("topic_id", topic_id)
            .execute()
        )

        mapped_ids = {
            row["past_paper_question_id"]
            for row in (mappings.data or [])
        }

        questions = [
            question
            for question in questions
            if question["id"] in mapped_ids
        ]

    # Get source document titles.
    document_ids = list({
        question["document_id"]
        for question in questions
    })

    document_titles = {}

    if document_ids:
        documents = (
            get_db()
            .table("documents")
            .select("id, title")
            .eq("user_id", user_id)
            .eq("course_id", course_id)
            .in_("id", document_ids)
            .execute()
        )

        document_titles = {
            document["id"]: document["title"]
            for document in (documents.data or [])
        }

    # Add the topic mappings for each question.
    question_ids = [
        question["id"]
        for question in questions
    ]

    topic_titles = {}

    if question_ids:
        mappings = (
            get_db()
            .table("question_topic_mappings")
            .select("past_paper_question_id, topic_id")
            .eq("user_id", user_id)
            .in_("past_paper_question_id", question_ids)
            .execute()
        )

        mapped_topic_ids = list({
            row["topic_id"]
            for row in (mappings.data or [])
        })

        if mapped_topic_ids:
            topics = (
                get_db()
                .table("syllabus_topics")
                .select("id, title")
                .eq("user_id", user_id)
                .eq("course_id", course_id)
                .in_("id", mapped_topic_ids)
                .execute()
            )

            topic_titles = {
                topic["id"]: topic["title"]
                for topic in (topics.data or [])
            }

        question_topics = {}

        for mapping in (mappings.data or []):
            question_topics.setdefault(
                mapping["past_paper_question_id"],
                []
            ).append({
                "id": mapping["topic_id"],
                "title": topic_titles.get(mapping["topic_id"]),
            })

    else:
        question_topics = {}

    # Build a clean response.
    output = []

    for question in questions:
        output.append({
            "id": question["id"],
            "question_number": question["question_number"],
            "question_text": question["question_text"],
            "question_type": question["question_type"],
            "marks": question["marks"],
            "paper_year": question["paper_year"],
            "page_number": question["page_number"],
            "document_id": question["document_id"],
            "document_title": document_titles.get(
                question["document_id"]
            ),
            "topics": question_topics.get(question["id"], []),
        })

    return {
        "questions": output,
        "total": len(output),
    }


def start_past_paper_practice(
    user_id,
    course_id,
    topic_id,
    question_ids,
):
    """
    Create a practice session from reviewed past-paper questions.
    """

    if not question_ids:
        raise ValueError("Select at least one past-paper question")

    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Duplicate questions are not allowed")

    if len(question_ids) > 50:
        raise ValueError("A session cannot exceed 50 questions")

    db = get_db()

    selected_questions = []

    for question_id in question_ids:
        result = (
            db.table("past_paper_questions")
            .select(
                "id, course_id, user_id, question_text, question_type, "
                "marks, paper_year, page_number, document_id"
            )
            .eq("id", question_id)
            .eq("user_id", user_id)
            .eq("course_id", course_id)
            .eq("review_status", "reviewed")
            .maybe_single()
            .execute()
        )

        question = data_of(result)

        if not question:
            raise ValueError(
                "A selected past-paper question is unavailable "
                "or has not been reviewed"
            )

        if topic_id:
            mapping = (
                db.table("question_topic_mappings")
                .select("id")
                .eq("past_paper_question_id", question_id)
                .eq("topic_id", topic_id)
                .eq("user_id", user_id)
                .maybe_single()
                .execute()
            )

            if not data_of(mapping):
                raise ValueError(
                    "A selected question is not mapped to this topic"
                )

        selected_questions.append(question)

    # Create a normal practice session.
    session_result = (
        db.table("practice_sessions")
        .insert({
            "user_id": user_id,
            "course_id": course_id,
            "topic_id": topic_id,
            "total_questions": len(selected_questions),
            "correct_answers": 0,
            "status": "in_progress",
        })
        .select("*")
        .single()
        .execute()
    )

    session = session_result.data

    # Save the exact past-paper question set and order.
    assignment_rows = []

    for index, question in enumerate(
        selected_questions,
        start=1
    ):
        assignment_rows.append({
            "session_id": session["id"],
            "past_paper_question_id": question["id"],
            "question_id": None,
            "user_id": user_id,
            "question_order": index,
        })

    db.table("practice_session_questions").insert(
        assignment_rows
    ).execute()

    return {
        "session_id": session["id"],
        "practice_type": "past_paper",
        "total_questions": len(selected_questions),
        "questions": [
            {
                "id": question["id"],
                "question_text": question["question_text"],
                "question_type": question["question_type"],
                "marks": question["marks"],
                "paper_year": question["paper_year"],
                "page_number": question["page_number"],
                "document_id": question["document_id"],
            }
            for question in selected_questions
        ],
    }


def submit_past_paper_answer(
    user_id,
    course_id,
    session_id,
    question_id,
    response_text,
    response_time_seconds=None,
):
    """
    Save a written response to a past-paper question.

    Written answers are not automatically marked.
    """

    if not response_text or not response_text.strip():
        raise ValueError("Write an answer before submitting")

    db = get_db()

    # Verify session ownership and course.
    session_result = (
        db.table("practice_sessions")
        .select("*")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .maybe_single()
        .execute()
    )

    session = data_of(session_result)

    if not session:
        raise ValueError("Practice session not found")

    if session["status"] != "in_progress":
        raise ValueError("Practice session is not active")

    # Verify this question was assigned to this session.
    assignment_result = (
        db.table("practice_session_questions")
        .select("id")
        .eq("session_id", session_id)
        .eq("past_paper_question_id", question_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )

    if not data_of(assignment_result):
        raise ValueError(
            "This past-paper question does not belong to this session"
        )

    # Verify the question is still reviewed and belongs to the course.
    question_result = (
        db.table("past_paper_questions")
        .select("id, question_text, marks, page_number, document_id")
        .eq("id", question_id)
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .eq("review_status", "reviewed")
        .maybe_single()
        .execute()
    )

    question = data_of(question_result)

    if not question:
        raise ValueError("Past-paper question is unavailable")

    # Do not allow the same question to be submitted twice.
    existing = (
        db.table("past_paper_practice_attempts")
        .select("id")
        .eq("session_id", session_id)
        .eq("past_paper_question_id", question_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )

    if data_of(existing):
        raise ValueError("This question has already been answered")

    attempt_result = (
        db.table("past_paper_practice_attempts")
        .insert({
            "session_id": session_id,
            "past_paper_question_id": question_id,
            "user_id": user_id,
            "response_text": response_text.strip(),
            "response_time_seconds": response_time_seconds,
        })
        .select("*")
        .single()
        .execute()
    )

    attempt = attempt_result.data

    # Complete the session when all assigned questions have answers.
    count_result = (
        db.table("past_paper_practice_attempts")
        .select("id", count="exact")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .execute()
    )

    answered_count = count_result.count or 0
    total_questions = session["total_questions"]

    if answered_count >= total_questions:
        db.table("practice_sessions").update({
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", session_id).eq("user_id", user_id).execute()

    return {
        "attempt_id": attempt["id"],
        "session_id": session_id,
        "question_id": question_id,
        "saved": True,
        "marking_status": "needs_review",
        "message": (
            "Your written answer has been saved. "
            "It has not been automatically marked."
        ),
        "answered_questions": answered_count,
        "total_questions": total_questions,
    }
