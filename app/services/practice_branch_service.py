from uuid import UUID

from app.db.client import get_db
from app.repositories import practice_repository
from app.repositories import stage6_repository
from app.search.algorithms import dfs_topic_subtree


def start_branch_practice(
    user_id: UUID,
    course_id: UUID,
    root_topic_id: UUID,
    max_questions_per_topic: int = 5,
    max_total_questions: int = 50,
):
    """
    Start a practice session using DFS over a syllabus topic branch.

    Questions are selected from the root topic and its descendants.
    Each topic contributes at most max_questions_per_topic questions.
    """

    root = stage6_repository.get_owned_topic(
        user_id,
        course_id,
        root_topic_id,
    )

    if not root:
        raise ValueError("Topic not found in this course.")

    if max_questions_per_topic < 1 or max_questions_per_topic > 20:
        raise ValueError(
            "max_questions_per_topic must be between 1 and 20."
        )

    if max_total_questions < 1 or max_total_questions > 50:
        raise ValueError(
            "max_total_questions must be between 1 and 50."
        )

    topics = stage6_repository.get_topics(
        user_id,
        course_id,
    )

    # DFS returns the root followed by all descendants.
    subtree = dfs_topic_subtree(
        str(root_topic_id),
        topics,
    )

    topic_map = {
        str(topic["id"]): topic
        for topic in topics
    }

    db = get_db()

    selected_questions = []
    topic_coverage = []

    # Follow DFS order when collecting questions.
    for branch_item in subtree:
        topic_id = str(branch_item["topic_id"])
        topic = topic_map.get(topic_id)

        if not topic:
            continue

        remaining = max_total_questions - len(selected_questions)

        if remaining <= 0:
            break

        topic_limit = min(
            max_questions_per_topic,
            remaining,
        )

        result = (
            db.table("generated_questions")
            .select("*")
            .eq("user_id", str(user_id))
            .eq("course_id", str(course_id))
            .eq("topic_id", topic_id)
            .eq("status", "active")
            .order("created_at", desc=True)
            .limit(topic_limit)
            .execute()
        )

        questions = result.data or []

        topic_coverage.append({
            "topic_id": topic_id,
            "topic_title": topic["title"],
            "depth": branch_item["depth"],
            "available_questions": len(questions),
            "selected_questions": len(questions),
        })

        for question in questions:
            selected_questions.append({
                **question,
                "topic_title": topic["title"],
                "topic_depth": branch_item["depth"],
            })

    if not selected_questions:
        raise ValueError(
            "No generated practice questions were found in this topic branch."
        )

    # Create one session for the complete selected branch.
    session = practice_repository.create_session(
        user_id=str(user_id),
        course_id=str(course_id),
        topic_id=str(root_topic_id),
        total_questions=len(selected_questions),
    )

    assignment_rows = [
        {
            "session_id": session["id"],
            "question_id": question["id"],
            "user_id": str(user_id),
            "question_order": index,
        }
        for index, question in enumerate(
            selected_questions,
            start=1,
        )
    ]

    db.table("practice_session_questions").insert(
        assignment_rows
    ).execute()

    result_questions = []

    for question in selected_questions:
        options_result = (
            db.table("question_options")
            .select("id, option_text, option_order")
            .eq("question_id", question["id"])
            .order("option_order")
            .execute()
        )

        result_questions.append({
            "id": question["id"],
            "topic_id": question["topic_id"],
            "topic_title": question["topic_title"],
            "topic_depth": question["topic_depth"],
            "question_text": question["question_text"],
            "question_type": question.get("question_type") or "mcq",
            "difficulty": question.get("difficulty"),
            "max_marks": question.get("max_marks"),
            "options": options_result.data or [],
        })

    return {
        "session_id": session["id"],
        "root_topic": {
            "id": str(root_topic_id),
            "title": root["title"],
        },
        "algorithm": "Depth-First Search",
        "total_questions": len(result_questions),
        "topic_coverage": topic_coverage,
        "questions": result_questions,
    }