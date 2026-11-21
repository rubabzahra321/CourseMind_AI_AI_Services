
from uuid import UUID
from datetime import datetime, timezone

from app.db.client import get_db
from app.db.helpers import data_of


def get_topic_stats(user_id: UUID, course_id: UUID):
    """
    Analytics data for the reflex and utility agents.

    This remains based on topic_ai_stats because it contains
    frequency, accuracy, leaf status, and exam urgency.
    """
    return (
        get_db()
        .table("topic_ai_stats")
        .select("*")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .execute()
        .data or []
    )


def get_topic_progress_memory(user_id: UUID, course_id: UUID):
    """
    Model-based agent memory.

    The source of attempts, accuracy, status, and last practice date
    is topic_progress, not topic_ai_stats.
    """
    db = get_db()

    topics = (
        db.table("syllabus_topics")
        .select("id,title")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .order("position")
        .execute()
        .data or []
    )

    progress_rows = (
        db.table("topic_progress")
        .select(
            "topic_id,attempts,correct_answers,accuracy_percentage,"
            "last_practised_at,status,previous_status"
        )
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .execute()
        .data or []
    )

    progress_map = {
        str(row["topic_id"]): row
        for row in progress_rows
    }

    memory = []

    # Include unpractised topics too. A topic without a memory row
    # should produce the "not practised yet" reminder.
    for topic in topics:
        topic_id = str(topic["id"])
        progress = progress_map.get(topic_id, {})

        memory.append({
            "topic_id": topic_id,
            "topic_title": topic["title"],
            "attempts": progress.get("attempts", 0),
            "correct_answers": progress.get("correct_answers", 0),
            "accuracy_percentage": progress.get("accuracy_percentage"),
            "last_practised_at": progress.get("last_practised_at"),
            "status": progress.get(
                "status",
                "Not Yet Assessed",
            ),
            "previous_status": progress.get("previous_status"),
        })

    return memory


def get_failed_documents(user_id: UUID, course_id: UUID):
    return (
        get_db()
        .table("documents")
        .select("id,title,status")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .eq("status", "failed")
        .execute()
        .data or []
    )


def get_topics(user_id: UUID, course_id: UUID):
    return (
        get_db()
        .table("syllabus_topics")
        .select("id,title,parent_topic_id,position")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .order("position")
        .execute()
        .data or []
    )


def get_prerequisites(user_id: UUID, course_id: UUID):
    return (
        get_db()
        .table("topic_prerequisites")
        .select("topic_id,prerequisite_topic_id")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .execute()
        .data or []
    )


def get_owned_topic(user_id: UUID, course_id: UUID, topic_id: UUID):
    result = (
        get_db()
        .table("syllabus_topics")
        .select("id,title,parent_topic_id")
        .eq("id", str(topic_id))
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .maybe_single()
        .execute()
    )

    return data_of(result)


def get_course(user_id: UUID, course_id: UUID):
    result = (
        get_db().table("courses")
        .select("id,title,exam_date,hours_per_day")
        .eq("id", str(course_id))
        .eq("user_id", str(user_id))
        .maybe_single()
        .execute()
    )
    return data_of(result)


def archive_active_plans(user_id: UUID, course_id: UUID):
    (
        get_db()
        .table("study_plans")
        .update({"status": "archived"})
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .eq("status", "active")
        .execute()
    )


def create_study_plan(user_id, course_id, exam_date, summary):
    result = (
        get_db()
        .table("study_plans")
        .insert({
            "user_id": str(user_id),
            "course_id": str(course_id),
            "exam_date": str(exam_date),
            "status": "active",
            "summary": summary,
        })
        .execute()
    )

    return result.data[0]


def insert_study_plan_items(items):
    if not items:
        return []

    return (
        get_db()
        .table("study_plan_items")
        .insert(items)
        .execute()
        .data or []
    )


def list_active_plan(user_id: UUID, course_id: UUID):
    result = (
        get_db()
        .table("study_plans")
        .select("id,exam_date,status,summary,created_at")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .eq("status", "active")
        .maybe_single()
        .execute()
    )

    return data_of(result)


def list_plan_items(user_id: UUID, course_id: UUID, plan_id: UUID):
    return (
        get_db()
        .table("study_plan_items")
        .select("*")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .eq("plan_id", str(plan_id))
        .order("planned_date")
        .order("position")
        .execute()
        .data or []
    )


def update_plan_item(user_id, course_id, item_id, changes):
    result = (
        get_db().table("study_plan_items")
        .update(changes)
        .eq("id", str(item_id))
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .execute()
    )

    return data_of(result)


def mark_plan_item_done(user_id, course_id, item_id):
    now = datetime.now(timezone.utc).isoformat()

    return update_plan_item(
        user_id,
        course_id,
        item_id,
        {
            "is_done": True,
            "done_at": now,
            "status": "done",
        },
    )
