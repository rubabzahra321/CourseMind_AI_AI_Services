
from uuid import UUID
from datetime import datetime, timezone

from app.db.client import get_db


def get_topic_stats(user_id: UUID, course_id: UUID):
    return (
        get_db().table("topic_ai_stats")
        .select("*")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .execute()
        .data or []
    )


def get_failed_documents(user_id: UUID, course_id: UUID):
    return (
        get_db().table("documents")
        .select("id,title,status")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .eq("status", "failed")
        .execute()
        .data or []
    )


def get_topics(user_id: UUID, course_id: UUID):
    return (
        get_db().table("syllabus_topics")
        .select("id,title,parent_topic_id,position")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .order("position")
        .execute()
        .data or []
    )


def get_prerequisites(user_id: UUID, course_id: UUID):
    return (
        get_db().table("topic_prerequisites")
        .select("topic_id,prerequisite_topic_id")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .execute()
        .data or []
    )


def get_owned_topic(user_id: UUID, course_id: UUID, topic_id: UUID):
    result = (
        get_db().table("syllabus_topics")
        .select("id,title,parent_topic_id")
        .eq("id", str(topic_id))
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .maybe_single()
        .execute()
    )
    return result.data


def get_course(user_id: UUID, course_id: UUID):
    result = (
        get_db().table("courses")
        .select("id,title,exam_date")
        .eq("id", str(course_id))
        .eq("user_id", str(user_id))
        .maybe_single()
        .execute()
    )
    return result.data


def archive_active_plans(user_id: UUID, course_id: UUID):
    (
        get_db().table("study_plans")
        .update({"status": "archived"})
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .eq("status", "active")
        .execute()
    )


def create_study_plan(user_id, course_id, exam_date, summary):
    result = (
        get_db().table("study_plans")
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
        get_db().table("study_plan_items")
        .insert(items)
        .execute()
        .data or []
    )


def list_active_plan(user_id: UUID, course_id: UUID):
    result = (
        get_db().table("study_plans")
        .select("id,exam_date,status,summary,created_at")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .eq("status", "active")
        .maybe_single()
        .execute()
    )
    return result.data


def list_plan_items(user_id: UUID, course_id: UUID, plan_id: UUID):
    return (
        get_db().table("study_plan_items")
        .select("*")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .eq("plan_id", str(plan_id))
        .order("planned_date")
        .order("position")
        .execute()
        .data or []
    )


def mark_plan_item_done(user_id, course_id, item_id):
    result = (
        get_db().table("study_plan_items")
        .update({
            "is_done": True,
            "done_at": datetime.now(timezone.utc).isoformat(),
        })
        .eq("id", str(item_id))
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .execute()
    )
    return result.data
