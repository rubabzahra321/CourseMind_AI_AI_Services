
from uuid import UUID
from datetime import datetime, timezone, date

from app.agents.stage6_agents import (
    simple_reflex_agent,
    model_based_reflex_agent,
    utility_based_agent,
    calculate_utility,
    _build_dfs_study_order,
    schedule_revision_tasks,
)

from app.search.algorithms import (
    bfs_prerequisites,
    dfs_topic_subtree,
    greedy_best_first,
)

from app.repositories import stage6_repository as repo


def get_agent_dashboard(user_id: UUID, course_id: UUID):
    # Analytics are used by the simple reflex and utility agents.
    stats = repo.get_topic_stats(user_id, course_id)

    # Stored memory is read separately from topic_progress.
    progress_memory = repo.get_topic_progress_memory(
        user_id,
        course_id,
    )

    failed_documents = repo.get_failed_documents(
        user_id,
        course_id,
    )

    return {
        "study_alerts": simple_reflex_agent(
            stats,
            failed_documents,
        ),
        "topic_reminders": model_based_reflex_agent(
            progress_memory,
        ),
        "recommended_next_action": utility_based_agent(stats),
        "quick_revision_order": greedy_best_first(
            [
                {
                    **topic,
                    "utility_score": calculate_utility(topic),
                }
                for topic in stats
                if topic.get("is_leaf", True)
            ]
        ),
        "model_memory_source": "topic_progress",
    }


def get_prerequisite_path(
    user_id: UUID,
    course_id: UUID,
    topic_id: UUID,
):
    topic = repo.get_owned_topic(
        user_id,
        course_id,
        topic_id,
    )

    if not topic:
        raise ValueError("Topic not found in this course.")

    topics = repo.get_topics(user_id, course_id)
    prerequisites = repo.get_prerequisites(user_id, course_id)

    path = bfs_prerequisites(str(topic_id), prerequisites)

    topic_map = {
        str(item["id"]): item
        for item in topics
    }

    return [
        {
            "topic_id": item,
            "topic_title": topic_map.get(item, {}).get(
                "title",
                "Unknown topic",
            ),
            "position": index + 1,
        }
        for index, item in enumerate(path)
    ]


def get_syllabus_subtree(
    user_id: UUID,
    course_id: UUID,
    topic_id: UUID,
):
    topic = repo.get_owned_topic(
        user_id,
        course_id,
        topic_id,
    )

    if not topic:
        raise ValueError("Topic not found in this course.")

    topics = repo.get_topics(user_id, course_id)

    subtree = dfs_topic_subtree(str(topic_id), topics)

    topic_map = {
        str(item["id"]): item
        for item in topics
    }

    return [
        {
            **item,
            "topic_title": topic_map.get(
                item["topic_id"],
                {},
            ).get("title", "Unknown topic"),
        }
        for item in subtree
    ]


def generate_study_plan(user_id: UUID, course_id: UUID):
    course = repo.get_course(user_id, course_id)

    if not course:
        raise ValueError("Course not found.")

    if not course.get("exam_date"):
        raise ValueError("Set an exam date for this course first.")

    hours_per_day = float(course.get("hours_per_day") or 2)

    # -------------------------------------------------
    # 1. Get topic statistics and syllabus information.
    # -------------------------------------------------

    stats = repo.get_topic_stats(user_id, course_id)

    topics = repo.get_topics(user_id, course_id)

    prerequisites = repo.get_prerequisites(user_id, course_id)

    # -------------------------------------------------
    # 2. Keep the Phase 5 BFS + DFS topic ordering.
    # -------------------------------------------------

    topic_schedule = _build_dfs_study_order(
        stats,
        topics,
        prerequisites,
    )

    # -------------------------------------------------
    # 3. Apply Phase 6 time-based revision scheduling.
    # -------------------------------------------------

    result = schedule_revision_tasks(
        stats,
        topic_schedule,
        course["exam_date"],
        hours_per_day,
    )

    schedule = result["items"]

    # Do not archive the previous plan if generation failed.
    repo.archive_active_plans(user_id, course_id)

    plan = repo.create_study_plan(
        user_id,
        course_id,
        course["exam_date"],
        {
            "goal": "Complete revision tasks before the exam.",
            "algorithm": "Time-Based Goal-Based Agent",
            "hours_per_day": hours_per_day,
            "daily_minutes": result["daily_minutes"],
            "available_days": result["available_days"],
            "total_tasks": result["total_tasks"],
            "scheduled_tasks": result["scheduled_tasks"],
            "unscheduled_tasks": len(result["unscheduled_tasks"]),
        },
    )

    items = [
        {
            "plan_id": plan["id"],
            "course_id": str(course_id),
            "user_id": str(user_id),
            "topic_id": item["topic_id"],
            "planned_date": item["planned_date"],
            "position": item["position"],
            "utility_score": item["utility_score"],
            "reason": item["reason"],
            "task_type": item["task_type"],
            "estimated_minutes": item["estimated_minutes"],
        }
        for item in schedule
    ]

    saved_items = repo.insert_study_plan_items(items)

    return {
        "plan": plan,
        "items": saved_items,
        "summary": {
            "daily_minutes": result["daily_minutes"],
            "available_days": result["available_days"],
            "total_tasks": result["total_tasks"],
            "scheduled_tasks": result["scheduled_tasks"],
            "unscheduled_tasks": result["unscheduled_tasks"],
        },
        "feasibility": {
            "hours_per_day": hours_per_day,
            "available_days": result["available_days"],
            "daily_minutes": result["daily_minutes"],
            "total_tasks": result["total_tasks"],
            "scheduled_tasks": result["scheduled_tasks"],
            "unscheduled_tasks": result["unscheduled_tasks"],
            "is_feasible": len(result["unscheduled_tasks"]) == 0,
            "warning": (
                None
                if len(result["unscheduled_tasks"]) == 0
                else (
                    f"{len(result['unscheduled_tasks'])} tasks "
                    "could not be scheduled before the exam. "
                    "Increase your daily study hours or revise "
                    "your preparation priorities."
                )
            ),
        },
        "algorithm": "Time-Based Goal-Based Agent",
    }


def get_current_study_plan(user_id: UUID, course_id: UUID):
    plan = repo.list_active_plan(user_id, course_id)

    if not plan:
        return None

    items = repo.list_plan_items(
        user_id,
        course_id,
        plan["id"],
    )

    return {
        "plan": plan,
        "items": items,
    }


def complete_study_plan_item(user_id, course_id, item_id):
    result = repo.mark_plan_item_done(
        user_id,
        course_id,
        item_id,
    )

    if not result:
        raise ValueError("Study plan item not found.")

    return result[0]


def update_study_plan_item(
    user_id,
    course_id,
    item_id,
    status,
    planned_date=None,
):
    allowed_statuses = {
        "pending",
        "done",
        "skipped",
        "rescheduled",
    }

    if status not in allowed_statuses:
        raise ValueError("Invalid study task status.")

    changes = {
        "status": status,
    }

    if status == "done":
        changes["is_done"] = True
        changes["done_at"] = datetime.now(
            timezone.utc
        ).isoformat()

    elif status == "skipped":
        changes["is_done"] = False
        changes["done_at"] = None
        changes["skipped_at"] = datetime.now(
            timezone.utc
        ).isoformat()

    elif status == "rescheduled":
        if not planned_date:
            raise ValueError(
                "A new date is required when rescheduling a task."
            )

        try:
            new_date = date.fromisoformat(planned_date)
        except (TypeError, ValueError):
            raise ValueError(
                "Use a valid date in YYYY-MM-DD format."
            )

        if new_date < date.today():
            raise ValueError(
                "A task cannot be rescheduled to a past date."
            )

        changes["rescheduled_from"] = planned_date
        changes["planned_date"] = planned_date
        changes["is_done"] = False
        changes["done_at"] = None
        changes["skipped_at"] = None

    else:
        changes["is_done"] = False
        changes["done_at"] = None
        changes["skipped_at"] = None

    result = repo.update_plan_item(
        user_id,
        course_id,
        item_id,
        changes,
    )

    if not result:
        raise ValueError("Study plan item not found.")

    return result[0] if isinstance(result, list) else result
