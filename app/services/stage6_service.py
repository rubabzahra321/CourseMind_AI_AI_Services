
from uuid import UUID

from app.agents.stage6_agents import (
    simple_reflex_agent,
    model_based_reflex_agent,
    utility_based_agent,
    goal_based_agent,
    calculate_utility,
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

    stats = repo.get_topic_stats(user_id, course_id)

    # Supply the actual syllabus tree and prerequisite graph.
    syllabus_topics = repo.get_topics(user_id, course_id)
    prerequisites = repo.get_prerequisites(user_id, course_id)

    schedule = goal_based_agent(
        topic_stats=stats,
        exam_date=course["exam_date"],
        syllabus_topics=syllabus_topics,
        prerequisites=prerequisites,
    )

    # Do not archive an existing plan until a new valid schedule exists.
    repo.archive_active_plans(user_id, course_id)

    plan = repo.create_study_plan(
        user_id,
        course_id,
        course["exam_date"],
        {
            "goal": (
                "Cover syllabus branches before the exam, "
                "respecting prerequisite order."
            ),
            "topic_count": len(schedule),
            "algorithm": (
                "Goal-Based Agent using BFS prerequisites "
                "and DFS syllabus coverage"
            ),
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
        }
        for item in schedule
    ]

    saved_items = repo.insert_study_plan_items(items)

    return {
        "plan": plan,
        "items": saved_items,
        "algorithms": {
            "priority": "Utility score",
            "prerequisites": "Breadth-First Search",
            "syllabus_coverage": "Depth-First Search",
        },
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
