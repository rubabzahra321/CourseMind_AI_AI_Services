from uuid import UUID
from app.repositories import topic_stats_repository


def get_course_topic_stats(user_id: UUID, course_id: UUID) -> dict:
    topics = topic_stats_repository.find_by_course(user_id, course_id)
    return {
        # False means no mapped past-paper data yet; the dashboard should say so.
        "course_has_frequency_data": any(t.get("course_has_frequency_data") for t in topics),
        "topics": topics,
    }