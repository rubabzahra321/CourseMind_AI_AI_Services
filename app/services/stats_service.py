
from uuid import UUID

from app.core.config import settings
from app.repositories import topic_stats_repository


def get_course_topic_stats(
    user_id: UUID,
    course_id: UUID
) -> dict:

    topics = topic_stats_repository.find_by_course(
        user_id,
        course_id
    )

    minimum = settings.min_papers_for_frequency

    total_ready_papers = max(
        (
            int(topic.get("total_ready_papers") or 0)
            for topic in topics
        ),
        default=0
    )

    sufficient = total_ready_papers >= minimum

    for topic in topics:

        topic_papers = int(
            topic.get("total_ready_papers") or 0
        )

        topic["data_sufficient"] = (
            topic_papers >= minimum
        )

        topic["data_sufficiency_warning"] = (
            None
            if topic["data_sufficient"]
            else (
                f"Only {topic_papers} ready past paper(s) are available. "
                f"At least {minimum} are recommended for reliable "
                "frequency analysis."
            )
        )

    return {
        # Kept for compatibility with existing dashboard clients.
        "course_has_frequency_data": any(
            topic.get("course_has_frequency_data")
            for topic in topics
        ),

        "data_sufficiency": {
            "sufficient": sufficient,
            "ready_papers": total_ready_papers,
            "recommended_minimum": minimum,
            "warning": (
                None
                if sufficient
                else (
                    f"Only {total_ready_papers} ready past paper(s) "
                    f"are available. Upload at least {minimum} "
                    "for more reliable frequency analysis."
                )
            ),
        },

        "topics": topics,
    }
