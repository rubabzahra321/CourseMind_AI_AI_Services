from uuid import UUID
from app.db.client import get_db


def find_by_course(user_id: UUID, course_id: UUID) -> list[dict]:
    # Service role bypasses RLS, so both filters are mandatory.
    res = (
        get_db()
        .table("topic_ai_stats")
        .select("*")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .execute()
    )
    return res.data