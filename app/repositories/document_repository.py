from app.db.client import get_db


def get(document_id: str, user_id: str, course_id: str) -> dict | None:
    # Service role bypasses RLS: filter by owner and course every time.
    res = (
        get_db().table("documents").select("*")
        .eq("id", document_id).eq("user_id", user_id).eq("course_id", course_id)
        .maybe_single().execute()
    )
    return res.data if res else None


def update(document_id: str, fields: dict) -> None:
    get_db().table("documents").update(fields).eq("id", document_id).execute()