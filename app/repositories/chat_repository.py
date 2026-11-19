from app.db.client import get_db
from app.db.helpers import data_of


def get_session(session_id, user_id, course_id):

    result = (
        get_db()
        .table("chat_sessions")
        .select("*")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .maybe_single()
        .execute()
    )

    return data_of(result)


def create_session(user_id, course_id, question):

    result = (
        get_db()
        .table("chat_sessions")
        .insert({
            "user_id": user_id,
            "course_id": course_id,
            "title": question[:60],
        })
        .select("*")
        .single()
        .execute()
    )

    return result.data


def get_recent_messages(session_id, user_id, limit=8):

    result = (
        get_db()
        .table("chat_messages")
        .select("role, content")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )

    return list(reversed(result.data or []))


def save_message(
    session_id,
    user_id,
    role,
    content,
    insufficient_evidence=False,
    model_name=None,
):

    result = (
        get_db()
        .table("chat_messages")
        .insert({
            "session_id": session_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "model_name": model_name,
            "insufficient_evidence": insufficient_evidence,
        })
        .select("*")
        .single()
        .execute()
    )

    return result.data


def save_sources(message_id, user_id, sources):

    if not sources:
        return

    rows = []

    for source in sources:

        rows.append({
            "message_id": message_id,
            "chunk_id": source["chunk_id"],
            "document_id": source["document_id"],
            "user_id": user_id,
            "page_number": source["page_start"],
            "source_rank": source["rank"],
            "excerpt": source["content"][:1000],
        })

    get_db().table("chat_message_sources").insert(rows).execute()