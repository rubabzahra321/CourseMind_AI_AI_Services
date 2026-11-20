
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
    depth_mode=None,
    evidence_strength=None,
):
    payload = {
        "session_id": session_id,
        "user_id": user_id,
        "role": role,
        "content": content,
        "model_name": model_name,
        "insufficient_evidence": insufficient_evidence,
        "depth_mode": depth_mode,
        "evidence_strength": evidence_strength,
    }

    result = (
        get_db()
        .table("chat_messages")
        .insert(payload)
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
            "chunk_id": source.get("chunk_id"),
            "document_id": source["document_id"],
            "user_id": user_id,
            "page_number": source.get("page_start"),
            "source_rank": source["rank"],
            "excerpt": source["content"][:1000],
        })

    get_db().table("chat_message_sources").insert(rows).execute()


def list_sessions(user_id, course_id, limit=30):
    result = (
        get_db()
        .table("chat_sessions")
        .select("id, title, created_at, updated_at")
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .order("updated_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def list_messages(session_id, user_id, course_id, limit=100):
    session = get_session(session_id, user_id, course_id)
    if not session:
        raise ValueError("Chat session not found")

    result = (
        get_db()
        .table("chat_messages")
        .select(
            "id, role, content, insufficient_evidence, "
            "depth_mode, evidence_strength, model_name, created_at"
        )
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .order("created_at")
        .limit(limit)
        .execute()
    )

    messages = result.data or []

    for message in messages:
        source_result = (
            get_db()
            .table("chat_message_sources")
            .select(
                "document_id, page_number, source_rank, excerpt"
            )
            .eq("message_id", message["id"])
            .eq("user_id", user_id)
            .order("source_rank")
            .execute()
        )
        message["sources"] = source_result.data or []

    return {
        "session": session,
        "messages": messages,
    }
