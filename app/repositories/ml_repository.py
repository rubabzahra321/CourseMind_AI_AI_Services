from app.db.client import get_db


TABLE = "ml_models"


def list_models(user_id, course_id, model_type):
    """All versions, newest first."""

    return (
        get_db()
        .table(TABLE)
        .select("*")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .eq("model_type", model_type)
        .order("version", desc=True)
        .execute()
        .data
        or []
    )


def get_active(user_id, course_id, model_type):
    rows = (
        get_db()
        .table(TABLE)
        .select("*")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .eq("model_type", model_type)
        .eq("is_active", True)
        .execute()
        .data
        or []
    )

    return rows[0] if rows else None


def insert_model(row):
    return (
        get_db()
        .table(TABLE)
        .insert(row)
        .select("*")
        .single()
        .execute()
        .data
    )


def activate(model_id, user_id, course_id, model_type):
    """Make one version the only active one."""

    (
        get_db()
        .table(TABLE)
        .update({"is_active": False})
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .eq("model_type", model_type)
        .eq("is_active", True)
        .execute()
    )

    return (
        get_db()
        .table(TABLE)
        .update({"is_active": True})
        .eq("id", str(model_id))
        .eq("user_id", str(user_id))
        .select("*")
        .execute()
        .data
    )
