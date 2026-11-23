from app.db.client import get_db


TABLE = "ml_models"


def _scope(query, user_id, course_id):
    """
    A model belongs to one student's course, or is GLOBAL (both ids None):
    a pooled model trained on many students' anonymised data.
    """

    query = (
        query.eq("user_id", str(user_id))
        if user_id is not None else query.is_("user_id", "null")
    )

    return (
        query.eq("course_id", str(course_id))
        if course_id is not None else query.is_("course_id", "null")
    )


def list_models(user_id, course_id, model_type):
    """All versions, newest first."""

    return (
        _scope(get_db().table(TABLE).select("*"), user_id, course_id)
        .eq("model_type", model_type)
        .order("version", desc=True)
        .execute()
        .data
        or []
    )


def get_active(user_id, course_id, model_type):
    rows = (
        _scope(get_db().table(TABLE).select("*"), user_id, course_id)
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
        _scope(
            get_db().table(TABLE).update({"is_active": False}),
            user_id, course_id,
        )
        .eq("model_type", model_type)
        .eq("is_active", True)
        .execute()
    )

    return (
        get_db()
        .table(TABLE)
        .update({"is_active": True})
        .eq("id", str(model_id))
        .select("*")
        .execute()
        .data
    )
