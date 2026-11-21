
from app.db.client import get_db


def create_document_notification(
    user_id,
    document_id,
    notification_type,
    title,
    message,
):
    db = get_db()

    return (
        db.table("notifications")
        .insert({
            "user_id": str(user_id),
            "document_id": str(document_id),
            "notification_type": notification_type,
            "title": title,
            "message": message,
            "metadata": {
                "document_id": str(document_id),
            },
        })
        .execute()
    )
