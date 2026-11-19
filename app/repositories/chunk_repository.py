from app.db.client import get_db

BATCH = 100


def delete_for_document(document_id: str, user_id: str) -> None:
    # Makes retries safe: no duplicate chunks.
    get_db().table("document_chunks").delete() \
        .eq("document_id", document_id).eq("user_id", user_id).execute()


def insert_many(rows: list[dict]) -> None:
    for i in range(0, len(rows), BATCH):
        get_db().table("document_chunks").insert(rows[i:i + BATCH]).execute()