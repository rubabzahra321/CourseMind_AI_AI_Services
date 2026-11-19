from app.db.client import get_db


def search_chunks(
    user_id: str,
    course_id: str,
    query_embedding: list[float],
    top_k: int,
    min_similarity: float,
) -> list[dict]:

    vector_text = "[" + ",".join(
        str(float(value)) for value in query_embedding
    ) + "]"

    result = get_db().rpc(
        "match_document_chunks",
        {
            "p_query_embedding": vector_text,
            "p_user_id": user_id,
            "p_course_id": course_id,
            "p_match_count": top_k,
            "p_min_similarity": min_similarity,
        }
    ).execute()

    return result.data or []