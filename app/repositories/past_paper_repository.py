
from app.db.client import get_db


def replace_pending_extracted_questions(
    user_id: str,
    course_id: str,
    document_id: str,
    questions: list[dict]
) -> int:

    """Replace only pending AI-extracted rows; never overwrite reviewed/manual corrections."""

    db = get_db()

    existing = (
        db.table("past_paper_questions")
        .select("id, source_metadata")
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .eq("document_id", document_id)
        .eq("review_status", "pending")
        .execute()
    ).data or []

    for old in existing:

        metadata = old.get("source_metadata") or {}

        if metadata.get("extraction_source") == "ai_extracted":

            (
                db.table("past_paper_questions")
                .delete()
                .eq("id", old["id"])
                .execute()
            )

    rows = []

    for item in questions:

        rows.append({
            "user_id": user_id,
            "course_id": course_id,
            "document_id": document_id,
            "question_number": item.get("question_number"),
            "question_text": item["question_text"],
            "question_type": item.get("question_type", "unknown"),
            "marks": item.get("marks"),
            "paper_year": item.get("paper_year"),
            "page_number": item.get("page_number"),
            "review_status": "pending",
            "extraction_confidence": item.get(
                "extraction_confidence"
            ),
            "source_metadata": {
                "extraction_source": "ai_extracted",
                "extractor_version": 1,
            },
        })

    if not rows:
        return 0

    result = (
        db.table("past_paper_questions")
        .insert(rows)
        .select("id")
        .execute()
    )

    return len(result.data or [])
