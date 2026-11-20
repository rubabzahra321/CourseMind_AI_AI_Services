
from app.db.client import get_db


def replace_extracted_topics(
    user_id: str,
    course_id: str,
    document_id: str,
    chapters: list[dict]
) -> int:

    """Replace only this document's still-unreviewed AI extraction; preserve manual/reviewed rows."""

    db = get_db()

    # Delete only unreviewed AI-extracted topics from this document.
    (
        db.table("syllabus_topics").delete()
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .eq("source_document_id", document_id)
        .eq("origin", "ai_extracted")
        .eq("reviewed", False)
        .execute()
    )

    count = 0

    def insert_node(
        node: dict,
        parent_id: str | None,
        position: int
    ) -> str:

        nonlocal count

        row = {
            "user_id": user_id,
            "course_id": course_id,
            "source_document_id": document_id,
            "parent_topic_id": parent_id,
            "title": node["title"].strip()[:250],
            "description": node.get("description"),
            "position": position,
            "source_page_start": node.get("page_start"),
            "source_page_end": node.get("page_end"),
            "origin": "ai_extracted",
            "reviewed": False,
        }

        result = (
            db.table("syllabus_topics")
            .insert(row)
            .select("id")
            .single()
            .execute()
        )

        count += 1

        return result.data["id"]

    for chapter_index, chapter in enumerate(chapters):

        chapter_id = insert_node(
            chapter,
            None,
            chapter_index
        )

        for topic_index, topic in enumerate(
            chapter.get("topics", [])
        ):

            topic_id = insert_node(
                topic,
                chapter_id,
                topic_index
            )

            for subtopic_index, subtopic in enumerate(
                topic.get("subtopics", [])
            ):

                insert_node(
                    subtopic,
                    topic_id,
                    subtopic_index
                )

    return count
