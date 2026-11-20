
import math
from collections import defaultdict
from uuid import UUID

from app.ai.gemini import embed_query
from app.core.config import settings
from app.db.client import get_db


DEFAULT_WEIGHTS = {
    "frequency": settings.priority_frequency_weight,
    "marks": settings.priority_marks_weight,
    "syllabus_weight": settings.priority_syllabus_weight,
    "mastery": settings.priority_mastery_weight,
}

ALLOWED_WEIGHTS = set(DEFAULT_WEIGHTS)


def _db():
    return get_db()


def _rows(query):
    result = query.execute()
    return result.data or []


def _one(query):
    result = query.maybe_single().execute()
    return result.data if result else None


def _course_topics(user_id, course_id):
    return _rows(
        _db().table("syllabus_topics")
        .select(
            "id, course_id, user_id, title, description, "
            "parent_topic_id, source_page_start, source_page_end"
        )
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .order("position")
    )


def _topic_stats(user_id, course_id):
    rows = _rows(
        _db().table("topic_ai_stats")
        .select("*")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
    )
    return {
        str(row["topic_id"]): row
        for row in rows
    }


def _course_weights(user_id, course_id):
    course = _one(
        _db().table("courses")
        .select("id, ai_priority_weights")
        .eq("id", str(course_id))
        .eq("user_id", str(user_id))
    )

    if not course:
        return dict(DEFAULT_WEIGHTS)

    saved = course.get("ai_priority_weights") or {}

    weights = dict(DEFAULT_WEIGHTS)

    for key in ALLOWED_WEIGHTS:
        value = saved.get(key)

        if isinstance(value, (int, float)) and value >= 0:
            weights[key] = float(value)

    return weights


def _normalise_weights(weights):
    total = sum(weights.values())

    if total <= 0:
        raise ValueError("At least one priority weight must be greater than zero.")

    return {
        key: value / total
        for key, value in weights.items()
    }


def get_priority_weights(user_id, course_id):
    weights = _course_weights(user_id, course_id)

    return {
        "weights": weights,
        "total": round(sum(weights.values()), 4),
        "description": {
            "frequency": "How often the topic appears in reviewed past papers.",
            "marks": "Average marks assigned to mapped questions.",
            "syllabus_weight": "Relative size of the topic's syllabus branch.",
            "mastery": "How much practice performance indicates the student needs revision."
        }
    }


def update_priority_weights(user_id, course_id, new_weights):
    if not isinstance(new_weights, dict):
        raise ValueError("Weights must be an object.")

    if set(new_weights.keys()) != ALLOWED_WEIGHTS:
        raise ValueError(
            "Provide exactly frequency, marks, syllabus_weight and mastery."
        )

    cleaned = {}

    for key, value in new_weights.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < 0
            or value > 1
        ):
            raise ValueError(
                f"{key} must be a number between 0 and 1."
            )

        cleaned[key] = float(value)

    normalised = _normalise_weights(cleaned)

    _db().table("courses").update({
        "ai_priority_weights": normalised
    }).eq(
        "id", str(course_id)
    ).eq(
        "user_id", str(user_id)
    ).execute()

    return {
        "weights": normalised,
        "total": round(sum(normalised.values()), 4),
        "message": "Priority weights updated successfully."
    }


def _question_evidence(user_id, course_id, topic_id):
    mappings = _rows(
        _db().table("question_topic_mappings")
        .select("past_paper_question_id")
        .eq("user_id", str(user_id))
        .eq("topic_id", str(topic_id))
    )

    question_ids = list({
        row["past_paper_question_id"]
        for row in mappings
    })

    if not question_ids:
        return []

    questions = _rows(
        _db().table("past_paper_questions")
        .select(
            "id, document_id, question_number, question_text, "
            "marks, paper_year, page_number, review_status"
        )
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .eq("review_status", "reviewed")
        .in_("id", question_ids)
    )

    document_ids = list({
        row["document_id"]
        for row in questions
    })

    documents = []

    if document_ids:
        documents = _rows(
            _db().table("documents")
            .select("id, title, original_filename, academic_year")
            .eq("user_id", str(user_id))
            .eq("course_id", str(course_id))
            .eq("document_type", "past_paper")
            .eq("status", "ready")
            .in_("id", document_ids)
        )

    document_map = {
        row["id"]: row
        for row in documents
    }

    evidence = []

    for question in questions:
        document = document_map.get(
            question["document_id"]
        )

        if not document:
            continue

        evidence.append({
            "question_id": question["id"],
            "question_number": question.get("question_number"),
            "question_text": question["question_text"],
            "marks": question.get("marks"),
            "paper_year": (
                question.get("paper_year")
                or document.get("academic_year")
            ),
            "page_number": question.get("page_number"),
            "document_id": document["id"],
            "document_title": (
                document.get("title")
                or document.get("original_filename")
            ),
        })

    return evidence


def get_topic_evidence(user_id, course_id, topic_id):
    topics = {
        str(row["id"]): row
        for row in _course_topics(user_id, course_id)
    }

    topic = topics.get(str(topic_id))

    if not topic:
        raise ValueError("Topic not found in this course.")

    evidence = _question_evidence(
        user_id,
        course_id,
        topic_id
    )

    papers = {}

    for item in evidence:
        doc_id = item["document_id"]

        if doc_id not in papers:
            papers[doc_id] = {
                "document_id": doc_id,
                "document_title": item["document_title"],
                "year": item["paper_year"],
                "questions": [],
            }

        papers[doc_id]["questions"].append({
            "question_id": item["question_id"],
            "question_number": item["question_number"],
            "question_text": item["question_text"],
            "marks": item["marks"],
            "page_number": item["page_number"],
        })

    total_ready_papers = len(_rows(
        _db().table("documents")
        .select("id")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .eq("document_type", "past_paper")
        .eq("status", "ready")
    ))

    papers_with_topic = len(papers)

    frequency = (
        100.0 * papers_with_topic / total_ready_papers
        if total_ready_papers else 0.0
    )

    sufficient = (
        total_ready_papers >= settings.min_papers_for_frequency
    )

    return {
        "topic": {
            "id": topic["id"],
            "title": topic["title"],
        },
        "frequency": {
            "papers_with_topic": papers_with_topic,
            "total_ready_papers": total_ready_papers,
            "percentage": round(frequency, 2),
            "data_sufficient": sufficient,
            "warning": (
                None if sufficient else
                f"Only {total_ready_papers} ready past paper(s) are available. "
                f"At least {settings.min_papers_for_frequency} are recommended "
                "before treating frequency as reliable."
            ),
        },
        "papers": list(papers.values()),
        "question_count": len(evidence),
    }


def _branch_sizes(topics):
    children = defaultdict(list)

    for topic in topics:
        children[topic.get("parent_topic_id")].append(
            str(topic["id"])
        )

    sizes = {}

    def count_descendants(topic_id, visiting=None):
        if topic_id in sizes:
            return sizes[topic_id]

        visiting = visiting or set()

        if topic_id in visiting:
            return 1

        visiting.add(topic_id)

        total = 1

        for child_id in children.get(topic_id, []):
            total += count_descendants(
                child_id,
                visiting.copy()
            )

        sizes[topic_id] = total

        return total

    for topic in topics:
        count_descendants(str(topic["id"]))

    return sizes


def get_topic_priorities(user_id, course_id):
    topics = _course_topics(user_id, course_id)
    stats = _topic_stats(user_id, course_id)
    weights = _normalise_weights(
        _course_weights(user_id, course_id)
    )

    if not topics:
        return {
            "weights": weights,
            "topics": []
        }

    sizes = _branch_sizes(topics)
    total_size = sum(sizes.values()) or 1

    result = []

    for topic in topics:
        topic_id = str(topic["id"])
        stat = stats.get(topic_id, {})

        frequency = float(
            stat.get("paper_frequency_percentage") or 0
        ) / 100.0

        evidence = _question_evidence(
            user_id,
            course_id,
            topic_id
        )

        marks = [
            float(item["marks"])
            for item in evidence
            if item.get("marks") is not None
        ]

        average_marks = (
            sum(marks) / len(marks)
            if marks else 0.0
        )

        # Marks factor is capped at 10 marks.
        marks_factor = min(average_marks / 10.0, 1.0)

        # Syllabus weight is based on the size of the topic branch.
        syllabus_factor = sizes[topic_id] / total_size

        accuracy = stat.get("accuracy_percentage")

        if accuracy is None:
            mastery_factor = 0.5
        else:
            mastery_factor = 1.0 - (
                max(0.0, min(float(accuracy), 100.0)) / 100.0
            )

        factors = {
            "frequency": round(frequency, 4),
            "marks": round(marks_factor, 4),
            "syllabus_weight": round(syllabus_factor, 4),
            "mastery": round(mastery_factor, 4),
        }

        score = sum(
            factors[key] * weights[key]
            for key in weights
        ) * 100.0

        result.append({
            "topic_id": topic_id,
            "topic_title": topic["title"],
            "priority_score": round(score, 2),
            "priority_level": (
                "high" if score >= 70 else
                "medium" if score >= 40 else
                "low"
            ),
            "factors": {
                "values": factors,
                "weights": weights,
                "contributions": {
                    key: round(
                        factors[key] * weights[key] * 100,
                        2
                    )
                    for key in weights
                },
                "formula": (
                    "score = sum(factor_value * factor_weight) * 100"
                ),
                "average_question_marks": round(
                    average_marks, 2
                ),
                "practice_accuracy": accuracy,
                "syllabus_branch_nodes": sizes[topic_id],
            },
            "data_sufficient": (
                int(stat.get("total_ready_papers") or 0)
                >= settings.min_papers_for_frequency
            ),
        })

    result.sort(
        key=lambda item: item["priority_score"],
        reverse=True
    )

    return {
        "weights": weights,
        "topics": result
    }


def _parse_vector(value):
    if value is None:
        return None

    if isinstance(value, list):
        return [float(x) for x in value]

    if isinstance(value, str):
        cleaned = value.strip().strip("[]")

        if not cleaned:
            return None

        return [
            float(part)
            for part in cleaned.split(",")
        ]

    return None


def _cosine_similarity(first, second):
    if not first or not second or len(first) != len(second):
        return 0.0

    dot = sum(a * b for a, b in zip(first, second))
    norm_a = math.sqrt(sum(a * a for a in first))
    norm_b = math.sqrt(sum(b * b for b in second))

    if not norm_a or not norm_b:
        return 0.0

    return dot / (norm_a * norm_b)


def suggest_topic_chunks(user_id, course_id, topic_id, limit=10):
    topics = {
        str(row["id"]): row
        for row in _course_topics(user_id, course_id)
    }

    topic = topics.get(str(topic_id))

    if not topic:
        raise ValueError("Topic not found in this course.")

    text = topic["title"]

    if topic.get("description"):
        text += "\n" + topic["description"]

    query_vector = embed_query(text)

    chunks = _rows(
        _db().table("document_chunks")
        .select(
            "id, document_id, chunk_index, content, "
            "page_start, page_end, embedding"
        )
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .limit(1000)
    )

    documents = _rows(
        _db().table("documents")
        .select("id, title, document_type")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .eq("status", "ready")
    )

    document_map = {
        item["id"]: item
        for item in documents
    }

    suggestions = []

    for chunk in chunks:
        document = document_map.get(
            chunk["document_id"]
        )

        if not document:
            continue

        # Only lecture material is suggested for this mapping task.
        if document["document_type"] != "lecture_material":
            continue

        vector = _parse_vector(
            chunk.get("embedding")
        )

        similarity = _cosine_similarity(
            query_vector,
            vector
        )

        # Cosine can be negative; map it to the 0..1 range.
        similarity = max(0.0, min(1.0, similarity))

        if similarity < settings.material_similarity_threshold:
            continue

        suggestions.append({
            "chunk_id": chunk["id"],
            "document_id": chunk["document_id"],
            "document_title": document["title"],
            "chunk_index": chunk["chunk_index"],
            "content_preview": chunk["content"][:500],
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "similarity": round(similarity, 4),
        })

    suggestions.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return {
        "topic_id": str(topic_id),
        "suggestions": suggestions[:limit],
        "threshold": settings.material_similarity_threshold,
    }


def create_topic_chunk_link(
    user_id,
    course_id,
    topic_id,
    chunk_id,
    method="manual",
    similarity=None
):
    if method not in {"manual", "ai"}:
        raise ValueError("Method must be manual or ai.")

    chunk = _one(
        _db().table("document_chunks")
        .select("id, document_id")
        .eq("id", str(chunk_id))
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
    )

    if not chunk:
        raise ValueError("Lecture chunk not found in this course.")

    topic = _one(
        _db().table("syllabus_topics")
        .select("id")
        .eq("id", str(topic_id))
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
    )

    if not topic:
        raise ValueError("Topic not found in this course.")

    payload = {
        "topic_id": str(topic_id),
        "chunk_id": str(chunk_id),
        "document_id": chunk["document_id"],
        "course_id": str(course_id),
        "user_id": str(user_id),
        "method": method,
        "similarity": similarity,
    }

    result = _one(
        _db().table("topic_chunk_links")
        .upsert(
            payload,
            on_conflict="topic_id,chunk_id"
        )
        .select("*")
    )

    return result


def get_topic_coverage(user_id, course_id):
    topics = _course_topics(user_id, course_id)

    links = _rows(
        _db().table("topic_chunk_links")
        .select("topic_id, chunk_id, document_id, similarity, method")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
    )

    grouped = defaultdict(list)

    for link in links:
        grouped[str(link["topic_id"])].append(link)

    coverage = []

    for topic in topics:
        topic_id = str(topic["id"])
        topic_links = grouped.get(topic_id, [])
        count = len(topic_links)

        if count == 0:
            status = "no_material"
        elif count < settings.enough_material_chunk_count:
            status = "thin"
        else:
            status = "enough"

        coverage.append({
            "topic_id": topic_id,
            "topic_title": topic["title"],
            "status": status,
            "linked_chunk_count": count,
            "links": topic_links,
        })

    return {
        "threshold_for_enough": settings.enough_material_chunk_count,
        "topics": coverage,
    }


def detect_course_gaps(user_id, course_id):
    topics = _course_topics(user_id, course_id)
    stats = _topic_stats(user_id, course_id)
    coverage = get_topic_coverage(user_id, course_id)

    coverage_map = {
        item["topic_id"]: item
        for item in coverage["topics"]
    }

    gaps = []

    for topic in topics:
        topic_id = str(topic["id"])
        stat = stats.get(topic_id, {})
        material = coverage_map.get(topic_id, {})

        frequency = float(
            stat.get("paper_frequency_percentage") or 0
        )

        papers = int(
            stat.get("papers_with_topic") or 0
        )

        total_papers = int(
            stat.get("total_ready_papers") or 0
        )

        attempts = int(
            stat.get("attempted_questions") or 0
        )

        accuracy = stat.get("accuracy_percentage")

        common = {
            "topic_id": topic_id,
            "topic_title": topic["title"],
            "frequency_percentage": frequency,
            "papers_with_topic": papers,
            "total_ready_papers": total_papers,
            "practice_attempts": attempts,
            "practice_accuracy": accuracy,
            "material_status": material.get("status", "no_material"),
        }

        if material.get("status") == "no_material":
            gaps.append({
                **common,
                "gap_type": "no_material",
                "severity": "high",
                "reason": (
                    "No lecture chunks have been linked to this topic."
                ),
                "recommended_action": (
                    "Upload lecture material or map relevant lecture passages."
                ),
            })

        elif material.get("status") in {"thin"}:
            gaps.append({
                **common,
                "gap_type": "thin_material",
                "severity": "medium",
                "reason": (
                    "Only a small amount of lecture material is linked "
                    "to this topic."
                ),
                "recommended_action": (
                    "Add more relevant lecture passages."
                ),
            })

        if (
            material.get("status") != "no_material"
            and attempts == 0
        ):
            gaps.append({
                **common,
                "gap_type": "material_but_no_practice",
                "severity": "medium",
                "reason": (
                    "Lecture material is available, but no practice "
                    "attempts have been recorded for this topic."
                ),
                "recommended_action": (
                    "Practise questions from this topic."
                ),
            })

        if (
            frequency >= settings.often_examined_frequency_percentage
            and accuracy is not None
            and float(accuracy) < 50
        ):
            gaps.append({
                **common,
                "gap_type": "often_examined_but_weak",
                "severity": "high",
                "reason": (
                    f"This topic appears in {frequency:.1f}% of reviewed "
                    "past papers, but practice accuracy is below 50%."
                ),
                "recommended_action": (
                    "Prioritise revision and solve more questions."
                ),
            })

        if (
            total_papers > 0
            and papers == 0
        ):
            gaps.append({
                **common,
                "gap_type": "never_assessed",
                "severity": "low",
                "reason": (
                    "No reviewed past-paper questions are mapped to "
                    "this topic, although ready past papers exist."
                ),
                "recommended_action": (
                    "Review past-paper questions and map relevant ones "
                    "to this topic."
                ),
            })

    return {
        "gap_count": len(gaps),
        "gaps": gaps,
    }
