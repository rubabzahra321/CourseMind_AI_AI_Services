"""
K-03  Paper insights   - how often and how heavily each topic is examined,
                         shown only when there are enough past papers.
S-01  Knowledge map    - for every syllabus topic: lecture passages, past-paper
                         questions, practice questions and mastery in one place.
"""

from collections import defaultdict

from app.core.config import settings
from app.db.client import get_db


PAGE_SIZE = 1000          # PostgREST returns at most 1000 rows per request
DETAIL_LIMIT_DEFAULT = 20
EXCERPT_CHARS = 300


# ---------------------------------------------------------------- helpers
def _all_rows(build_query):
    """Read every row of a query (build_query() must return a fresh query)."""

    rows = []
    start = 0

    while True:
        page = (
            build_query()
            .range(start, start + PAGE_SIZE - 1)
            .execute()
            .data
            or []
        )

        rows.extend(page)

        if len(page) < PAGE_SIZE:
            return rows

        start += PAGE_SIZE


def _topics(user_id, course_id):
    return _all_rows(
        lambda: get_db().table("syllabus_topics")
        .select("id, title, description, parent_topic_id, position")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .order("position")
    )


def _ready_past_papers(user_id, course_id):
    return _all_rows(
        lambda: get_db().table("documents")
        .select("id, title, original_filename, academic_year")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .eq("document_type", "past_paper")
        .eq("status", "ready")
    )


def _reviewed_questions(user_id, course_id):
    return _all_rows(
        lambda: get_db().table("past_paper_questions")
        .select(
            "id, document_id, question_number, question_text, marks, "
            "paper_year, page_number"
        )
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .eq("review_status", "reviewed")
    )


def _question_ids_by_topic(user_id, topic_ids):
    """topic_id -> set(past_paper_question_id)."""

    mappings = _all_rows(
        lambda: get_db().table("question_topic_mappings")
        .select("past_paper_question_id, topic_id")
        .eq("user_id", str(user_id))
    )

    wanted = set(topic_ids)
    result = defaultdict(set)

    for row in mappings:
        topic_id = str(row["topic_id"])

        if topic_id in wanted:
            result[topic_id].add(str(row["past_paper_question_id"]))

    return result


def _past_question_index(user_id, course_id):
    """Reviewed questions of READY past papers: id -> question (+ paper)."""

    papers = {str(p["id"]): p for p in _ready_past_papers(user_id, course_id)}
    index = {}

    for question in _reviewed_questions(user_id, course_id):
        paper = papers.get(str(question["document_id"]))

        if not paper:
            continue

        index[str(question["id"])] = {
            "question_id": question["id"],
            "question_number": question.get("question_number"),
            "question_text": question["question_text"],
            "marks": question.get("marks"),
            "paper_year": question.get("paper_year") or paper.get("academic_year"),
            "page_number": question.get("page_number"),
            "document_id": paper["id"],
            "document_title": paper.get("title") or paper.get("original_filename"),
        }

    return papers, index


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- K-03
def get_paper_insights(user_id, course_id):
    """
    Per-topic exam frequency and marks.

    Percentages and averages are only returned when the course has at least
    `min_papers_for_frequency` ready past papers. Below that, only the plain
    counts are shown, because a percentage from one paper would mislead.
    """

    topics = _topics(user_id, course_id)
    papers, index = _past_question_index(user_id, course_id)
    mapped = _question_ids_by_topic(user_id, [str(t["id"]) for t in topics])

    total_papers = len(papers)
    minimum = settings.min_papers_for_frequency
    reliable = total_papers >= minimum

    warning = None

    if total_papers == 0:
        warning = (
            "No past papers are ready yet, so there is no exam "
            "frequency data."
        )
    elif not reliable:
        warning = (
            f"Only {total_papers} ready past paper(s) are available. "
            f"At least {minimum} are needed before frequency and marks "
            "are shown, because results from so few papers can mislead."
        )

    insights = []

    for topic in topics:
        topic_id = str(topic["id"])
        questions = [
            index[qid] for qid in mapped.get(topic_id, set()) if qid in index
        ]

        paper_ids = {str(q["document_id"]) for q in questions}
        years = sorted({q["paper_year"] for q in questions if q["paper_year"]})
        marks = [m for m in (_number(q["marks"]) for q in questions) if m is not None]

        item = {
            "topic_id": topic_id,
            "topic_title": topic["title"],
            "parent_topic_id": topic.get("parent_topic_id"),
            "question_count": len(questions),
            "papers_with_topic": len(paper_ids),
            "years": years,
            "reliable": reliable,
            "frequency_percentage": None,
            "often_examined": None,
            "marks": None,
        }

        if reliable:
            frequency = 100.0 * len(paper_ids) / total_papers

            item["frequency_percentage"] = round(frequency, 2)
            item["often_examined"] = (
                frequency >= settings.often_examined_frequency_percentage
            )
            item["marks"] = {
                "total": round(sum(marks), 2) if marks else None,
                "average": round(sum(marks) / len(marks), 2) if marks else None,
                "highest": max(marks) if marks else None,
                "questions_with_marks": len(marks),
                "questions_without_marks": len(questions) - len(marks),
            }

        insights.append(item)

    insights.sort(
        key=lambda i: (
            -(i["frequency_percentage"] or 0),
            -i["papers_with_topic"],
            -i["question_count"],
            i["topic_title"].lower(),
        )
    )

    return {
        "reliable": reliable,
        "warning": warning,
        "total_ready_papers": total_papers,
        "minimum_papers_required": minimum,
        "often_examined_threshold": settings.often_examined_frequency_percentage,
        "note": (
            "Counts include only reviewed questions mapped directly "
            "to each topic."
        ),
        "topics": insights,
    }


# ---------------------------------------------------------------- S-01
def _material_status(count):
    if count == 0:
        return "no_material"

    if count < settings.enough_material_chunk_count:
        return "thin"

    return "enough"


def _progress_by_topic(user_id, course_id):
    rows = _all_rows(
        lambda: get_db().table("topic_progress")
        .select(
            "topic_id, attempts, correct_answers, accuracy_percentage, "
            "last_practised_at, status"
        )
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
    )

    return {str(row["topic_id"]): row for row in rows}


def _mastery(progress):
    if not progress:
        return {
            "status": "Not Yet Assessed",
            "accuracy_percentage": None,
            "attempts": 0,
            "last_practised_at": None,
        }

    return {
        "status": progress.get("status") or "Not Yet Assessed",
        "accuracy_percentage": progress.get("accuracy_percentage"),
        "attempts": progress.get("attempts") or 0,
        "last_practised_at": progress.get("last_practised_at"),
    }


def _practice_counts(user_id, course_id):
    rows = _all_rows(
        lambda: get_db().table("generated_questions")
        .select("id, topic_id")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .eq("status", "active")
    )

    counts = defaultdict(int)

    for row in rows:
        if row.get("topic_id"):
            counts[str(row["topic_id"])] += 1

    return counts


def _link_counts(user_id, course_id):
    rows = _all_rows(
        lambda: get_db().table("topic_chunk_links")
        .select("topic_id")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
    )

    counts = defaultdict(int)

    for row in rows:
        counts[str(row["topic_id"])] += 1

    return counts


def get_knowledge_map(user_id, course_id):
    """Course-wide map: one row per topic linking all four kinds of evidence."""

    topics = _topics(user_id, course_id)
    topic_ids = [str(t["id"]) for t in topics]

    _papers, index = _past_question_index(user_id, course_id)
    mapped = _question_ids_by_topic(user_id, topic_ids)
    passages = _link_counts(user_id, course_id)
    practice = _practice_counts(user_id, course_id)
    progress = _progress_by_topic(user_id, course_id)

    rows = []

    for topic in topics:
        topic_id = str(topic["id"])
        past_count = sum(1 for qid in mapped.get(topic_id, set()) if qid in index)

        rows.append({
            "topic_id": topic_id,
            "topic_title": topic["title"],
            "parent_topic_id": topic.get("parent_topic_id"),
            "position": topic.get("position"),
            "passages": {
                "count": passages.get(topic_id, 0),
                "status": _material_status(passages.get(topic_id, 0)),
            },
            "past_paper_questions": {"count": past_count},
            "practice_questions": {"count": practice.get(topic_id, 0)},
            "mastery": _mastery(progress.get(topic_id)),
        })

    return {
        "totals": {
            "topics": len(rows),
            "topics_with_material": sum(1 for r in rows if r["passages"]["count"] > 0),
            "topics_with_past_questions": sum(
                1 for r in rows if r["past_paper_questions"]["count"] > 0
            ),
            "topics_with_practice": sum(
                1 for r in rows if r["practice_questions"]["count"] > 0
            ),
            "topics_assessed": sum(
                1 for r in rows if r["mastery"]["status"] != "Not Yet Assessed"
            ),
        },
        "topics": rows,
    }


def get_topic_knowledge(user_id, course_id, topic_id, limit=DETAIL_LIMIT_DEFAULT):
    """Everything linked to ONE topic, with sources, for the detail panel."""

    topics = {str(t["id"]): t for t in _topics(user_id, course_id)}
    topic = topics.get(str(topic_id))

    if not topic:
        raise ValueError("Topic not found in this course.")

    topic_id = str(topic_id)
    limit = max(1, min(int(limit), 100))

    # ---- lecture passages
    links = _all_rows(
        lambda: get_db().table("topic_chunk_links")
        .select("chunk_id, document_id, similarity, method")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .eq("topic_id", topic_id)
    )

    links.sort(key=lambda l: -(_number(l.get("similarity")) or 0))
    shown = links[:limit]
    chunk_ids = [str(l["chunk_id"]) for l in shown]

    chunks, documents = {}, {}

    if chunk_ids:
        for row in (
            get_db().table("document_chunks")
            .select("id, document_id, content, page_start, page_end")
            .eq("user_id", str(user_id))
            .in_("id", chunk_ids)
            .execute().data or []
        ):
            chunks[str(row["id"])] = row

        doc_ids = list({str(l["document_id"]) for l in shown})

        for row in (
            get_db().table("documents")
            .select("id, title, original_filename")
            .eq("user_id", str(user_id))
            .in_("id", doc_ids)
            .execute().data or []
        ):
            documents[str(row["id"])] = row

    passages = []

    for link in shown:
        chunk = chunks.get(str(link["chunk_id"]))

        if not chunk:
            continue

        document = documents.get(str(link["document_id"]), {})

        passages.append({
            "chunk_id": link["chunk_id"],
            "document_id": link["document_id"],
            "document_title": document.get("title") or document.get("original_filename"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "similarity": link.get("similarity"),
            "method": link["method"],
            "excerpt": (chunk.get("content") or "")[:EXCERPT_CHARS],
        })

    # ---- past-paper questions
    _papers, index = _past_question_index(user_id, course_id)
    mapped = _question_ids_by_topic(user_id, [topic_id]).get(topic_id, set())
    past_all = [index[q] for q in mapped if q in index]
    past_all.sort(key=lambda q: (-(q["paper_year"] or 0), str(q["question_number"] or "")))

    # ---- practice questions (+ how often answered correctly)
    questions = _all_rows(
        lambda: get_db().table("generated_questions")
        .select("id, question_text, difficulty, source_document_id, created_at")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .eq("topic_id", topic_id)
        .eq("status", "active")
        .order("created_at", desc=True)
    )

    question_ids = [str(q["id"]) for q in questions[:limit]]
    answered, correct = defaultdict(int), defaultdict(int)

    if question_ids:
        for attempt in (
            get_db().table("practice_attempts")
            .select("question_id, is_correct")
            .eq("user_id", str(user_id))
            .in_("question_id", question_ids)
            .execute().data or []
        ):
            qid = str(attempt["question_id"])
            answered[qid] += 1
            correct[qid] += 1 if attempt.get("is_correct") else 0

    practice = [
        {
            "question_id": q["id"],
            "question_text": q["question_text"],
            "difficulty": q["difficulty"],
            "source_document_id": q.get("source_document_id"),
            "times_answered": answered.get(str(q["id"]), 0),
            "times_correct": correct.get(str(q["id"]), 0),
        }
        for q in questions[:limit]
    ]

    progress = _progress_by_topic(user_id, course_id).get(topic_id)

    return {
        "topic": {
            "id": topic_id,
            "title": topic["title"],
            "description": topic.get("description"),
            "parent_topic_id": topic.get("parent_topic_id"),
        },
        "passages": {
            "total": len(links),
            "status": _material_status(len(links)),
            "items": passages,
        },
        "past_paper_questions": {
            "total": len(past_all),
            "items": past_all[:limit],
        },
        "practice_questions": {
            "total": len(questions),
            "items": practice,
        },
        "mastery": _mastery(progress),
        "limit": limit,
    }
