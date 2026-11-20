
from app.services import analytics_service

from tests.conftest import (
    USER,
    COURSE,
    TOPIC,
    DOC,
)


def test_topic_evidence_returns_reviewed_questions(db):

    db.seed("documents", {
        "id": DOC,
        "user_id": USER,
        "course_id": COURSE,
        "title": "Past Paper 2024",
        "original_filename": "paper.pdf",
        "document_type": "past_paper",
        "academic_year": 2024,
        "status": "ready",
    })

    question = db.seed("past_paper_questions", {
        "user_id": USER,
        "course_id": COURSE,
        "document_id": DOC,
        "question_number": "Q1",
        "question_text": "Explain binary search.",
        "marks": 5,
        "paper_year": 2024,
        "page_number": 2,
        "review_status": "reviewed",
    })

    db.seed("question_topic_mappings", {
        "user_id": USER,
        "topic_id": TOPIC,
        "past_paper_question_id": question["id"],
        "mapping_method": "manual",
    })

    result = analytics_service.get_topic_evidence(
        USER,
        COURSE,
        TOPIC
    )

    assert result["question_count"] == 1
    assert result["frequency"]["papers_with_topic"] == 1
    assert result["papers"][0]["year"] == 2024
    assert result["papers"][0]["questions"][0]["page_number"] == 2


def test_evidence_contains_data_sufficiency_warning(db):

    db.seed("syllabus_topics", {
        "id": TOPIC,
        "user_id": USER,
        "course_id": COURSE,
        "title": "Binary Trees",
    })

    result = analytics_service.get_topic_evidence(
        USER,
        COURSE,
        TOPIC
    )

    assert result["frequency"]["data_sufficient"] is False
    assert result["frequency"]["warning"] is not None


def test_priority_weights_are_normalised():

    weights = {
        "frequency": 0.5,
        "marks": 0.2,
        "syllabus_weight": 0.2,
        "mastery": 0.1,
    }

    result = analytics_service._normalise_weights(
        weights
    )

    assert round(sum(result.values()), 5) == 1.0
    assert result["frequency"] == 0.5


def test_priority_weights_reject_zero_total():

    try:
        analytics_service._normalise_weights({
            "frequency": 0,
            "marks": 0,
            "syllabus_weight": 0,
            "mastery": 0,
        })
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_cosine_similarity_returns_one_for_same_vector():

    result = analytics_service._cosine_similarity(
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    )

    assert round(result, 5) == 1.0


def test_coverage_marks_topic_with_no_links_as_no_material(
    db,
    world
):

    result = analytics_service.get_topic_coverage(
        world.user,
        world.course
    )

    topic = next(
        item for item in result["topics"]
        if item["topic_id"] == world.topic
    )

    assert topic["status"] == "no_material"
    assert topic["linked_chunk_count"] == 0


def test_manual_topic_chunk_mapping_is_saved(db, world):

    chunk = db.rows(
        "document_chunks",
        document_id=world.doc
    )[0]

    result = analytics_service.create_topic_chunk_link(
        world.user,
        world.course,
        world.topic,
        chunk["id"],
        method="manual"
    )

    assert result["topic_id"] == world.topic
    assert result["chunk_id"] == chunk["id"]
    assert result["method"] == "manual"


def test_coverage_becomes_enough_after_three_links(db, world):

    chunks = db.rows(
        "document_chunks",
        document_id=world.doc
    )

    for chunk in chunks:
        analytics_service.create_topic_chunk_link(
            world.user,
            world.course,
            world.topic,
            chunk["id"],
            method="manual"
        )

    result = analytics_service.get_topic_coverage(
        world.user,
        world.course
    )

    topic = next(
        item for item in result["topics"]
        if item["topic_id"] == world.topic
    )

    assert topic["status"] == "enough"
    assert topic["linked_chunk_count"] == 3


def test_gap_detector_flags_no_material(db, world):

    result = analytics_service.detect_course_gaps(
        world.user,
        world.course
    )

    types = {
        gap["gap_type"]
        for gap in result["gaps"]
    }

    assert "no_material" in types


def test_gap_detector_flags_material_without_practice(
    db,
    world
):

    chunks = db.rows(
        "document_chunks",
        document_id=world.doc
    )

    analytics_service.create_topic_chunk_link(
        world.user,
        world.course,
        world.topic,
        chunks[0]["id"],
        method="manual"
    )

    result = analytics_service.detect_course_gaps(
        world.user,
        world.course
    )

    types = {
        gap["gap_type"]
        for gap in result["gaps"]
    }

    assert "material_but_no_practice" in types


def test_similarity_suggestions_only_return_lecture_material(
    db,
    world,
    monkeypatch
):

    from app.core.config import settings

    # Give the lecture chunks matching embeddings.
    chunks = db.rows(
        "document_chunks",
        document_id=world.doc
    )

    for chunk in chunks:
        (
            db.table("document_chunks")
            .update({
                "embedding": "[1.0,0.0,0.0]"
            })
            .eq("id", chunk["id"])
            .execute()
        )

    monkeypatch.setattr(
        analytics_service,
        "embed_query",
        lambda _text: [1.0, 0.0, 0.0]
    )

    monkeypatch.setattr(
        settings,
        "material_similarity_threshold",
        0.5
    )

    result = analytics_service.suggest_topic_chunks(
        world.user,
        world.course,
        world.topic
    )

    assert len(result["suggestions"]) == 3
    assert result["suggestions"][0]["similarity"] == 1.0


def test_gap_detector_flags_often_examined_weak_topic(
    db,
    world
):

    db.seed("topic_ai_stats", {
        "topic_id": world.topic,
        "course_id": world.course,
        "user_id": world.user,
        "topic_title": "Binary Trees",
        "papers_with_topic": 4,
        "total_ready_papers": 5,
        "paper_frequency_percentage": 80,
        "attempted_questions": 5,
        "correct_answers": 1,
        "accuracy_percentage": 20,
        "performance_label": "Weak",
    })

    chunk = db.rows(
        "document_chunks",
        document_id=world.doc
    )[0]

    analytics_service.create_topic_chunk_link(
        world.user,
        world.course,
        world.topic,
        chunk["id"],
        method="manual"
    )

    result = analytics_service.detect_course_gaps(
        world.user,
        world.course
    )

    types = {
        gap["gap_type"]
        for gap in result["gaps"]
    }

    assert "often_examined_but_weak" in types
