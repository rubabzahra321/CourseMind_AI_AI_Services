
import json

import pytest

from app.core.errors import PermanentJobError
from app.repositories import (
    past_paper_repository,
    syllabus_topic_repository
)
from app.services import processing_service

from tests.conftest import COURSE, DOC, USER


PDF_PATH = f"{USER}/{COURSE}/{DOC}.pdf"


def seed_ready_doc(db, kind, academic_year=2024):

    db.seed("documents", {
        "id": DOC,
        "user_id": USER,
        "course_id": COURSE,
        "title": "Source paper",
        "storage_path": PDF_PATH,
        "status": "ready",
        "document_type": kind,
        "academic_year": academic_year,
    })

    db.files[PDF_PATH] = (
        b"not used because page extraction is mocked"
    )

    return {
        "id": "job-id",
        "document_id": DOC,
        "user_id": USER,
        "course_id": COURSE,
        "job_type": (
            "extract_syllabus_topics"
            if kind == "syllabus"
            else "extract_past_paper_questions"
        ),
        "status": "processing",
        "attempts": 1,
        "max_attempts": 3
    }


def test_syllabus_repository_saves_hierarchy_and_review_flags(db):

    count = syllabus_topic_repository.replace_extracted_topics(
        USER,
        COURSE,
        DOC,
        [{
            "title": "Unit 1",
            "page_start": 1,
            "page_end": 3,
            "topics": [{
                "title": "Arrays",
                "page_start": 1,
                "page_end": 2,
                "subtopics": [{
                    "title": "Traversal",
                    "page_start": 2,
                    "page_end": 2
                }]
            }]
        }]
    )

    rows = db.rows(
        "syllabus_topics",
        source_document_id=DOC
    )

    assert count == 3
    assert len(rows) == 3

    chapter = next(
        row for row in rows
        if row["title"] == "Unit 1"
    )

    topic = next(
        row for row in rows
        if row["title"] == "Arrays"
    )

    subtopic = next(
        row for row in rows
        if row["title"] == "Traversal"
    )

    assert topic["parent_topic_id"] == chapter["id"]
    assert subtopic["parent_topic_id"] == topic["id"]

    assert all(
        row["origin"] == "ai_extracted"
        and row["reviewed"] is False
        for row in rows
    )


def test_syllabus_retry_replaces_unreviewed_ai_but_preserves_reviewed_and_manual(db):

    syllabus_topic_repository.replace_extracted_topics(
        USER,
        COURSE,
        DOC,
        [{
            "title": "Old",
            "topics": []
        }]
    )

    old = db.rows(
        "syllabus_topics",
        source_document_id=DOC
    )[0]

    db.table("syllabus_topics").update(
        {"reviewed": True}
    ).eq("id", old["id"]).execute()

    db.seed("syllabus_topics", {
        "user_id": USER,
        "course_id": COURSE,
        "source_document_id": DOC,
        "title": "Manual",
        "origin": "manual",
        "reviewed": True
    })

    syllabus_topic_repository.replace_extracted_topics(
        USER,
        COURSE,
        DOC,
        [{
            "title": "New",
            "topics": []
        }]
    )

    titles = {
        r["title"]
        for r in db.rows(
            "syllabus_topics",
            source_document_id=DOC
        )
    }

    assert titles == {
        "Old",
        "Manual",
        "New"
    }


def test_syllabus_worker_extracts_and_saves(db, monkeypatch):

    job = seed_ready_doc(
        db,
        "syllabus"
    )

    monkeypatch.setattr(
        processing_service,
        "_read_text_pages",
        lambda _doc: [
            (1, "Syllabus text long enough")
        ]
    )

    monkeypatch.setattr(
        processing_service,
        "extract_syllabus_topics",
        lambda _pages: {
            "chapters": [{
                "title": "Chapter 1",
                "topics": [{
                    "title": "Topic 1",
                    "subtopics": []
                }]
            }]
        }
    )

    monkeypatch.setattr(
        processing_service.job_repository,
        "complete",
        lambda _id: None
    )

    processing_service.run_job(job)

    assert {
        r["title"]
        for r in db.rows(
            "syllabus_topics",
            source_document_id=DOC
        )
    } == {
        "Chapter 1",
        "Topic 1"
    }


def test_past_paper_repository_marks_pending_and_retry_safe(db):

    questions = [{
        "question_number": "Q1",
        "question_text": "Define an array.",
        "question_type": "short",
        "marks": 5,
        "paper_year": 2024,
        "page_number": 1,
        "extraction_confidence": 0.9
    }]

    assert (
        past_paper_repository.replace_pending_extracted_questions(
            USER,
            COURSE,
            DOC,
            questions
        ) == 1
    )

    row = db.rows(
        "past_paper_questions",
        document_id=DOC
    )[0]

    assert row["review_status"] == "pending"

    assert (
        row["source_metadata"]["extraction_source"]
        == "ai_extracted"
    )

    db.table("past_paper_questions").update({
        "review_status": "reviewed",
        "question_text": "Corrected by user"
    }).eq("id", row["id"]).execute()

    past_paper_repository.replace_pending_extracted_questions(
        USER,
        COURSE,
        DOC,
        questions
    )

    rows = db.rows(
        "past_paper_questions",
        document_id=DOC
    )

    assert len(rows) == 2

    assert any(
        r["question_text"] == "Corrected by user"
        and r["review_status"] == "reviewed"
        for r in rows
    )


def test_past_paper_worker_extracts_and_saves_pending(db, monkeypatch):

    job = seed_ready_doc(
        db,
        "past_paper",
        2023
    )

    monkeypatch.setattr(
        processing_service,
        "_read_text_pages",
        lambda _doc: [
            (2, "Past paper question text")
        ]
    )

    monkeypatch.setattr(
        processing_service,
        "extract_past_paper_questions",
        lambda _pages, year: [{
            "question_number": "2(a)",
            "question_text": "Explain binary search.",
            "question_type": "long",
            "marks": 8,
            "paper_year": year,
            "page_number": 2,
            "extraction_confidence": 0.95,
        }]
    )

    monkeypatch.setattr(
        processing_service.job_repository,
        "complete",
        lambda _id: None
    )

    job["job_type"] = "extract_past_paper_questions"

    processing_service.run_job(job)

    rows = db.rows(
        "past_paper_questions",
        document_id=DOC
    )

    assert len(rows) == 1
    assert rows[0]["review_status"] == "pending"
    assert rows[0]["paper_year"] == 2023
    assert rows[0]["page_number"] == 2


def test_wrong_document_type_fails_extraction_job(db, monkeypatch):

    job = seed_ready_doc(
        db,
        "lecture_material"
    )

    monkeypatch.setattr(
        processing_service.job_repository,
        "fail",
        lambda *_: None
    )

    processing_service.run_job(job)

    assert db.rows(
        "syllabus_topics",
        source_document_id=DOC
    ) == []


def test_syllabus_extractor_accepts_valid_gemini_json(monkeypatch):

    from types import SimpleNamespace
    from app.ai import syllabus_extractor

    monkeypatch.setattr(
        syllabus_extractor,
        "get_gemini_client",
        lambda: SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **_kwargs:
                    SimpleNamespace(
                        text=json.dumps({
                            "chapters": [{
                                "title": "Unit 1",
                                "topics": []
                            }]
                        })
                    )
            )
        )
    )

    result = syllabus_extractor.extract_syllabus_topics(
        [(1, "Syllabus content")]
    )

    assert (
        result["chapters"][0]["title"]
        == "Unit 1"
    )


def test_syllabus_extractor_rejects_malformed_gemini_json(monkeypatch):

    from types import SimpleNamespace
    from app.ai import syllabus_extractor

    monkeypatch.setattr(
        syllabus_extractor,
        "get_gemini_client",
        lambda: SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **_kwargs:
                    SimpleNamespace(text="not json")
            )
        )
    )

    with pytest.raises(
        PermanentJobError,
        match="invalid syllabus"
    ):

        syllabus_extractor.extract_syllabus_topics(
            [(1, "Syllabus content")]
        )


def test_past_paper_extractor_parses_fields_and_fallback_year(monkeypatch):

    from types import SimpleNamespace
    from app.ai import past_paper_extractor

    output = {
        "questions": [{
            "question_number": "Q1",
            "question_text": "Define a stack.",
            "question_type": "short",
            "marks": 4,
            "paper_year": None,
            "page_number": 2,
            "confidence": 0.91
        }]
    }

    monkeypatch.setattr(
        past_paper_extractor,
        "get_gemini_client",
        lambda: SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **_kwargs:
                    SimpleNamespace(
                        text=json.dumps(output)
                    )
            )
        )
    )

    result = past_paper_extractor.extract_past_paper_questions(
        [(2, "Question text")],
        2022
    )

    assert result[0]["paper_year"] == 2022
    assert result[0]["marks"] == 4
    assert result[0]["page_number"] == 2


def test_past_paper_extractor_rejects_malformed_output(monkeypatch):

    from types import SimpleNamespace
    from app.ai import past_paper_extractor

    monkeypatch.setattr(
        past_paper_extractor,
        "get_gemini_client",
        lambda: SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **_kwargs:
                    SimpleNamespace(
                        text='{"questions": []}'
                    )
            )
        )
    )

    with pytest.raises(
        PermanentJobError,
        match="invalid past-paper"
    ):

        past_paper_extractor.extract_past_paper_questions(
            [(1, "Question text")]
        )


def test_enqueue_is_idempotent_for_active_document_job(db):

    from app.repositories import job_repository

    document = {
        "id": DOC,
        "user_id": USER,
        "course_id": COURSE
    }

    first = job_repository.enqueue(
        document,
        "extract_syllabus_topics",
        "stable-key"
    )

    second = job_repository.enqueue(
        document,
        "extract_syllabus_topics",
        "stable-key"
    )

    assert first["id"] == second["id"]

    assert len(
        db.rows(
            "processing_jobs",
            document_id=DOC,
            job_type="extract_syllabus_topics"
        )
    ) == 1


def test_document_worker_automatically_enqueues_syllabus_extraction(db, monkeypatch):

    from datetime import datetime, timezone
    from app.repositories import job_repository

    seed_ready_doc(
        db,
        "syllabus"
    )

    monkeypatch.setattr(
        processing_service,
        "_process_document",
        lambda _job: None
    )

    monkeypatch.setattr(
        processing_service.document_repository,
        "get",
        lambda *_args: {
            "id": DOC,
            "user_id": USER,
            "course_id": COURSE,
            "document_type": "syllabus",
            "processed_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "status": "ready",
        }
    )

    monkeypatch.setattr(
        job_repository,
        "complete",
        lambda _id: None
    )

    job = {
        "id": "process-job",
        "document_id": DOC,
        "user_id": USER,
        "course_id": COURSE,
        "job_type": "process_document",
        "status": "processing",
        "attempts": 1,
        "max_attempts": 3
    }

    processing_service.run_job(job)

    queued = db.rows(
        "processing_jobs",
        document_id=DOC,
        job_type="extract_syllabus_topics"
    )

    assert len(queued) == 1
    assert queued[0]["status"] == "queued"


def test_document_worker_automatically_enqueues_past_paper_extraction(db, monkeypatch):

    from datetime import datetime, timezone
    from app.repositories import job_repository

    seed_ready_doc(
        db,
        "past_paper"
    )

    monkeypatch.setattr(
        processing_service,
        "_process_document",
        lambda _job: None
    )

    monkeypatch.setattr(
        processing_service.document_repository,
        "get",
        lambda *_args: {
            "id": DOC,
            "user_id": USER,
            "course_id": COURSE,
            "document_type": "past_paper",
            "processed_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "status": "ready",
        }
    )

    monkeypatch.setattr(
        job_repository,
        "complete",
        lambda _id: None
    )

    job = {
        "id": "process-job",
        "document_id": DOC,
        "user_id": USER,
        "course_id": COURSE,
        "job_type": "process_document",
        "status": "processing",
        "attempts": 1,
        "max_attempts": 3
    }

    processing_service.run_job(job)

    queued = db.rows(
        "processing_jobs",
        document_id=DOC,
        job_type="extract_past_paper_questions"
    )

    assert len(queued) == 1
    assert queued[0]["status"] == "queued"
