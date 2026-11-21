
import logging
from datetime import datetime, timezone

from app.core.config import settings
from app.core.errors import PermanentJobError

from app.repositories import (
    chunk_repository,
    document_repository,
    job_repository,
    storage_repository,
    embedding_repository,
    syllabus_topic_repository,
    past_paper_repository,
)

from app.services import chunking, pdf_service

from app.ai.gemini import embed_documents
from app.ai.syllabus_extractor import extract_syllabus_topics
from app.ai.past_paper_extractor import extract_past_paper_questions
from app.repositories import notification_repository

log = logging.getLogger("worker")

SAFE_FAILURE = (
    "Document processing failed after several attempts. "
    "You can retry."
)


def _get_document(job: dict) -> dict:

    doc = document_repository.get(
        job["document_id"],
        job["user_id"],
        job["course_id"]
    )

    if not doc:
        raise PermanentJobError(
            "Document no longer exists."
        )

    return doc


def _read_text_pages(doc: dict) -> list[tuple[int, str]]:

    pdf_bytes = storage_repository.download(
        doc["storage_path"]
    )

    pages = pdf_service.extract_pages(pdf_bytes)

    text_pages = [
        (number, text)
        for number, text in pages
        if len(text) >= settings.min_page_chars
    ]

    if not text_pages:
        raise PermanentJobError(
            "No readable text found. This may be a scanned PDF; "
            "OCR is not implemented yet."
        )

    return text_pages


def _process_document(job: dict) -> None:

    doc = _get_document(job)

    document_repository.update(
        doc["id"],
        {
            "status": "processing",
            "processing_error": None
        }
    )

    pdf_bytes = storage_repository.download(
        doc["storage_path"]
    )

    all_pages = pdf_service.extract_pages(pdf_bytes)

    pages = [
        (number, text)
        for number, text in all_pages
        if len(text) >= settings.min_page_chars
    ]

    if not pages:
        raise PermanentJobError(
            "No readable text found. This may be a scanned PDF; "
            "OCR is not implemented yet."
        )

    chunks = chunking.build_chunks(
        pages,
        settings.chunk_size_chars,
        settings.chunk_overlap_chars
    )

    if not chunks:
        raise PermanentJobError(
            "No text chunks were generated."
        )

    rows = [
        {
            **chunk,
            "document_id": doc["id"],
            "course_id": doc["course_id"],
            "user_id": doc["user_id"]
        }
        for chunk in chunks
    ]

    chunk_repository.delete_for_document(
        doc["id"],
        doc["user_id"]
    )

    vectors = []

    for start in range(0, len(rows), 32):

        batch = rows[start:start + 32]

        vectors.extend(
            embed_documents(
                [item["content"] for item in batch]
            )
        )

    embedding_repository.save_embeddings(
        rows,
        vectors
    )

    document_repository.update(
        doc["id"],
        {
            "status": "ready",
            "page_count": len(all_pages),
            "processed_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "processing_error": None,
        }
    )


def _process_syllabus_topics(job: dict) -> None:

    doc = _get_document(job)

    if doc.get("document_type") != "syllabus":
        raise PermanentJobError(
            "This document is not a syllabus."
        )

    if doc.get("status") != "ready":
        raise RuntimeError(
            "Syllabus document is not ready for extraction."
        )

    extracted = extract_syllabus_topics(
        _read_text_pages(doc)
    )

    count = syllabus_topic_repository.replace_extracted_topics(
        job["user_id"],
        job["course_id"],
        doc["id"],
        extracted["chapters"]
    )

    if count < 1:
        raise PermanentJobError(
            "No syllabus topics were extracted."
        )

    log.info(
        "saved %s syllabus nodes for document %s",
        count,
        doc["id"]
    )


def _process_past_paper_questions(job: dict) -> None:

    doc = _get_document(job)

    if doc.get("document_type") != "past_paper":
        raise PermanentJobError(
            "This document is not a past paper."
        )

    if doc.get("status") != "ready":
        raise RuntimeError(
            "Past-paper document is not ready for extraction."
        )

    questions = extract_past_paper_questions(
        _read_text_pages(doc),
        doc.get("academic_year")
    )

    count = past_paper_repository.replace_pending_extracted_questions(
        job["user_id"],
        job["course_id"],
        doc["id"],
        questions
    )

    if count < 1:
        raise PermanentJobError(
            "No past-paper questions were extracted."
        )

    log.info(
        "saved %s past-paper questions for document %s",
        count,
        doc["id"]
    )


def _enqueue_extraction_after_document_ready(job: dict) -> None:

    doc = _get_document(job)

    job_type_by_document = {
        "syllabus": "extract_syllabus_topics",
        "past_paper": "extract_past_paper_questions",
    }

    next_type = job_type_by_document.get(
        doc.get("document_type")
    )

    if next_type:

        processed_at = doc.get("processed_at") or "ready"

        job_repository.enqueue(
            doc,
            next_type,
            f"{doc['id']}:{next_type}:{processed_at}"
        )


def _fail(
    job: dict,
    job_error: str,
    user_message: str
) -> None:

    job_repository.fail(
        job["id"],
        job_error
    )

    if job["job_type"] == "process_document":

        document_repository.update(
            job["document_id"],
            {
                "status": "failed",
                "processing_error": user_message
            }
        )


def _retry(job: dict, detail: str) -> None:

    if job["job_type"] == "process_document":

        document_repository.update(
            job["document_id"],
            {"status": "queued"}
        )

    job_repository.retry_later(
        job["id"],
        detail,
        delay_seconds=30 * job["attempts"]
    )


def run_job(job: dict) -> None:

    try:

        handlers = {
            "process_document": _process_document,
            "extract_syllabus_topics": _process_syllabus_topics,
            "extract_past_paper_questions": (
                _process_past_paper_questions
            ),
        }

        handler = handlers.get(
            job["job_type"]
        )

        if handler is None:
            raise PermanentJobError(
                f"Unsupported job type: {job['job_type']}"
            )

        handler(job)

        job_repository.complete(
            job["id"]
        )

        if job["job_type"] == "process_document":

            _enqueue_extraction_after_document_ready(
                job
            )

        log.info(
            "job %s completed",
            job["id"]
        )

    except PermanentJobError as exc:

        log.warning(
            "job %s failed permanently: %s",
            job["id"],
            exc
        )

        _fail(
            job,
            str(exc),
            str(exc)
        )

    except Exception as exc:

        log.exception(
            "job %s crashed",
            job["id"]
        )

        detail = f"{type(exc).__name__}: {exc}"

        if job["attempts"] >= job["max_attempts"]:

            _fail(
                job,
                detail,
                SAFE_FAILURE
            )

        else:

            _retry(
                job,
                detail
            )


def recover_stale_jobs() -> None:

    for job in job_repository.find_stale(
        settings.worker_stale_minutes
    ):

        log.warning(
            "recovering stale job %s",
            job["id"]
        )

        if job["attempts"] >= job["max_attempts"]:

            _fail(
                job,
                "Worker stopped while processing",
                SAFE_FAILURE
            )

        else:

            _retry(
                job,
                "Worker stopped while processing"
            )
