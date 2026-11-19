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
)

from app.services import chunking, pdf_service
from app.ai.gemini import embed_documents


log = logging.getLogger("worker")

SAFE_FAILURE = (
    "Document processing failed after several attempts. "
    "You can retry."
)


def _process_document(job: dict) -> None:

    doc = document_repository.get(
        job["document_id"],
        job["user_id"],
        job["course_id"]
    )

    if not doc:
        raise PermanentJobError("Document no longer exists.")

    document_repository.update(
        doc["id"],
        {
            "status": "processing",
            "processing_error": None,
        }
    )

    # 1. Download original PDF.
    pdf_bytes = storage_repository.download(doc["storage_path"])

    # 2. Extract page-level text.
    pages = pdf_service.extract_pages(pdf_bytes)

    text_pages = [
        page for page in pages
        if len(page[1]) >= settings.min_page_chars
    ]

    if not text_pages:
        raise PermanentJobError(
            "No readable text found. "
            "This may be a scanned PDF. OCR is not implemented yet."
        )

    # 3. Split into page-aware chunks.
    chunks = chunking.build_chunks(
        text_pages,
        settings.chunk_size_chars,
        settings.chunk_overlap_chars,
    )

    if not chunks:
        raise PermanentJobError("No text chunks were generated.")

    # 4. Attach document ownership.
    rows = [
        {
            **chunk,
            "document_id": doc["id"],
            "course_id": doc["course_id"],
            "user_id": doc["user_id"],
        }
        for chunk in chunks
    ]

    # 5. Remove old chunks before retrying.
    chunk_repository.delete_for_document(
        doc["id"],
        doc["user_id"]
    )

    # 6. Generate Gemini embeddings in batches.
    all_embeddings = []

    batch_size = 32

    for start in range(0, len(rows), batch_size):

        batch = rows[start:start + batch_size]

        texts = [
            item["content"]
            for item in batch
        ]

        vectors = embed_documents(texts)

        all_embeddings.extend(vectors)

    # 7. Store chunks and their embeddings.
    embedding_repository.save_embeddings(
        rows,
        all_embeddings
    )

    # 8. Mark ready only after embeddings are saved.
    document_repository.update(
        doc["id"],
        {
            "status": "ready",
            "page_count": len(pages),
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "processing_error": None,
        }
    )


def _fail(job: dict, job_error: str, user_message: str) -> None:

    job_repository.fail(job["id"], job_error)

    document_repository.update(
        job["document_id"],
        {
            "status": "failed",
            "processing_error": user_message,
        }
    )


def run_job(job: dict) -> None:

    try:

        if job["job_type"] != "process_document":
            raise PermanentJobError(
                f"Unsupported job type: {job['job_type']}"
            )

        _process_document(job)

        job_repository.complete(job["id"])

        log.info("job %s completed", job["id"])

    except PermanentJobError as exc:

        log.warning(
            "job %s failed permanently: %s",
            job["id"],
            exc
        )

        _fail(job, str(exc), str(exc))

    except Exception as exc:

        log.exception("job %s crashed", job["id"])

        detail = f"{type(exc).__name__}: {exc}"

        if job["attempts"] >= job["max_attempts"]:

            _fail(job, detail, SAFE_FAILURE)

        else:

            document_repository.update(
                job["document_id"],
                {"status": "queued"}
            )

            job_repository.retry_later(
                job["id"],
                detail,
                delay_seconds=30 * job["attempts"]
            )


def recover_stale_jobs() -> None:

    for job in job_repository.find_stale(
        settings.worker_stale_minutes
    ):

        log.warning("recovering stale job %s", job["id"])

        if job["attempts"] >= job["max_attempts"]:

            _fail(
                job,
                "Worker stopped while processing",
                SAFE_FAILURE
            )

        else:

            document_repository.update(
                job["document_id"],
                {"status": "queued"}
            )

            job_repository.retry_later(
                job["id"],
                "Worker stopped while processing",
                delay_seconds=0
            )