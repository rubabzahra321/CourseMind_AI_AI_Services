from datetime import datetime, timedelta, timezone

import pytest

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

from app.core.config import settings
from app.core.errors import PermanentJobError
from app.repositories import job_repository
from app.services import pdf_service, processing_service
from tests.conftest import COURSE, DOC, USER

PATH = f"{USER}/{COURSE}/{DOC}.pdf"


def make_pdf(pages: list[str], password: str | None = None) -> bytes:
    pdf = fitz.open()
    for text in pages:
        page = pdf.new_page()
        if text:
            page.insert_textbox(fitz.Rect(40, 40, 560, 800), text, fontsize=11)
    kwargs = {}
    if password:
        kwargs = dict(encryption=fitz.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password + "x")
    return pdf.tobytes(**kwargs)


# ---------------------------------------------------------------- pdf_service
def test_extract_pages_numbers_pages_from_one():
    pages = pdf_service.extract_pages(make_pdf(["First page text here", "Second page text here"]))
    assert [n for n, _ in pages] == [1, 2]
    assert "First page" in pages[0][1] and "Second page" in pages[1][1]


def test_extract_pages_rejects_corrupt_file():
    with pytest.raises(PermanentJobError):
        pdf_service.extract_pages(b"this is definitely not a pdf")


def test_extract_pages_rejects_password_protected_pdf():
    with pytest.raises(PermanentJobError, match="password"):
        pdf_service.extract_pages(make_pdf(["secret content here"], password="pw"))


# ---------------------------------------------------------------- helpers
def queue_document(db, pdf_bytes, *, attempts=1, max_attempts=3, job_type="process_document"):
    db.seed("documents", {"id": DOC, "user_id": USER, "course_id": COURSE, "title": "Lecture",
                          "storage_path": PATH, "status": "queued", "document_type": "lecture_material"})
    db.files[PATH] = pdf_bytes
    return db.seed("processing_jobs", {
        "document_id": DOC, "user_id": USER, "course_id": COURSE, "job_type": job_type,
        "status": "processing", "attempts": attempts, "max_attempts": max_attempts,
        "available_at": datetime.now(timezone.utc).isoformat(),
    })


def doc(db):
    return db.rows("documents", id=DOC)[0]


# ---------------------------------------------------------------- happy path
TEXT = "Binary search trees keep keys ordered so lookup is logarithmic. " * 12


def test_run_job_processes_pdf_end_to_end(db, gemini):
    job = queue_document(db, make_pdf([TEXT, TEXT, TEXT]))
    processing_service.run_job(job)

    d = doc(db)
    assert d["status"] == "ready" and d["page_count"] == 3 and d["processed_at"]
    assert d["processing_error"] is None

    chunks = db.rows("document_chunks", document_id=DOC)
    assert len(chunks) >= 3
    assert {c["page_start"] for c in chunks} == {1, 2, 3}
    assert all(c["user_id"] == USER and c["course_id"] == COURSE for c in chunks)
    assert all(c["embedding"].startswith("[") and c["embedding"].count(",") == settings.gemini_embedding_dimensions - 1
               for c in chunks)
    assert [c["chunk_index"] for c in sorted(chunks, key=lambda c: c["chunk_index"])] == list(range(len(chunks)))
    assert db.rows("processing_jobs", id=job["id"])[0]["status"] == "completed"


def test_embeddings_are_requested_in_batches_of_32(db, gemini, monkeypatch):
    queue_document(db, make_pdf([TEXT]))
    long_text = " ".join(f"token{i}" for i in range(20000))  # one huge page -> many chunks
    monkeypatch.setattr(processing_service.pdf_service, "extract_pages", lambda _b: [(1, long_text)])
    processing_service.run_job(db.rows("processing_jobs")[0] if db.rows("processing_jobs") else
                               db.seed("processing_jobs", {"document_id": DOC, "user_id": USER, "course_id": COURSE,
                                                           "job_type": "process_document", "status": "processing",
                                                           "attempts": 1}))
    total_chunks = len(db.rows("document_chunks", document_id=DOC))
    assert total_chunks > 64
    assert len(gemini.doc_embed_batches) == -(-total_chunks // 32)
    assert max(gemini.doc_embed_batches) == 32 and sum(gemini.doc_embed_batches) == total_chunks


def test_retry_replaces_old_chunks_instead_of_duplicating(db, gemini):
    job = queue_document(db, make_pdf([TEXT]))
    db.seed("document_chunks", {"document_id": DOC, "user_id": USER, "course_id": COURSE,
                                "chunk_index": 0, "content": "STALE CHUNK FROM A FAILED RUN"})
    processing_service.run_job(job)
    contents = [c["content"] for c in db.rows("document_chunks", document_id=DOC)]
    assert not any("STALE" in c for c in contents) and len(contents) >= 1


# ---------------------------------------------------------------- permanent failures
def test_scanned_pdf_fails_permanently_with_clear_message(db, gemini):
    job = queue_document(db, make_pdf(["", ""]))  # pages without a text layer
    processing_service.run_job(job)
    assert doc(db)["status"] == "failed" and "scanned" in doc(db)["processing_error"].lower()
    j = db.rows("processing_jobs", id=job["id"])[0]
    assert j["status"] == "failed" and gemini.doc_embed_batches == []


def test_corrupt_pdf_fails_permanently_without_retry(db, gemini):
    job = queue_document(db, b"garbage bytes", attempts=1)
    processing_service.run_job(job)
    assert db.rows("processing_jobs", id=job["id"])[0]["status"] == "failed"
    assert doc(db)["status"] == "failed"


def test_missing_document_fails_job(db, gemini):
    job = db.seed("processing_jobs", {"document_id": DOC, "user_id": USER, "course_id": COURSE,
                                      "job_type": "process_document", "status": "processing", "attempts": 1})
    processing_service.run_job(job)
    assert db.rows("processing_jobs", id=job["id"])[0]["status"] == "failed"


def test_unsupported_job_type_fails(db, gemini):
    job = queue_document(db, make_pdf([TEXT]), job_type="generate_embeddings")
    processing_service.run_job(job)
    j = db.rows("processing_jobs", id=job["id"])[0]
    assert j["status"] == "failed" and "Unsupported job type" in j["last_error"]


# ---------------------------------------------------------------- transient failures
def test_transient_error_requeues_with_backoff(db, gemini):
    gemini.fail_embed_documents = True
    job = queue_document(db, make_pdf([TEXT]), attempts=1)
    processing_service.run_job(job)
    j = db.rows("processing_jobs", id=job["id"])[0]
    assert j["status"] == "queued" and "RuntimeError" in j["last_error"]
    assert j["available_at"] > datetime.now(timezone.utc).isoformat()
    assert j["locked_by"] is None and doc(db)["status"] == "queued"


def test_transient_error_on_last_attempt_fails_with_safe_message(db, gemini):
    gemini.fail_embed_documents = True
    job = queue_document(db, make_pdf([TEXT]), attempts=3, max_attempts=3)
    processing_service.run_job(job)
    assert db.rows("processing_jobs", id=job["id"])[0]["status"] == "failed"
    msg = doc(db)["processing_error"]
    assert msg == processing_service.SAFE_FAILURE and "RuntimeError" not in msg  # no internal leak


# ---------------------------------------------------------------- queue + stale recovery
def test_claim_next_hands_out_each_job_once(db):
    db.seed("processing_jobs", {"document_id": DOC, "user_id": USER, "course_id": COURSE,
                                "job_type": "process_document", "status": "queued"})
    first = job_repository.claim_next("worker-1")
    assert first and first["status"] == "processing" and first["attempts"] == 1
    assert job_repository.claim_next("worker-2") is None


def stale_job(db, attempts):
    old = (datetime.now(timezone.utc) - timedelta(minutes=settings.worker_stale_minutes + 5)).isoformat()
    db.seed("documents", {"id": DOC, "user_id": USER, "course_id": COURSE, "status": "processing"})
    return db.seed("processing_jobs", {"document_id": DOC, "user_id": USER, "course_id": COURSE,
                                       "job_type": "process_document", "status": "processing",
                                       "attempts": attempts, "max_attempts": 3, "locked_at": old})


def test_stale_job_is_requeued(db):
    job = stale_job(db, attempts=1)
    processing_service.recover_stale_jobs()
    assert db.rows("processing_jobs", id=job["id"])[0]["status"] == "queued"
    assert doc(db)["status"] == "queued"


def test_stale_job_out_of_attempts_is_failed(db):
    job = stale_job(db, attempts=3)
    processing_service.recover_stale_jobs()
    assert db.rows("processing_jobs", id=job["id"])[0]["status"] == "failed"
    assert doc(db)["status"] == "failed"


def test_fresh_processing_job_is_left_alone(db):
    db.seed("processing_jobs", {"document_id": DOC, "user_id": USER, "course_id": COURSE,
                                "job_type": "process_document", "status": "processing", "attempts": 1,
                                "locked_at": datetime.now(timezone.utc).isoformat()})
    processing_service.recover_stale_jobs()
    assert db.rows("processing_jobs")[0]["status"] == "processing"
