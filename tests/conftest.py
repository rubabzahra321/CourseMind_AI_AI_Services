import os
import uuid
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

LIVE = os.environ.get("RUN_LIVE_TESTS") == "1"

# Unit tests must NEVER touch a real project: force dummy credentials so that a
# developer's real .env is ignored (environment variables beat .env files).
if not LIVE:
    os.environ["SUPABASE_URL"] = "https://test-project.supabase.co"
    os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test-service-role-key"
    os.environ["AI_INTERNAL_KEY"] = "test-internal-key-0123456789"
    os.environ["GEMINI_API_KEY"] = "test-gemini-key"

from fastapi.testclient import TestClient  # noqa: E402

from tests.fakes import FakeDB  # noqa: E402

INTERNAL_KEY = "test-internal-key-0123456789"

USER = "11111111-1111-4111-8111-111111111111"
OTHER_USER = "22222222-2222-4222-8222-222222222222"
COURSE = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
OTHER_COURSE = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
TOPIC = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
DOC = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"


def new_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture(autouse=True)
def db(monkeypatch):
    """Replace the Supabase client with an in-memory fake (skipped for live tests)."""
    if LIVE:
        yield None
        return
    import app.db.client as client

    fake = FakeDB()
    client.get_db.cache_clear()
    monkeypatch.setattr(client, "create_client", lambda *a, **k: fake)
    yield fake
    client.get_db.cache_clear()


@pytest.fixture()
def api():
    from app.main import app

    # raise_server_exceptions=False -> an unhandled crash is seen as HTTP 500,
    # exactly what the Node backend would receive.
    return TestClient(app, raise_server_exceptions=False, headers={"X-Internal-Key": INTERNAL_KEY})


@pytest.fixture()
def anon_api():
    from app.main import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def world(db):
    """A small, valid dataset: 1 course, 1 topic, 1 ready document with 3 chunks."""
    exam = (date.today() + timedelta(days=30)).isoformat()
    db.seed("courses", {"id": COURSE, "user_id": USER, "title": "Data Structures", "exam_date": exam})
    db.seed("courses", {"id": OTHER_COURSE, "user_id": OTHER_USER, "title": "Someone else", "exam_date": exam})
    db.seed("syllabus_topics", {"id": TOPIC, "course_id": COURSE, "user_id": USER,
                                "title": "Binary Trees", "parent_topic_id": None, "position": 1})
    db.seed("documents", {"id": DOC, "course_id": COURSE, "user_id": USER, "title": "Lecture 1",
                          "status": "ready", "storage_path": f"{USER}/{COURSE}/{DOC}.pdf",
                          "document_type": "lecture_material"})
    for i in range(3):
        db.seed("document_chunks", {"document_id": DOC, "course_id": COURSE, "user_id": USER,
                                    "chunk_index": i, "content": f"Chunk {i} about binary trees.",
                                    "page_start": i + 1, "page_end": i + 1})
    return SimpleNamespace(user=USER, other_user=OTHER_USER, course=COURSE, other_course=OTHER_COURSE,
                           topic=TOPIC, doc=DOC)


def seed_question(db, *, user=USER, course=COURSE, topic=TOPIC, correct=2, text="What is a BST?"):
    """Insert a generated MCQ directly. Returns (question_id, [option_ids], correct_option_id)."""
    q = db.seed("generated_questions", {
        "user_id": user, "course_id": course, "topic_id": topic,
        "question_text": text, "explanation": "Because of the BST ordering property.",
        "difficulty": "medium",
    })
    option_ids = []
    for order in range(1, 5):
        opt = db.seed("question_options", {"question_id": q["id"], "option_text": f"Option {order}",
                      "option_order": order, "is_correct": order == correct})
        option_ids.append(opt["id"])
    return q["id"], option_ids, option_ids[correct - 1]


def make_mcqs(n: int) -> list[dict]:
    return [{
        "question_text": f"Question {i}?",
        "explanation": f"Explanation {i}",
        "options": [{"option_text": f"Q{i} option {k}", "is_correct": k == 1} for k in range(4)],
    } for i in range(n)]


@pytest.fixture()
def gemini(monkeypatch):
    """Replaces every Gemini call. Tweak the attributes inside a test to change behaviour."""
    from app.core.config import settings

    g = SimpleNamespace(
        embed_query_calls=[], answer_prompts=[], doc_embed_batches=[], mcq_calls=[],
        answer="Binary trees store nodes hierarchically [1].",
        fail_answer=False, fail_embed_documents=False, mcqs=None,
    )

    def embed_query(question):
        g.embed_query_calls.append(question)
        return [0.1] * settings.gemini_embedding_dimensions

    def generate_answer(prompt):
        g.answer_prompts.append(prompt)
        if g.fail_answer:
            raise RuntimeError("Gemini quota exceeded: secret-internal-detail")
        return g.answer

    def embed_documents(texts):
        g.doc_embed_batches.append(len(texts))
        if g.fail_embed_documents:
            raise RuntimeError("embedding service down")
        return [[0.0] * settings.gemini_embedding_dimensions for _ in texts]

    def generate_mcqs(**kwargs):
        g.mcq_calls.append(kwargs)
        return g.mcqs if g.mcqs is not None else make_mcqs(kwargs["number_of_questions"])

    monkeypatch.setattr("app.services.rag_service.embed_query", embed_query)
    monkeypatch.setattr("app.services.rag_service.generate_answer", generate_answer)
    monkeypatch.setattr("app.services.processing_service.embed_documents", embed_documents)
    monkeypatch.setattr("app.services.practice_service.generate_mcqs", generate_mcqs)
    return g
