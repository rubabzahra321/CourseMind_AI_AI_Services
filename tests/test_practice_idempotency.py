"""Task 3: a retried MCQ generation must never create duplicate questions."""
from tests.conftest import COURSE, DOC, TOPIC, USER, new_id

BASE = f"/internal/courses/{COURSE}/practice"


def body(**kw):
    return {"user_id": USER, "topic_id": TOPIC, "document_id": DOC,
            "number_of_questions": 3, **kw}


def test_retry_with_same_key_replays_the_response_without_new_questions(api, db, world, gemini):
    key = new_id()
    first = api.post(f"{BASE}/generate", json=body(idempotency_key=key))
    second = api.post(f"{BASE}/generate", json=body(idempotency_key=key))

    assert first.status_code == 200 and second.status_code == 200
    assert second.json() == first.json()
    assert len(db.rows("generated_questions")) == 3          # not 6
    assert len(gemini.mcq_calls) == 1                         # Gemini was not called again
    row = db.rows("ai_idempotency_requests")[0]
    assert row["status"] == "completed" and row["response"] == first.json()


def test_without_a_key_each_call_generates_new_questions(api, db, world, gemini):
    api.post(f"{BASE}/generate", json=body())
    api.post(f"{BASE}/generate", json=body())
    assert len(db.rows("generated_questions")) == 6


def test_same_key_with_a_different_request_is_rejected(api, db, world, gemini):
    key = new_id()
    api.post(f"{BASE}/generate", json=body(idempotency_key=key))
    r = api.post(f"{BASE}/generate", json=body(idempotency_key=key, number_of_questions=5))
    assert r.status_code == 400 and "different request" in r.json()["detail"]
    assert len(db.rows("generated_questions")) == 3


def test_request_still_processing_returns_409(api, db, world, gemini):
    key = new_id()
    first = api.post(f"{BASE}/generate", json=body(idempotency_key=key)).json()
    # simulate the original call still running (e.g. the client timed out and retried)
    db.rows("ai_idempotency_requests")[0].update(status="processing", response=None)

    r = api.post(f"{BASE}/generate", json=body(idempotency_key=key))
    assert r.status_code == 409 and "already processing" in r.json()["detail"]
    assert len(db.rows("generated_questions")) == len(first["question_ids"])


def test_two_simultaneous_requests_do_not_crash_with_500(api, db, world, gemini, monkeypatch):
    """The other request inserted the key between our lookup and insert."""
    from app.services import practice_service as ps
    key = new_id()
    real_get = ps._get_idempotency_request
    calls = {"n": 0}

    def get_blind_first_time(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            # the competing request creates the row right after our first (empty) lookup
            ps.get_db().table("ai_idempotency_requests").insert(
                {"user_id": USER, "course_id": COURSE, "idempotency_key": key,
                 "request_hash": "x", "status": "processing"}).execute()
            return None
        return real_get(*a, **kw)

    monkeypatch.setattr(ps, "_get_idempotency_request", get_blind_first_time)
    r = api.post(f"{BASE}/generate", json=body(idempotency_key=key))
    assert r.status_code == 409
    assert db.rows("generated_questions") == []


def test_failed_generation_can_be_retried_and_leaves_no_duplicates(api, db, world, gemini, monkeypatch):
    from app.repositories import question_repository as qr
    key = new_id()
    real_save = qr.save_generated_question
    state = {"n": 0}

    def flaky_save(**kw):
        state["n"] += 1
        if state["n"] == 3:                     # crash while saving the 3rd of 3 questions
            raise RuntimeError("db hiccup")
        return real_save(**kw)

    monkeypatch.setattr(qr, "save_generated_question", flaky_save)
    r1 = api.post(f"{BASE}/generate", json=body(idempotency_key=key))
    assert r1.status_code == 500
    assert db.rows("ai_idempotency_requests")[0]["status"] == "failed"
    assert db.rows("generated_questions") == []             # partial saves were cleaned up

    monkeypatch.setattr(qr, "save_generated_question", real_save)
    r2 = api.post(f"{BASE}/generate", json=body(idempotency_key=key))
    assert r2.status_code == 200 and r2.json()["generated_count"] == 3
    assert len(db.rows("generated_questions")) == 3          # exactly one set
    assert db.rows("ai_idempotency_requests")[0]["status"] == "completed"


def test_idempotency_keys_are_scoped_per_user(api, db, world, gemini):
    from tests.conftest import OTHER_USER
    key = new_id()
    api.post(f"{BASE}/generate", json=body(idempotency_key=key))
    # another user may reuse the same UUID without seeing or replaying the first user's result
    r = api.post(f"{BASE}/generate", json={**body(idempotency_key=key), "user_id": OTHER_USER})
    assert r.status_code == 400 and "different request" not in r.json()["detail"]
