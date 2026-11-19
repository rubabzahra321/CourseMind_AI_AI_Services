import pytest

from app.core.config import settings
from tests.conftest import (COURSE, DOC, OTHER_COURSE, OTHER_USER, TOPIC, USER, make_mcqs, new_id, seed_question)

BASE = f"/internal/courses/{COURSE}/practice"


# ====================================================================== generate
def gen_body(**kw):
    return {"user_id": USER, "topic_id": TOPIC, "document_id": DOC, **kw}


def test_generate_saves_questions_with_four_options_and_one_answer(api, db, world, gemini):
    r = api.post(f"{BASE}/generate", json=gen_body(number_of_questions=3, difficulty="hard"))
    assert r.status_code == 200
    body = r.json()
    assert body["generated_count"] == 3 and len(body["question_ids"]) == 3

    qs = db.rows("generated_questions")
    assert len(qs) == 3 and all(q["difficulty"] == "hard" and q["topic_id"] == TOPIC for q in qs)
    assert all(q["user_id"] == USER and q["course_id"] == COURSE and q["source_document_id"] == DOC for q in qs)
    for q in qs:
        opts = db.rows("question_options", question_id=q["id"])
        assert sorted(o["option_order"] for o in opts) == [1, 2, 3, 4]
        assert sum(o["is_correct"] for o in opts) == 1


def test_generate_passes_topic_difficulty_and_page_labelled_material(api, db, world, gemini):
    api.post(f"{BASE}/generate", json=gen_body(number_of_questions=2, difficulty="easy"))
    kw = gemini.mcq_calls[0]
    assert kw["topic_title"] == "Binary Trees" and kw["difficulty"] == "easy" and kw["number_of_questions"] == 2
    assert "Page 1:" in kw["material"] and "Chunk 0 about binary trees." in kw["material"]


def test_generate_truncates_material_to_configured_limit(api, db, world, gemini):
    db.seed("document_chunks", {"document_id": DOC, "course_id": COURSE, "user_id": USER, "chunk_index": 99,
                                "content": "z" * (settings.mcq_context_max_chars * 2), "page_start": 9, "page_end": 9})
    api.post(f"{BASE}/generate", json=gen_body())
    assert len(gemini.mcq_calls[0]["material"]) <= settings.mcq_context_max_chars


@pytest.mark.parametrize("override", [{"number_of_questions": 0}, {"number_of_questions": 21},
                                      {"difficulty": "impossible"}, {"topic_id": "not-a-uuid"}])
def test_generate_validates_input(api, world, gemini, override):
    assert api.post(f"{BASE}/generate", json=gen_body(**override)).status_code == 422


def test_generate_rejects_unknown_topic(api, world, gemini):
    r = api.post(f"{BASE}/generate", json=gen_body(topic_id=new_id()))
    assert r.status_code == 400 and "Topic" in r.json()["detail"]


def test_generate_rejects_other_users_topic(api, db, world, gemini):
    t = db.seed("syllabus_topics", {"course_id": OTHER_COURSE, "user_id": OTHER_USER, "title": "Theirs"})
    assert api.post(f"{BASE}/generate", json=gen_body(topic_id=t["id"])).status_code == 400


def test_generate_requires_a_ready_document(api, db, world, gemini):
    db.tables["documents"][0]["status"] = "processing"
    r = api.post(f"{BASE}/generate", json=gen_body())
    assert r.status_code == 400 and "document" in r.json()["detail"].lower()


def test_generate_rejects_document_without_text(api, db, world, gemini):
    db.tables["document_chunks"].clear()
    assert api.post(f"{BASE}/generate", json=gen_body()).status_code == 400


def test_generate_failure_is_reported_as_generic_500(api, world, gemini, monkeypatch):
    def boom(**_):
        raise RuntimeError("secret-internal-detail")
    monkeypatch.setattr("app.services.practice_service.generate_mcqs", boom)
    r = api.post(f"{BASE}/generate", json=gen_body())
    assert r.status_code == 500 and "secret-internal-detail" not in r.text


# ====================================================================== list questions
def test_list_questions_never_exposes_the_answer_key(api, db, world):
    seed_question(db)
    r = api.post(f"{BASE}/questions", json={"user_id": USER})
    assert r.status_code == 200 and len(r.json()) == 1
    q = r.json()[0]
    assert len(q["options"]) == 4
    assert all("is_correct" not in o for o in q["options"])


def test_list_questions_filters_by_topic_user_and_limit(api, db, world):
    other_topic = db.seed("syllabus_topics", {"course_id": COURSE, "user_id": USER, "title": "Graphs"})["id"]
    for _ in range(3):
        seed_question(db)
    seed_question(db, topic=other_topic)
    seed_question(db, user=OTHER_USER, course=OTHER_COURSE)
    assert len(api.post(f"{BASE}/questions", json={"user_id": USER}).json()) == 4
    assert len(api.post(f"{BASE}/questions", json={"user_id": USER, "topic_id": TOPIC}).json()) == 3
    assert len(api.post(f"{BASE}/questions", json={"user_id": USER, "limit": 2}).json()) == 2
    assert api.post(f"{BASE}/questions", json={"user_id": USER, "limit": 101}).status_code == 422


# ====================================================================== sessions
def start(api, ids, topic=TOPIC):
    return api.post(f"{BASE}/sessions", json={"user_id": USER, "topic_id": topic, "question_ids": ids})


def test_start_session_returns_ordered_questions_without_answers(api, db, world):
    q1, q2 = seed_question(db)[0], seed_question(db)[0]
    r = start(api, [q1, q2])
    assert r.status_code == 200
    body = r.json()
    assert body["total_questions"] == 2 and [q["id"] for q in body["questions"]] == [q1, q2]
    assert all("is_correct" not in o for q in body["questions"] for o in q["options"])
    sess = db.rows("practice_sessions")[0]
    assert sess["user_id"] == USER and sess["total_questions"] == 2 and sess["status"] == "in_progress"
    links = sorted(db.rows("practice_session_questions", session_id=body["session_id"]), key=lambda r: r["question_order"])
    assert [l["question_id"] for l in links] == [q1, q2] and [l["question_order"] for l in links] == [1, 2]


def test_start_session_rejects_duplicates_empty_and_oversized(api, db, world):
    q = seed_question(db)[0]
    assert start(api, [q, q]).status_code == 400
    assert start(api, []).status_code == 422
    assert start(api, [new_id() for _ in range(51)]).status_code == 422


def test_start_session_rejects_unknown_or_foreign_questions(api, db, world):
    foreign = seed_question(db, user=OTHER_USER, course=OTHER_COURSE)[0]
    assert start(api, [new_id()]).status_code == 400
    assert start(api, [foreign]).status_code == 400
    assert db.rows("practice_sessions") == []


def test_start_session_rejects_question_from_another_topic(api, db, world):
    other_topic = db.seed("syllabus_topics", {"course_id": COURSE, "user_id": USER, "title": "Graphs"})["id"]
    q = seed_question(db, topic=other_topic)[0]
    assert start(api, [q]).status_code == 400


# ====================================================================== submit answers
def attempt(api, session_id, question_id, option_id, **extra):
    return api.post(f"{BASE}/sessions/{session_id}/attempts",
                    json={"user_id": USER, "question_id": question_id, "selected_option_id": option_id, **extra})


def open_session(api, db, n=2, correct=2):
    qs = [seed_question(db, correct=correct, text=f"Q{i}") for i in range(n)]
    sid = start(api, [q[0] for q in qs]).json()["session_id"]
    return sid, qs


def test_correct_answer_is_scored_and_progress_is_refreshed(api, db, world):
    sid, qs = open_session(api, db)
    qid, opts, correct = qs[0]
    r = attempt(api, sid, qid, correct, response_time_seconds=12)
    assert r.status_code == 200
    body = r.json()
    assert body["is_correct"] is True and body["correct_option"]["id"] == correct
    assert body["explanation"] and body["progress"] == {"answered": 1, "total": 2, "correct": 1,
                                                       "accuracy_percentage": 100.0, "completed": False}
    saved = db.rows("practice_attempts")[0]
    assert saved["is_correct"] is True and saved["response_time_seconds"] == 12 and saved["user_id"] == USER
    assert ("refresh_topic_progress", {"p_user_id": USER, "p_course_id": COURSE, "p_topic_id": TOPIC}) in db.rpc_calls


def test_wrong_answer_reveals_the_actually_correct_option(api, db, world):
    sid, qs = open_session(api, db, correct=2)
    qid, opts, correct = qs[0]
    wrong = opts[0]
    body = attempt(api, sid, qid, wrong).json()
    assert body["is_correct"] is False
    assert body["correct_option"]["id"] == correct, "must show the right answer, not the student's wrong pick"
    assert body["correct_option"]["option_text"] == "Option 2"


def test_session_completes_after_last_answer_with_accuracy(api, db, world):
    sid, qs = open_session(api, db, n=2)
    attempt(api, sid, qs[0][0], qs[0][2])            # correct
    last = attempt(api, sid, qs[1][0], qs[1][1][0])  # wrong
    assert last.json()["progress"]["completed"] is True and last.json()["progress"]["accuracy_percentage"] == 50.0
    sess = db.rows("practice_sessions")[0]
    assert sess["status"] == "completed" and sess["correct_answers"] == 1 and sess["accuracy_percentage"] == 50.0
    assert sess["completed_at"]


def test_cannot_answer_the_same_question_twice(api, db, world):
    sid, qs = open_session(api, db)
    assert attempt(api, sid, qs[0][0], qs[0][2]).status_code == 200
    assert attempt(api, sid, qs[0][0], qs[0][2]).status_code == 400
    assert len(db.rows("practice_attempts")) == 1


def test_cannot_answer_a_question_that_is_not_in_the_session(api, db, world):
    sid, _ = open_session(api, db)
    stray = seed_question(db)
    assert attempt(api, sid, stray[0], stray[2]).status_code == 400


def test_option_from_another_question_is_rejected(api, db, world):
    sid, qs = open_session(api, db)
    assert attempt(api, sid, qs[0][0], qs[1][2]).status_code == 400
    assert db.rows("practice_attempts") == []


def test_unknown_or_foreign_session_is_rejected(api, db, world):
    q = seed_question(db)
    assert attempt(api, new_id(), q[0], q[2]).status_code == 400
    foreign = db.seed("practice_sessions", {"user_id": OTHER_USER, "course_id": OTHER_COURSE, "total_questions": 1})
    assert attempt(api, foreign["id"], q[0], q[2]).status_code == 400


def test_completed_session_accepts_no_more_answers(api, db, world):
    sid, qs = open_session(api, db, n=1)
    attempt(api, sid, qs[0][0], qs[0][2])
    again = attempt(api, sid, qs[0][0], qs[0][2])
    assert again.status_code == 400


def test_negative_response_time_is_rejected(api, db, world):
    sid, qs = open_session(api, db)
    assert attempt(api, sid, qs[0][0], qs[0][2], response_time_seconds=-1).status_code == 422
