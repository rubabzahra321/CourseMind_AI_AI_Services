from app.core.config import settings
from tests.conftest import COURSE, OTHER_COURSE, OTHER_USER, USER, new_id

URL = f"/internal/courses/{COURSE}/chat"


def match(content="Binary trees store data hierarchically.", page=2, sim=0.82, doc_id=None):
    return {"chunk_id": new_id(), "document_id": doc_id or new_id(), "document_title": "Lecture 1",
            "content": content, "page_start": page, "page_end": page, "similarity": sim}


def set_matches(db, matches):
    db.rpc_handlers["match_document_chunks"] = lambda params: matches


# ------------------------------------------------------------------ validation / auth
def test_chat_requires_internal_key(anon_api):
    assert anon_api.post(URL, json={"user_id": USER, "question": "hello there"}).status_code == 401


def test_chat_validates_body(api):
    assert api.post(URL, json={"user_id": USER, "question": "hi"}).status_code == 422          # too short
    assert api.post(URL, json={"user_id": USER, "question": "x" * 4001}).status_code == 422   # too long
    assert api.post(URL, json={"question": "valid question"}).status_code == 422               # no user
    assert api.post(URL, json={"user_id": "nope", "question": "valid question"}).status_code == 422


# ------------------------------------------------------------------ happy path
def test_chat_returns_grounded_answer_with_sources(api, db, gemini):
    set_matches(db, [match(page=2), match(content="Second passage", page=5, sim=0.6)])
    r = api.post(URL, json={"user_id": USER, "question": "What is a binary tree?"})
    assert r.status_code == 200
    body = r.json()
    assert body["insufficient_evidence"] is False and body["answer"] == gemini.answer
    assert [s["rank"] for s in body["sources"]] == [1, 2]
    assert body["sources"][0]["page_start"] == 2 and body["sources"][0]["similarity"] == 0.82
    assert all(len(s["excerpt"]) <= 500 for s in body["sources"])


def test_chat_persists_conversation_and_sources(api, db, gemini):
    set_matches(db, [match(), match()])
    body = api.post(URL, json={"user_id": USER, "question": "What is a binary tree?"}).json()

    sessions = db.rows("chat_sessions")
    assert len(sessions) == 1 and sessions[0]["user_id"] == USER and sessions[0]["course_id"] == COURSE
    assert sessions[0]["title"] == "What is a binary tree?"
    msgs = sorted(db.rows("chat_messages"), key=lambda m: m["created_at"])
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["id"] == body["message_id"] and msgs[1]["model_name"] == settings.gemini_chat_model
    assert len(db.rows("chat_message_sources", message_id=body["message_id"])) == 2


def test_long_question_title_is_truncated_to_60_chars(api, db, gemini):
    set_matches(db, [match()])
    api.post(URL, json={"user_id": USER, "question": "q" * 200})
    assert len(db.rows("chat_sessions")[0]["title"]) == 60


# ------------------------------------------------------------------ RAG behaviour
def test_no_evidence_does_not_call_gemini_and_flags_insufficient(api, db, gemini):
    set_matches(db, [])
    r = api.post(URL, json={"user_id": USER, "question": "Who won the 2018 World Cup?"})
    body = r.json()
    assert r.status_code == 200 and body["insufficient_evidence"] is True and body["sources"] == []
    assert gemini.answer_prompts == []                       # the model was never asked to guess
    saved = db.rows("chat_messages", role="assistant")[0]
    assert saved["insufficient_evidence"] is True
    assert db.rows("chat_message_sources") == []


def test_retrieval_is_scoped_to_user_and_course_with_configured_params(api, db, gemini):
    set_matches(db, [match()])
    api.post(URL, json={"user_id": USER, "question": "What is a binary tree?"})
    name, params = [c for c in db.rpc_calls if c[0] == "match_document_chunks"][0]
    assert params["p_user_id"] == USER and params["p_course_id"] == COURSE
    assert params["p_match_count"] == settings.rag_top_k
    assert params["p_min_similarity"] == settings.rag_min_similarity
    vec = params["p_query_embedding"]
    assert vec.startswith("[") and vec.endswith("]") and vec.count(",") == settings.gemini_embedding_dimensions - 1


def test_prompt_contains_numbered_sources_and_grounding_rules(api, db, gemini):
    set_matches(db, [match("ALPHA passage"), match("BETA passage")])
    api.post(URL, json={"user_id": USER, "question": "Explain trees please"})
    prompt = gemini.answer_prompts[0]
    assert "SOURCE [1]" in prompt and "SOURCE [2]" in prompt
    assert "ALPHA passage" in prompt and "BETA passage" in prompt
    assert "ONLY the provided" in prompt


# ------------------------------------------------------------------ sessions & memory
def test_follow_up_reuses_session_and_sends_history(api, db, gemini):
    set_matches(db, [match()])
    first = api.post(URL, json={"user_id": USER, "question": "First question here"}).json()
    api.post(URL, json={"user_id": USER, "question": "Second question here", "session_id": first["session_id"]})

    assert len(db.rows("chat_sessions")) == 1 and len(db.rows("chat_messages")) == 4
    prompt = gemini.answer_prompts[1]
    assert "user: First question here" in prompt and f"assistant: {gemini.answer}" in prompt
    assert "user: Second question here" not in prompt       # current question is not duplicated into history


def test_unknown_session_returns_404(api, db, gemini):
    r = api.post(URL, json={"user_id": USER, "question": "Hello there", "session_id": new_id()})
    assert r.status_code == 404


def test_cannot_use_another_users_session(api, db, gemini):
    other = db.seed("chat_sessions", {"user_id": OTHER_USER, "course_id": OTHER_COURSE, "title": "secret"})
    r = api.post(URL, json={"user_id": USER, "question": "Hello there", "session_id": other["id"]})
    assert r.status_code == 404
    assert db.rows("chat_messages") == []


def test_session_from_a_different_course_is_rejected(api, db, gemini):
    s = db.seed("chat_sessions", {"user_id": USER, "course_id": OTHER_COURSE, "title": "other course"})
    assert api.post(URL, json={"user_id": USER, "question": "Hello there", "session_id": s["id"]}).status_code == 404


# ------------------------------------------------------------------ failure handling
def test_gemini_failure_returns_generic_500_without_leaking_details(api, db, gemini):
    set_matches(db, [match()])
    gemini.fail_answer = True
    r = api.post(URL, json={"user_id": USER, "question": "What is a binary tree?"})
    assert r.status_code == 500
    assert r.json()["detail"] == "AI chat processing failed" and "secret-internal-detail" not in r.text
