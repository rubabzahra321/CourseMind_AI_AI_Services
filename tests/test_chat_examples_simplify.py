"""F-05: simplification and clearly labelled, model-added examples."""
from app.services import rag_service
from app.services.rag_service import (
    AI_EXAMPLE_LABEL, _label_ai_examples, _without_ai_examples,
)
from tests.conftest import COURSE, USER
from tests.test_chat_api import match, set_matches

URL = f"/internal/courses/{COURSE}/chat"


def ask(api, **extra):
    return api.post(URL, json={"user_id": USER, "question": "What is a binary tree?", **extra})


# ------------------------------------------------------------------ label helper
def test_answers_without_markers_are_returned_unchanged():
    text = "A tree has nodes [1].  \n\nSecond paragraph.\n\n\n"
    assert _label_ai_examples(text) == (text, [])


def test_marked_example_is_labelled_and_extracted():
    raw = "A tree has nodes [1]. [[AI-EXAMPLE]]A family tree is a tree.[[/AI-EXAMPLE]] Roots are on top [2]."
    labelled, examples = _label_ai_examples(raw)

    assert examples == ["A family tree is a tree."]
    assert AI_EXAMPLE_LABEL in labelled
    assert "> A family tree is a tree." in labelled
    assert "[[" not in labelled
    assert "A tree has nodes [1]." in labelled and "Roots are on top [2]." in labelled


def test_multiline_and_multiple_examples_are_all_labelled():
    raw = ("Intro [1].\n[[AI-EXAMPLE]]Line one\nLine two[[/AI-EXAMPLE]]\n"
           "More [2].\n[[AI-EXAMPLE]]Another[[/AI-EXAMPLE]]")
    labelled, examples = _label_ai_examples(raw)

    assert examples == ["Line one\nLine two", "Another"]
    assert labelled.count(AI_EXAMPLE_LABEL) == 2
    assert "> Line one\n> Line two" in labelled


def test_unclosed_marker_labels_the_rest_of_the_answer():
    labelled, examples = _label_ai_examples("Fact [1]. [[AI-EXAMPLE]]Imagine a folder tree")
    assert examples == ["Imagine a folder tree"]
    assert AI_EXAMPLE_LABEL in labelled


def test_empty_example_and_stray_markers_are_removed():
    labelled, examples = _label_ai_examples("Fact [1]. [[AI-EXAMPLE]]  [[/AI-EXAMPLE]] Done [[/AI-EXAMPLE]]")
    assert examples == []
    assert "[[" not in labelled and "AI-EXAMPLE" not in labelled


def test_without_ai_examples_removes_only_the_example():
    raw = "Fact [1]. [[AI-EXAMPLE]]Example with [2] inside[[/AI-EXAMPLE]] End."
    assert _without_ai_examples(raw) == "Fact [1].  End."


# ------------------------------------------------------------------ API
def test_defaults_are_off_and_response_says_so(api, db, gemini):
    set_matches(db, [match()])
    body = ask(api).json()

    assert body["simplified"] is False and body["include_examples"] is False
    assert body["contains_ai_examples"] is False and body["ai_examples"] == []
    assert body["answer"] == gemini.answer                       # untouched
    assert "Do not add examples" in gemini.answer_prompts[0]
    assert "SIMPLE LANGUAGE MODE" not in gemini.answer_prompts[0]


def test_simplify_adds_plain_language_instructions(api, db, gemini):
    set_matches(db, [match()])
    body = ask(api, simplify=True).json()

    prompt = gemini.answer_prompts[0]
    assert body["simplified"] is True
    assert "SIMPLE LANGUAGE MODE" in prompt
    assert "must not add facts that\nare missing from the sources" in prompt


def test_include_examples_asks_gemini_to_mark_its_own_examples(api, db, gemini):
    set_matches(db, [match()])
    ask(api, include_examples=True)

    prompt = gemini.answer_prompts[0]
    assert "[[AI-EXAMPLE]]" in prompt and "MAY write ONE short example" in prompt
    assert "Do not add examples" not in prompt


def test_model_written_example_is_labelled_in_response_and_history(api, db, gemini):
    set_matches(db, [match()])
    gemini.answer = "A tree has nodes [1]. [[AI-EXAMPLE]]Your folders form a tree.[[/AI-EXAMPLE]]"

    body = ask(api, include_examples=True).json()

    assert body["contains_ai_examples"] is True
    assert body["ai_examples"] == ["Your folders form a tree."]
    assert AI_EXAMPLE_LABEL in body["answer"]
    assert "[[" not in body["answer"]

    # the label is stored, so it is still there when the chat is reopened
    saved = [m for m in db.rows("chat_messages") if m["role"] == "assistant"][0]
    assert AI_EXAMPLE_LABEL in saved["content"] and "[[" not in saved["content"]


def test_an_unrequested_example_is_still_labelled(api, db, gemini):
    set_matches(db, [match()])
    gemini.answer = "Fact [1]. [[AI-EXAMPLE]]Sneaky example[[/AI-EXAMPLE]]"

    body = ask(api).json()                      # include_examples not requested

    assert body["contains_ai_examples"] is True and AI_EXAMPLE_LABEL in body["answer"]


def test_ai_examples_never_count_as_evidence(api, db, gemini):
    """A citation inside an AI example must not make the answer look 'strong'."""
    set_matches(db, [match(sim=0.9)])
    gemini.answer = "Trees store nodes. [[AI-EXAMPLE]]Think of a family tree [1].[[/AI-EXAMPLE]]"

    body = ask(api, include_examples=True).json()

    assert body["evidence_strength"] == "partial"        # no real citation outside the example


def test_cited_answer_with_example_keeps_strong_evidence(api, db, gemini):
    set_matches(db, [match(sim=0.9)])
    gemini.answer = "Trees store nodes [1]. [[AI-EXAMPLE]]A family tree.[[/AI-EXAMPLE]]"

    assert ask(api, include_examples=True).json()["evidence_strength"] == "strong"


def test_no_material_means_no_examples_and_no_gemini_call(api, db, gemini):
    set_matches(db, [])
    body = ask(api, include_examples=True, simplify=True).json()

    assert body["insufficient_evidence"] is True
    assert body["contains_ai_examples"] is False and body["ai_examples"] == []
    assert body["simplified"] is True and body["include_examples"] is True
    assert gemini.answer_prompts == []


def test_flags_must_be_booleans(api):
    assert ask(api, simplify="maybe").status_code == 422
    assert ask(api, include_examples=[1]).status_code == 422
