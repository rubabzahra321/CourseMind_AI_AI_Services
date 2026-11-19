import json
from types import SimpleNamespace

import pytest

from app.core.config import settings
from tests.conftest import make_mcqs

DIM = settings.gemini_embedding_dimensions


class FakeModels:
    def __init__(self, embed_vectors=None, text=None):
        self.embed_vectors, self.text, self.embed_kwargs, self.gen_kwargs = embed_vectors, text, [], []

    def embed_content(self, **kw):
        self.embed_kwargs.append(kw)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=v) for v in self.embed_vectors])

    def generate_content(self, **kw):
        self.gen_kwargs.append(kw)
        return SimpleNamespace(text=self.text)


def use_client(monkeypatch, models):
    monkeypatch.setattr("app.ai.gemini.get_gemini_client", lambda: SimpleNamespace(models=models))
    return models


# ---------------------------------------------------------------- embeddings
def test_embed_documents_uses_document_task_and_configured_dimensions(monkeypatch):
    from app.ai.gemini import embed_documents

    m = use_client(monkeypatch, FakeModels([[0.0] * DIM, [0.0] * DIM]))
    assert len(embed_documents(["a", "b"])) == 2
    cfg = m.embed_kwargs[0]["config"]
    assert cfg.task_type == "RETRIEVAL_DOCUMENT" and cfg.output_dimensionality == DIM
    assert m.embed_kwargs[0]["model"] == settings.gemini_embedding_model


def test_embed_query_uses_query_task(monkeypatch):
    from app.ai.gemini import embed_query

    m = use_client(monkeypatch, FakeModels([[0.0] * DIM]))
    assert len(embed_query("what is a tree?")) == DIM
    assert m.embed_kwargs[0]["config"].task_type == "RETRIEVAL_QUERY"


def test_embed_documents_empty_input_makes_no_api_call(monkeypatch):
    from app.ai.gemini import embed_documents

    m = use_client(monkeypatch, FakeModels([]))
    assert embed_documents([]) == [] and m.embed_kwargs == []


def test_embed_documents_rejects_wrong_dimension(monkeypatch):
    """The DB column is vector(768): a mismatch must fail loudly, not corrupt data."""
    from app.ai.gemini import embed_documents

    use_client(monkeypatch, FakeModels([[0.0] * (DIM + 1)]))
    with pytest.raises(RuntimeError, match="dimension"):
        embed_documents(["a"])


def test_embed_documents_rejects_wrong_count(monkeypatch):
    from app.ai.gemini import embed_documents

    use_client(monkeypatch, FakeModels([[0.0] * DIM]))
    with pytest.raises(RuntimeError, match="unexpected number"):
        embed_documents(["a", "b"])


def test_embed_query_rejects_wrong_dimension(monkeypatch):
    from app.ai.gemini import embed_query

    use_client(monkeypatch, FakeModels([[0.0] * 10]))
    with pytest.raises(RuntimeError):
        embed_query("q")


def test_generate_answer_strips_and_rejects_empty(monkeypatch):
    from app.ai.gemini import generate_answer

    use_client(monkeypatch, FakeModels(text="  hello  "))
    assert generate_answer("p") == "hello"
    use_client(monkeypatch, FakeModels(text=""))
    with pytest.raises(RuntimeError, match="empty"):
        generate_answer("p")


def test_default_embedding_dimension_matches_schema_vector_768():
    assert settings.gemini_embedding_dimensions == 768


# ---------------------------------------------------------------- MCQ generator
def run_generator(monkeypatch, payload, n=2, raw=None):
    from app.ai import question_generator as qg

    text = raw if raw is not None else json.dumps(payload)
    fake = SimpleNamespace(models=FakeModels(text=text))
    monkeypatch.setattr(qg, "client", fake)
    return qg.generate_mcqs("Trees", "medium", n, "material"), fake.models


def test_mcq_generator_accepts_valid_output(monkeypatch):
    out, models = run_generator(monkeypatch, {"questions": make_mcqs(3)}, n=3)
    assert len(out) == 3
    kw = models.gen_kwargs[0]
    assert kw["config"].response_mime_type == "application/json"
    assert "Topic: Trees" in kw["contents"] and "exactly 3" in kw["contents"]


def test_mcq_generator_rejects_wrong_question_count(monkeypatch):
    with pytest.raises(ValueError, match="unexpected number"):
        run_generator(monkeypatch, {"questions": make_mcqs(2)}, n=3)


def mutate(fn):
    qs = make_mcqs(1)
    fn(qs[0])
    return {"questions": qs}


@pytest.mark.parametrize("bad, message", [
    (lambda q: q.update(question_text=""), "Question text"),
    (lambda q: q.update(explanation=""), "explanation"),
    (lambda q: q["options"].pop(), "four options"),
    (lambda q: q["options"][0].update(is_correct=True), "exactly one correct"),   # two correct
    (lambda q: q["options"][1].update(is_correct=False), "exactly one correct"),  # none correct
    (lambda q: q["options"][2].update(option_text=""), "empty"),
])
def test_mcq_generator_rejects_malformed_questions(monkeypatch, bad, message):
    with pytest.raises(ValueError, match=message):
        run_generator(monkeypatch, mutate(bad), n=1)


def test_mcq_generator_rejects_empty_and_invalid_json(monkeypatch):
    with pytest.raises(RuntimeError, match="empty"):
        run_generator(monkeypatch, None, raw="")
    with pytest.raises(json.JSONDecodeError):
        run_generator(monkeypatch, None, raw="```json not json")
