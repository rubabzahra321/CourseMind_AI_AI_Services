"""
Optional tests against the REAL Supabase project and the REAL Gemini API.
Skipped by default. Run them with (PowerShell):

    $env:RUN_LIVE_TESTS = "1"; pytest -m live -v
"""
import uuid

import pytest

from tests.conftest import LIVE

pytestmark = [pytest.mark.live, pytest.mark.skipif(not LIVE, reason="set RUN_LIVE_TESTS=1 to run live tests")]


@pytest.fixture()
def live_api():
    from fastapi.testclient import TestClient
    from app.core.config import settings
    from app.main import app

    return TestClient(app, headers={"X-Internal-Key": settings.ai_internal_key})


def test_live_health_and_database(live_api):
    assert live_api.get("/health").json() == {"status": "ok"}
    assert live_api.get("/internal/db-check").json() == {"database": "reachable"}


def test_live_embedding_dimension_matches_database_column():
    from app.ai.gemini import embed_documents, embed_query
    from app.core.config import settings

    assert len(embed_query("what is a binary tree")) == settings.gemini_embedding_dimensions == 768
    assert len(embed_documents(["a tree has nodes", "graphs have edges"])) == 2


def test_live_chat_model_answers():
    from app.ai.gemini import generate_answer

    assert generate_answer("Reply with exactly the word: OK")


def test_live_vector_search_function_exists():
    from app.core.config import settings
    from app.repositories import rag_repository

    rows = rag_repository.search_chunks(str(uuid.uuid4()), str(uuid.uuid4()),
                                        [0.0] * settings.gemini_embedding_dimensions, 3, 0.0)
    assert rows == []


def test_live_ai_views_are_queryable():
    from app.db.client import get_db

    for view in ("topic_ai_stats", "topic_frequency", "topic_performance"):
        get_db().table(view).select("*").limit(1).execute()
