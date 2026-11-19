import pytest

from tests.conftest import COURSE, USER


def test_public_health_needs_no_key(anon_api):
    r = anon_api.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [{}, {"X-Internal-Key": "wrong"}, {"X-Internal-Key": ""}])
def test_internal_routes_reject_missing_or_wrong_key(anon_api, headers):
    r = anon_api.get("/internal/db-check", headers=headers)
    assert r.status_code == 401


def test_internal_route_accepts_correct_key(api, db):
    db.seed("courses", {"user_id": USER, "title": "x"})
    r = api.get("/internal/db-check")
    assert r.status_code == 200 and r.json() == {"database": "reachable"}


def test_every_internal_router_is_protected(anon_api):
    """No internal endpoint may answer without the key (guards against a forgotten dependency)."""
    from app.main import app

    unprotected = []
    for route in app.routes:
        path = getattr(route, "path", "")
        if not path.startswith("/internal"):
            continue
        for method in route.methods - {"HEAD", "OPTIONS"}:
            url = path.replace("{course_id}", COURSE).replace("{topic_id}", COURSE) \
                      .replace("{session_id}", COURSE).replace("{game_id}", COURSE)
            r = anon_api.request(method, url, json={})
            if r.status_code != 401:
                unprotected.append((method, path, r.status_code))
    assert unprotected == []


def test_blank_internal_key_must_not_open_the_service(anon_api, monkeypatch):
    """If AI_INTERNAL_KEY is accidentally left blank, requests with NO header must still be refused."""
    monkeypatch.setattr("app.core.security.settings.ai_internal_key", "")
    r = anon_api.get("/internal/db-check")
    assert r.status_code == 401
