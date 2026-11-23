from datetime import datetime, timedelta, timezone
import pytest
from app.ml import pattern_analysis as pa, registry
from app.services import pattern_analysis_service
from tests.conftest import COURSE, USER, new_id

@pytest.fixture(autouse=True)
def memory_store():
    store = registry.MemoryArtifactStore()
    registry.set_artifact_store(store)
    yield store
    registry.set_artifact_store(registry.SupabaseArtifactStore())


def test_topic_features_detect_repeated_wrong_question():
    now = datetime.now(timezone.utc)
    history = [
        {"is_correct": False, "question_id": "q1", "answered_at": (now-timedelta(days=2)).isoformat(), "response_time_seconds": 40},
        {"is_correct": False, "question_id": "q1", "answered_at": (now-timedelta(days=1)).isoformat(), "response_time_seconds": 42},
        {"is_correct": True, "question_id": "q2", "answered_at": now.isoformat(), "response_time_seconds": 30},
    ]
    row = pa.topic_features(history, now.isoformat())
    assert row["repeated_mistake_rate"] == 0.5
    assert row["attempt_count"] == 3


def test_fit_patterns_selects_k_and_returns_explanation_rules():
    rows = []
    for i in range(12):
        rows.append({
            "accuracy": 0.2 if i < 4 else (0.9 if i < 8 else 0.55),
            "trend_slope": -0.03 if i < 4 else (0.01 if i < 8 else 0.04),
            "average_response_time": 80 if i < 4 else 25,
            "attempt_count": 12 + i,
            "days_since_practice": 2 if i < 8 else 20,
            "repeated_mistake_rate": 0.7 if i < 4 else 0.1,
        })
    bundle, metrics = pa.fit_patterns(rows, 42)
    assert 3 <= metrics["selected_k"] <= 5
    assert -1 <= metrics["silhouette_score"] <= 1
    assert "|---" in metrics["explanation_rules"]
    assert pa.predict_pattern(bundle, rows[0]) in bundle["cluster_names"].values()


def test_fallback_labels_unassessed_and_weak_topics():
    assert pa.fallback_pattern({"attempt_count": 0}) == "not_yet_assessed"
    assert pa.fallback_pattern({"accuracy": .2, "trend_slope": -.02,
        "average_response_time": 60, "attempt_count": 5,
        "days_since_practice": 1, "repeated_mistake_rate": .5}) == "weak_and_not_improving"


def test_training_rejects_too_few_pooled_topics(monkeypatch):
    monkeypatch.setattr(pattern_analysis_service, "_all_topic_rows", lambda: [{"accuracy": .5}] * 7)
    with pytest.raises(ValueError, match="need at least 8"):
        pattern_analysis_service.train_patterns()


def test_learning_patterns_endpoint_requires_internal_key(api, anon_api):
    denied = anon_api.get(f"/internal/courses/{COURSE}/ml/learning-patterns", params={"user_id": USER})
    assert denied.status_code in (401, 403)
