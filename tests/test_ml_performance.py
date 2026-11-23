"""ML step 3: performance model (features, trend, pooled training, readiness)."""
import json

import numpy as np
import pytest

from app.core.config import settings
from app.ml import performance as pm
from app.ml import registry
from app.repositories import ml_repository
from app.services import performance_service as ps
from tests.conftest import COURSE, OTHER_COURSE, USER, new_id
from tests.performance_corpus import seed_into_db, simulate, to_histories

TRAIN, STATUS = "/internal/ml/performance/train", "/internal/ml/performance/status"
READINESS = f"/internal/courses/{COURSE}/ml/readiness"
DAY = 86400.0
ENSEMBLE = 20


# ============================================================ helpers
def attempt(i, correct, rt=10.0, difficulty=1, gap=DAY, start=0.0):
    return {"t": start + i * gap, "correct": correct, "rt": rt, "difficulty": difficulty}


def seq(results, **kw):
    return [attempt(i, c, **kw) for i, c in enumerate(results)]


@pytest.fixture(autouse=True)
def memory_store(monkeypatch):
    monkeypatch.setattr(settings, "ml_performance_ensemble_size", ENSEMBLE)
    store = registry.MemoryArtifactStore()
    registry.set_artifact_store(store)
    yield store
    registry.set_artifact_store(registry.SupabaseArtifactStore())


@pytest.fixture(scope="module")
def sim():
    return simulate(n_students=12, topics_per_student=3, attempts_per_topic=40, seed=7)


@pytest.fixture
def pooled(db, sim):
    """12 students; student 0 is the test student (USER, COURSE)."""
    ids = {
        "new_id": new_id,
        "user": {0: USER, **{i: new_id() for i in range(1, 12)}},
        "course": {0: COURSE, **{i: new_id() for i in range(1, 12)}},
        "topic": {(s, t): new_id() for s in range(12) for t in range(3)},
    }
    for topic in range(3):
        db.seed("syllabus_topics", {"id": ids["topic"][(0, topic)], "title": f"Topic {topic}", "position": topic + 1,
                                    "parent_topic_id": None, "user_id": USER, "course_id": COURSE})
    seed_into_db(db, sim, ids)
    return ids


def train(api, **extra):
    return api.post(TRAIN, json=extra)


def readiness(api, **params):
    return api.get(READINESS, params={"user_id": USER, **params})


# ============================================================ features
class TestFeatures:
    def f(self, results, now_gap=DAY, q=1, **kw):
        history = seq(results, **kw)
        return pm.history_features(history, history[-1]["t"] + now_gap, q)

    def test_accuracy_is_smoothed_so_one_answer_is_not_certainty(self):
        assert self.f([1])["acc_all"] == pytest.approx(2 / 3)
        assert self.f([0])["acc_all"] == pytest.approx(1 / 3)
        assert self.f([1, 1, 0, 1])["acc_all"] == pytest.approx(4 / 6)

    def test_recent_accuracy_looks_at_the_last_five_only(self):
        f = self.f([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        assert f["acc_recent"] == pytest.approx(6 / 7) and f["acc_all"] == pytest.approx(6 / 12)

    def test_moving_average_weights_recent_answers_more(self):
        improving, declining = self.f([0, 0, 0, 1, 1, 1]), self.f([1, 1, 1, 0, 0, 0])

        assert improving["ewma"] > declining["ewma"] and improving["acc_all"] == declining["acc_all"]

    def test_slope_sign_and_short_histories(self):
        assert self.f([0, 0, 1, 1, 1])["slope"] > 0 and self.f([1, 1, 0, 0, 0])["slope"] < 0
        assert self.f([1, 0, 1, 0])["slope"] == pytest.approx(0, abs=0.2)
        assert self.f([1, 0])["slope"] == 0.0                         # fewer than 3 answers

    def test_days_since_last_is_capped_and_zero_without_history(self):
        assert self.f([1, 0], now_gap=3 * DAY)["days_since_last"] == pytest.approx(3)
        assert self.f([1, 0], now_gap=400 * DAY)["days_since_last"] == pm.MAX_DAYS
        assert pm.history_features([], 5.0, 1)["days_since_last"] == 0 and pm.history_features([], 5.0, 1)["n_prev"] == 0

    def test_response_time_is_averaged_capped_and_flagged(self):
        assert self.f([1, 0], rt=20.0)["mean_response_time"] == 20 and self.f([1, 0], rt=20.0)["has_response_time"] == 1
        assert self.f([1, 0], rt=900.0)["mean_response_time"] == pm.MAX_RESPONSE_SECONDS
        none = pm.history_features([{"t": 0, "correct": 1, "rt": None, "difficulty": 1}], DAY, 1)
        assert none["mean_response_time"] == 0 and none["has_response_time"] == 0

    def test_difficulty_features(self):
        history = [attempt(0, 1, difficulty=0), attempt(1, 1, difficulty=2)]
        f = pm.history_features(history, 2 * DAY, 2)

        assert f["mean_prev_difficulty"] == 1.0 and f["q_difficulty"] == 2

    def test_streak_and_last_correct(self):
        assert self.f([0, 1, 1, 1])["streak"] == 3 and self.f([1, 1, 0])["streak"] == 0
        assert self.f([1, 0])["last_correct"] == 0 and self.f([0, 1])["last_correct"] == 1
        assert self.f([1] * 30)["streak"] == pm.MAX_STREAK

    def test_matrix_follows_the_feature_order(self):
        X = pm.feature_matrix([self.f([1, 0, 1])])

        assert X.shape == (1, len(pm.FEATURE_NAMES)) and X[0, pm.FEATURE_NAMES.index("n_prev")] == 3
        assert pm.feature_matrix([]).shape == (0, len(pm.FEATURE_NAMES))


# ============================================================ samples (no future leakage)
class TestSamples:
    def test_one_sample_per_attempt_after_the_minimum_history(self):
        rows, y, t = pm.build_samples([seq([1, 0, 1, 1, 0, 1])], min_history=2)

        assert len(rows) == 4 and list(y) == [1, 1, 0, 1] and [r["n_prev"] for r in rows] == [2, 3, 4, 5]
        assert list(t) == [2 * DAY, 3 * DAY, 4 * DAY, 5 * DAY]

    def test_histories_shorter_than_the_minimum_give_nothing(self):
        assert len(pm.build_samples([seq([1, 0])], min_history=2)[0]) == 0

    def test_features_use_only_earlier_attempts(self):
        base = seq([1, 0, 1, 1, 0, 1, 0, 0])
        changed = [dict(a) for a in base]
        changed[-1]["correct"] = 1                                     # change the LAST answer only

        rows_a, y_a, _ = pm.build_samples([base])
        rows_b, y_b, _ = pm.build_samples([changed])

        assert rows_a[:-1] == rows_b[:-1]                              # every earlier sample identical
        assert y_a[-1] != y_b[-1]                                      # only the last target differs

    def test_input_order_does_not_matter(self):
        history = seq([1, 0, 1, 1, 0, 1])

        assert pm.build_samples([history])[0] == pm.build_samples([list(reversed(history))])[0]

    def test_each_student_topic_history_is_separate(self):
        rows, _y, _t = pm.build_samples([seq([1, 1, 1, 1]), seq([0, 0, 0, 0])], min_history=2)

        assert [r["acc_all"] for r in rows] == [0.75, 0.8, 0.25, 0.2]


# ============================================================ trend and bands
class TestTrend:
    def test_improving_flat_declining(self):
        assert pm.trend(seq([0, 0, 0, 1, 1, 1, 1]))["label"] == "improving"
        assert pm.trend(seq([1, 1, 1, 0, 0, 0, 0]))["label"] == "declining"
        assert pm.trend(seq([1, 0, 0, 1, 1, 0, 0, 1]))["label"] == "flat"

    def test_needs_enough_attempts(self):
        t = pm.trend(seq([0, 1, 1, 1]))

        assert t == {"label": "not_enough_data", "slope": None, "attempts_used": 4}

    def test_only_the_latest_window_counts(self):
        old_slump_then_recovery = seq([0] * 20 + [1] * 10)

        assert pm.trend(old_slump_then_recovery, window=10)["label"] == "flat"      # last 10 are all correct

    def test_threshold_is_adjustable(self):
        history = seq([0, 0, 1, 0, 1, 1, 1, 1])
        slope = pm.trend(history)["slope"]

        assert pm.trend(history, threshold=slope + 0.01)["label"] == "flat"
        assert pm.trend(history, threshold=slope - 0.01)["label"] == "improving"

    @pytest.mark.parametrize("value,expected", [(None, None), (0.0, "low"), (0.49, "low"), (0.5, "medium"),
                                                  (0.74, "medium"), (0.75, "high"), (1.0, "high")])
    def test_bands(self, value, expected):
        assert pm.band(value) == expected


# ============================================================ evaluation (no DB)
@pytest.fixture(scope="module")
def data():
    rows, y, t = pm.build_samples(to_histories(simulate()))
    return pm.feature_matrix(rows), y, t


class TestEvaluation:
    def test_baseline_probability_is_the_feature_itself(self):
        X = np.array([[0.0] * len(pm.FEATURE_NAMES)])
        X[0, pm.FEATURE_NAMES.index("acc_all")] = 0.7

        model = pm.FeatureProbability(pm.FEATURE_NAMES.index("acc_all")).fit(X, [1])

        assert model.predict_proba(X)[0].tolist() == pytest.approx([0.3, 0.7])
        assert pm.FeatureProbability(0).predict_proba(np.array([[5.0], [-3.0]]))[:, 1].tolist() == [0.999, 0.001]

    def test_split_is_by_time_so_the_test_period_is_later(self):
        times = np.array([5, 1, 9, 3, 7, 2, 8, 4, 6, 10], dtype=float)

        train, test = pm.temporal_split(times, 0.3)

        assert len(train) == 7 and len(test) == 3
        assert times[train].max() < times[test].min() and sorted(times[test]) == [8, 9, 10]

    def test_score_of_always_guessing_the_average_is_zero_skill(self):
        y = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        m = pm.score(y, np.full(8, y.mean()), y.mean())

        assert m["skill"] == pytest.approx(0, abs=1e-3)

    def test_score_rewards_good_and_punishes_bad_predictions(self):
        y = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        good = pm.score(y, np.where(y == 1, 0.9, 0.1), 0.5)
        bad = pm.score(y, np.where(y == 1, 0.1, 0.9), 0.5)

        assert good["skill"] > 0.5 and good["auc"] == 1.0 and good["accuracy"] == 1.0
        assert bad["skill"] < 0 and bad["auc"] == 0.0

    def test_auc_is_none_when_only_one_outcome(self):
        assert pm.score(np.ones(5, dtype=int), np.full(5, 0.7), 0.5)["auc"] is None

    def test_calibration_table_compares_predicted_with_observed(self):
        y = np.array([1, 1, 0, 0, 1, 0, 1, 1])
        p = np.array([0.9, 0.85, 0.1, 0.15, 0.9, 0.2, 0.8, 0.95])

        table, ece = pm.calibration_table(y, p, bins=5)

        assert sum(b["samples"] for b in table) == 8 and 0 <= ece <= 1
        assert table[0]["observed_rate"] == 0 and table[-1]["observed_rate"] == 1

    def test_compares_four_candidates_with_two_baselines(self, data):
        X, y, t = data
        results, best, split = pm.evaluate_candidates(X, y, t, 0.3, 42, ENSEMBLE)

        assert set(results) == {"baseline_overall", "baseline_ewma", "logistic_regression", "gradient_boosting"}
        assert {n for n, r in results.items() if r["is_baseline"]} == {"baseline_overall", "baseline_ewma"}
        assert results[best]["selection_score"] == max(r["selection_score"] for r in results.values())
        assert split["train"] + split["test"] == len(y)

    def test_a_trained_model_beats_guessing_the_average_on_the_later_period(self, data):
        X, y, t = data
        results, best, _s = pm.evaluate_candidates(X, y, t, 0.3, 42, ENSEMBLE)

        assert results[best]["metrics"]["skill"] > 0 and results[best]["metrics"]["auc"] > 0.6

    def test_repeatable_with_the_same_seed(self, data):
        X, y, t = data
        a = pm.evaluate_candidates(X, y, t, 0.3, 42, ENSEMBLE)
        b = pm.evaluate_candidates(X, y, t, 0.3, 42, ENSEMBLE)

        assert {n: r["selection_score"] for n, r in a[0].items()} == {n: r["selection_score"] for n, r in b[0].items()}

    def test_refuses_when_a_period_has_only_one_outcome(self):
        X = np.random.RandomState(0).rand(100, len(pm.FEATURE_NAMES))

        with pytest.raises(ValueError, match="only correct"):
            pm.evaluate_candidates(X, np.ones(100, dtype=int), np.arange(100.0), 0.3, 42, ENSEMBLE)

    def test_the_model_learns_that_harder_questions_lower_the_chance(self, data):
        X, y, _t = data
        model = pm.fit_final("logistic_regression", X, y, 42, ENSEMBLE)

        weights = {i["feature"]: i for i in pm.feature_importance(model)}

        assert weights["q_difficulty"]["effect"] == "lowers readiness"

    def test_importance_formats(self, data):
        X, y, _t = data

        assert pm.feature_importance(pm.fit_final("gradient_boosting", X, y, 42, ENSEMBLE))[0]["importance"] > 0
        assert pm.feature_importance(pm.fit_final("baseline_overall", X, y, 42)) is None

    def test_predictions_are_probabilities(self, data):
        X, y, _t = data
        p = pm.predict_probability(pm.fit_final("logistic_regression", X, y, 42), X[:50])

        assert ((p > 0) & (p < 1)).all() and len(pm.predict_probability(None, np.empty((0, 12)))) == 0


# ============================================================ pooled training
class TestTraining:
    def test_trains_one_global_model_from_pooled_data(self, api, db, pooled, memory_store):
        r = train(api)
        assert r.status_code == 200
        card = r.json()

        assert card["scope"] == "global" and card["version"] == 1 and card["is_active"] is True
        assert card["algorithm"] in pm.LABELS and len(card["comparison"]) == 4
        assert sum(c["selected"] for c in card["comparison"]) == 1 and sum(c["is_baseline"] for c in card["comparison"]) == 2

        m = card["metrics"]
        assert m["selection_metric"] == "log_loss_skill" and set(m["baseline_scores"]) == {"baseline_overall", "baseline_ewma"}
        assert m["improvement_over_best_baseline"] == round(
            m["selection_score"] - m["baseline_scores"][m["best_baseline"]], 4)
        assert m["split"]["train"] > m["split"]["test"] >= settings.ml_performance_min_test
        assert m["test_metrics"]["samples"] == m["split"]["test"] and m["calibration"] and m["feature_importance"]

        data = m["training_data"]
        assert data["students"] == 12 and data["topic_histories"] == 36 and data["attempts"] == 1440
        assert data["samples_used"] == 1440 - 36 * settings.ml_performance_min_history

        row = db.rows("ml_models")[0]
        assert row["user_id"] is None and row["course_id"] is None
        assert row["artifact_path"] == "ml-models/global/performance_v1.joblib" and row["artifact_path"] in memory_store.files

    def test_nothing_personal_is_stored(self, api, db, pooled):
        card = train(api).json()
        row = db.rows("ml_models")[0]
        _row, bundle = registry.load_active_bundle(None, None, "performance")
        everything = json.dumps(card) + json.dumps(row, default=str)

        assert set(bundle) == {"model", "algorithm", "feature_names"}
        for user_id in pooled["user"].values():
            assert user_id not in everything
        assert "Q " not in everything and "no names, ids or question text" in card["metrics"]["privacy_note"]

    def test_retraining_makes_version_2_and_keeps_one_active(self, api, db, pooled):
        first, second = train(api).json(), train(api).json()

        assert second["version"] == 2 and second["activated"] is True
        assert second["metrics"]["selection_score"] == first["metrics"]["selection_score"]
        assert [r["is_active"] for r in sorted(db.rows("ml_models"), key=lambda r: r["version"])] == [False, True]

    def test_a_worse_version_is_kept_out_unless_forced(self, api, db, pooled):
        train(api)
        db.rows("ml_models")[0]["metrics"]["selection_score"] = 2.0

        worse = train(api).json()
        assert worse["activated"] is False and "Kept version 1" in worse["activation_reason"]

        forced = train(api, force=True).json()
        assert forced["activated"] is True and {r["version"] for r in db.rows("ml_models") if r["is_active"]} == {3}

    def test_too_little_history_returns_422_and_stores_nothing(self, api, db, sim):
        ids = {"new_id": new_id, "user": {i: new_id() for i in range(12)}, "course": {i: new_id() for i in range(12)},
               "topic": {(s, t): new_id() for s in range(12) for t in range(3)}}
        seed_into_db(db, [a for a in sim if a["student"] == 0 and a["topic"] == 0], ids)       # 40 answers

        r = train(api)

        assert r.status_code == 422 and "Not enough practice history" in r.json()["detail"]
        assert "at least 150" in r.json()["detail"] and "recent accuracy" in r.json()["detail"]
        assert db.rows("ml_models") == []

    def test_a_period_with_only_correct_answers_cannot_be_trained(self, api, db, sim):
        ids = {"new_id": new_id, "user": {i: new_id() for i in range(12)}, "course": {i: new_id() for i in range(12)},
               "topic": {(s, t): new_id() for s in range(12) for t in range(3)}}
        seed_into_db(db, [{**a, "correct": 1} for a in sim], ids)

        r = train(api)

        assert r.status_code == 422 and "only correct (or only wrong)" in r.json()["detail"]

    def test_answers_to_questions_without_a_topic_are_ignored(self, api, db, pooled):
        qid = new_id()
        db.seed("generated_questions", {"id": qid, "question_text": "x", "difficulty": "easy", "status": "active",
                                        "topic_id": None, "user_id": USER, "course_id": COURSE})
        for i in range(5):
            db.seed("practice_attempts", {"session_id": new_id(), "question_id": qid, "user_id": USER,
                                          "is_correct": True, "answered_at": f"2026-12-0{i + 1}T10:00:00+00:00"})

        assert train(api).json()["metrics"]["training_data"]["attempts"] == 1440

    def test_requires_internal_key_and_a_valid_body(self, api, anon_api):
        assert anon_api.post(TRAIN, json={}).status_code == 401
        assert api.post(TRAIN, json={"force": "maybe"}).status_code == 422


# ============================================================ global scope in the registry
class TestGlobalScope:
    def test_a_global_model_and_a_course_model_do_not_mix(self, db):
        course_row = {"id": new_id(), "user_id": USER, "course_id": COURSE, "model_type": "performance",
                      "algorithm": "x", "version": 1, "n_training_samples": 1, "metrics": {}, "comparison": [],
                      "artifact_path": "p", "is_active": True}
        global_row = {**course_row, "id": new_id(), "user_id": None, "course_id": None, "artifact_path": "g"}
        db.seed("ml_models", course_row, global_row)

        assert ml_repository.get_active(None, None, "performance")["artifact_path"] == "g"
        assert ml_repository.get_active(USER, COURSE, "performance")["artifact_path"] == "p"
        assert [r["artifact_path"] for r in ml_repository.list_models(None, None, "performance")] == ["g"]

    def test_activating_a_global_model_leaves_course_models_alone(self, db):
        base = {"model_type": "performance", "algorithm": "x", "n_training_samples": 1, "metrics": {},
                "comparison": [], "artifact_path": "p"}
        course = {**base, "id": new_id(), "user_id": USER, "course_id": COURSE, "version": 1, "is_active": True}
        old_global = {**base, "id": new_id(), "user_id": None, "course_id": None, "version": 1, "is_active": True}
        new_global = {**base, "id": new_id(), "user_id": None, "course_id": None, "version": 2, "is_active": False}
        db.seed("ml_models", course, old_global, new_global)

        ml_repository.activate(new_global["id"], None, None, "performance")
        state = {(r["user_id"], r["version"]): r["is_active"] for r in db.rows("ml_models")}

        assert state == {(USER, 1): True, (None, 1): False, (None, 2): True}


# ============================================================ status
class TestStatus:
    def test_before_training(self, api, db, pooled):
        body = api.get(STATUS).json()

        assert body["scope"] == "global" and body["using_fallback"] is True and body["active_model"] is None
        assert body["can_train"] is True and body["message"] is None
        assert body["training_data"]["students"] == 12 and body["training_data"]["training_examples"] > 1000

    def test_explains_what_is_missing(self, api, db):
        body = api.get(STATUS).json()

        assert body["can_train"] is False and "Not enough practice history" in body["message"]

    def test_after_training(self, api, db, pooled):
        train(api); train(api)

        body = api.get(STATUS).json()

        assert body["using_fallback"] is False and body["active_model"]["version"] == 2
        assert [h["version"] for h in body["history"]] == [2, 1]

    def test_requires_internal_key(self, anon_api):
        assert anon_api.get(STATUS).status_code == 401


# ============================================================ readiness
class TestReadiness:
    @pytest.fixture
    def now(self, monkeypatch, sim):
        last = max(a["t"] for a in sim if a["student"] == 0).timestamp() + 2 * DAY
        monkeypatch.setattr(ps, "_now", lambda: last)
        return last

    def test_predicts_readiness_for_every_topic_with_the_global_model(self, api, db, pooled, now):
        train(api)

        r = readiness(api)
        assert r.status_code == 200
        body = r.json()

        assert body["model_used"] == "performance" and body["model_version"] == 1 and body["difficulty"] == "medium"
        assert body["algorithm"] in pm.LABELS and body["model_skill"] is not None
        assert [t["topic_title"] for t in body["topics"]] == ["Topic 0", "Topic 1", "Topic 2"]

        for topic in body["topics"]:
            assert topic["attempts"] == 40 and topic["status"] == "assessed"
            assert 0 < topic["readiness"] < 1 and topic["band"] == pm.band(topic["readiness"])
            assert 0 <= topic["accuracy"] <= 1 and 0 <= topic["recent_accuracy"] <= 1
            assert topic["trend"]["label"] in ("improving", "flat", "declining")
            assert topic["days_since_last"] >= 2

        assert body["summary"]["assessed"] == 3 and body["summary"]["not_assessed"] == 0
        assert sum(body["summary"]["by_band"].values()) == 3 and sum(body["summary"]["by_trend"].values()) == 3

    def test_weakest_topics_are_listed_lowest_readiness_first(self, api, db, pooled, now):
        train(api)

        body = readiness(api).json()
        by_readiness = sorted(t["readiness"] for t in body["topics"])

        assert [w["readiness"] for w in body["weakest"]] == by_readiness

    def test_topics_never_practised_are_not_assessed_and_not_invented(self, api, db, pooled, now):
        db.seed("syllabus_topics", {"id": new_id(), "title": "Untouched", "position": 9, "parent_topic_id": None,
                                    "user_id": USER, "course_id": COURSE})
        train(api)

        body = readiness(api).json()
        untouched = next(t for t in body["topics"] if t["topic_title"] == "Untouched")

        assert untouched["status"] == "not_assessed" and untouched["readiness"] is None and untouched["band"] is None
        assert untouched["attempts"] == 0 and untouched["trend"]["label"] == "not_enough_data"
        assert body["summary"]["not_assessed"] == 1 and "Untouched" not in [w["topic_title"] for w in body["weakest"]]

    def test_a_topic_with_one_or_two_answers_is_marked_few_attempts(self, api, db, pooled, now):
        topic = new_id()
        db.seed("syllabus_topics", {"id": topic, "title": "Brand new", "position": 9, "parent_topic_id": None,
                                    "user_id": USER, "course_id": COURSE})
        qid = new_id()
        db.seed("generated_questions", {"id": qid, "question_text": "x", "difficulty": "medium", "status": "active",
                                        "topic_id": topic, "user_id": USER, "course_id": COURSE})
        db.seed("practice_attempts", {"session_id": new_id(), "question_id": qid, "user_id": USER, "is_correct": True,
                                      "answered_at": "2026-12-01T10:00:00+00:00"})
        train(api)

        item = next(t for t in readiness(api).json()["topics"] if t["topic_title"] == "Brand new")

        assert item["status"] == "few_attempts" and item["attempts"] == 1 and item["readiness"] is not None
        assert item["trend"]["label"] == "not_enough_data"

    def test_harder_questions_lower_the_readiness(self, api, db, pooled, now):
        train(api)

        easy, hard = readiness(api, difficulty="easy").json(), readiness(api, difficulty="hard").json()

        assert easy["difficulty"] == "easy" and hard["difficulty"] == "hard"
        for e, h in zip(easy["topics"], hard["topics"]):
            assert e["readiness"] >= h["readiness"]

    def test_waiting_lowers_readiness_when_the_model_learned_forgetting(self, api, db, pooled, monkeypatch, sim):
        train(api)
        last = max(a["t"] for a in sim if a["student"] == 0).timestamp()

        monkeypatch.setattr(ps, "_now", lambda: last + 1 * DAY)
        soon = readiness(api).json()["topics"]
        monkeypatch.setattr(ps, "_now", lambda: last + 45 * DAY)
        later = readiness(api).json()["topics"]

        assert sum(t["readiness"] for t in later) <= sum(t["readiness"] for t in soon)
        assert all(t["days_since_last"] > 40 for t in later)

    def test_without_a_model_it_uses_recent_accuracy_and_says_so(self, api, db, pooled, now):
        body = readiness(api).json()

        assert body["model_used"] == "fallback_recent_accuracy" and body["model_version"] is None
        assert body["algorithm"] is None and body["model_skill"] is None
        for topic in body["topics"]:
            assert 0 < topic["readiness"] < 1 and topic["status"] == "assessed"

    def test_a_missing_model_file_falls_back_instead_of_failing(self, api, db, pooled, now, memory_store):
        train(api)
        memory_store.files.clear()

        r = readiness(api)

        assert r.status_code == 200 and r.json()["model_used"] == "fallback_recent_accuracy"

    def test_other_students_data_is_never_used(self, api, db, pooled, now):
        train(api)

        topics = readiness(api).json()["topics"]

        assert all(t["attempts"] == 40 for t in topics)                       # not 40 x 12

    def test_another_course_of_mine_is_separate(self, api, db, pooled, now):
        db.seed("syllabus_topics", {"id": new_id(), "title": "Elsewhere", "position": 1, "parent_topic_id": None,
                                    "user_id": USER, "course_id": OTHER_COURSE})

        body = api.get(f"/internal/courses/{OTHER_COURSE}/ml/readiness", params={"user_id": USER}).json()

        assert [t["topic_title"] for t in body["topics"]] == ["Elsewhere"] and body["topics"][0]["status"] == "not_assessed"

    def test_a_course_without_topics_is_empty_not_an_error(self, api, db, now):
        body = readiness(api).json()

        assert body["topics"] == [] and body["weakest"] == [] and body["summary"]["topics"] == 0

    def test_validation(self, api, anon_api):
        assert anon_api.get(READINESS, params={"user_id": USER}).status_code == 401
        assert api.get(READINESS).status_code == 422
        assert readiness(api, difficulty="impossible").status_code == 422
        assert api.get(READINESS, params={"user_id": "nope"}).status_code == 422
