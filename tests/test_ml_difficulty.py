"""ML step 2: difficulty prediction (features, models, registry, predictions)."""
import numpy as np
import pytest

from app.core.config import settings
from app.ml import difficulty as dm
from app.ml import registry
from app.services import difficulty_service as ds
from tests.conftest import COURSE, OTHER_COURSE, OTHER_USER, USER, new_id
from tests.difficulty_corpus import PAPER_QUESTIONS, make_questions

BASE = f"/internal/courses/{COURSE}/ml/difficulty"
TRAIN, STATUS, PREDICT = f"{BASE}/train", f"{BASE}/status", f"{BASE}/predict"


# ============================================================ fixtures
TREES = 12                      # small ensembles keep the tests fast


@pytest.fixture(autouse=True)
def memory_store(monkeypatch):
    monkeypatch.setattr(settings, "ml_difficulty_ensemble_size", TREES)
    store = registry.MemoryArtifactStore()
    registry.set_artifact_store(store)
    yield store
    registry.set_artifact_store(registry.SupabaseArtifactStore())


def seed_questions(db, per_level=14, user=USER, course=COURSE, status="active"):
    ids = []
    for text, level in make_questions(per_level):
        qid = new_id()
        ids.append(qid)
        db.seed("generated_questions", {
            "id": qid, "question_text": text, "difficulty": level, "status": status,
            "topic_id": None, "user_id": user, "course_id": course})
    return ids


def add_attempts(db, qid, correct, wrong, user=USER):
    for is_correct in [True] * correct + [False] * wrong:
        db.seed("practice_attempts", {"session_id": new_id(), "question_id": qid,
                                      "user_id": user, "is_correct": is_correct})


def add_paper_question(db, text, marks=None, status="pending", predicted=None, user=USER, course=COURSE):
    qid = new_id()
    db.seed("past_paper_questions", {
        "id": qid, "question_text": text, "question_number": "1", "marks": marks,
        "question_type": "short", "review_status": status, "predicted_difficulty": predicted,
        "user_id": user, "course_id": course})
    return qid


def train(api, **extra):
    return api.post(TRAIN, json={"user_id": USER, **extra})


def predict(api, **extra):
    return api.post(PREDICT, json={"user_id": USER, **extra})


def paper_row(db, qid):
    return next(q for q in db.rows("past_paper_questions") if q["id"] == qid)


# ============================================================ features
class TestFeatures:
    def f(self, text, **kw):
        return dm.extract_features(text, **kw)

    @pytest.mark.parametrize("text,level", [
        ("Define a stack.", 1), ("What is a queue?", 1), ("List two uses of a heap.", 1),
        ("Explain how a stack works.", 2), ("Differentiate between BFS and DFS.", 2),
        ("Calculate the height of the tree.", 3), ("Insert 5 into the AVL tree.", 3),
        ("Compare quick sort and merge sort.", 4), ("Prove that the algorithm terminates.", 4),
        ("The stack is a structure.", 0),
    ])
    def test_command_verb_level(self, text, level):
        assert self.f(text)["verb_level"] == level

    def test_the_first_command_verb_decides(self):
        assert self.f("Explain and then justify your answer.")["verb_level"] == 2

    def test_a_noun_later_in_the_question_is_not_a_command(self):
        # "sort" is a command in "Sort the list" but a noun in "quick sort"
        assert self.f("Sort the list of numbers.")["verb_level"] == 3
        assert self.f("Define quick sort.")["verb_level"] == 1
        assert self.f("List two uses of an insertion sort.")["verb_level"] == 1

    def test_counts_length_digits_symbols_and_subparts(self):
        features = self.f("Solve (a) 2 + 3 = 5 and (b) 10 / 2, then state why. Is it not true?")

        assert features["n_words"] > 10 and features["n_digits"] == 6
        assert features["n_math_symbols"] == 3                    # + = /
        assert features["n_subparts"] == 2 and features["has_negation"] == 1
        assert features["n_sentences"] == 2 and features["n_commas"] == 1

    def test_marks_are_used_when_present(self):
        assert self.f("Define a stack.", marks=5) ["marks"] == 5 and self.f("Define a stack.", marks=5)["has_marks"] == 1
        assert self.f("Define a stack.")["has_marks"] == 0 and self.f("Define a stack.")["marks"] == 0

    def test_empty_or_missing_text_does_not_crash(self):
        for value in ("", "   ", None):
            features = self.f(value)
            assert features["n_words"] == 0 and features["verb_level"] == 0 and features["n_sentences"] == 0

    def test_option_features_for_multiple_choice(self):
        close = [{"option_text": "A stack is LIFO", "is_correct": True},
                 {"option_text": "A stack is FIFO", "is_correct": False},
                 {"option_text": "A stack is LILO", "is_correct": False},
                 {"option_text": "A stack is FILO", "is_correct": False}]
        far = [{"option_text": "Last in first out ordering of elements", "is_correct": True},
               {"option_text": "Binary", "is_correct": False},
               {"option_text": "Hash", "is_correct": False},
               {"option_text": "Graph", "is_correct": False}]

        a, b = self.f("Which is true?", options=close), self.f("Which is true?", options=far)

        assert a["has_options"] == 1 and a["n_options"] == 4
        assert a["option_overlap"] > b["option_overlap"]           # close distractors score higher
        assert a["correct_is_longest"] == 0 and b["correct_is_longest"] == 1

    def test_no_options_means_zero_option_features(self):
        features = self.f("Explain a stack.", options=[])

        assert features["has_options"] == 0 and features["option_overlap"] == 0 == features["n_options"]

    def test_matrix_follows_the_feature_order(self):
        rows = [self.f("Define a stack."), self.f("Prove it, (a) and (b).")]
        X = dm.feature_matrix(rows)

        assert X.shape == (2, len(dm.FEATURE_NAMES))
        assert X[1, dm.FEATURE_NAMES.index("verb_level")] == 4
        assert dm.feature_matrix([]).shape == (0, len(dm.FEATURE_NAMES))


# ============================================================ baseline rule
class TestVerbRule:
    def levels(self, *texts):
        X = dm.feature_matrix([dm.extract_features(t) for t in texts])
        return [p["difficulty"] for p in dm.rule_predict(X)]

    def test_verbs_set_the_level(self):
        assert self.levels("Define a stack.", "Explain a stack.", "Calculate the height.",
                           "Prove the claim.") == ["easy", "medium", "medium", "hard"]

    def test_unknown_verb_is_medium_and_long_questions_are_one_level_harder(self):
        long_text = "Define " + "a very long winded question " * 8

        assert self.levels("The tree.") == ["medium"]
        assert self.levels(long_text) == ["medium"]                # easy + long -> medium
        assert self.levels("Explain " + "a very long winded question " * 8) == ["hard"]

    def test_probabilities_sum_to_one(self):
        X = dm.feature_matrix([dm.extract_features("Define a stack.")])

        assert sum(dm.rule_predict(X)[0]["probabilities"].values()) == pytest.approx(1, abs=1e-3)


    def test_works_with_the_integer_labels_cross_validation_uses(self):
        """Regression: cross-validation codes labels as 0,1,2, not 'easy','hard','medium'."""
        X, y = corpus_arrays()
        names = sorted(set(y))                                       # ['easy', 'hard', 'medium']
        rule = dm.VerbRule(class_names=names).fit(X, np.array([names.index(v) for v in y]))

        proba = rule.predict_proba(X)
        predicted = [names[i] for i in np.argmax(proba, axis=1)]

        assert np.mean(np.array(predicted) == np.array(y)) > 0.6      # NOT stuck on one class
        assert set(predicted) == {"easy", "medium", "hard"}


# ============================================================ evaluation (no DB)
def corpus_arrays(per_level=14):
    questions = make_questions(per_level)
    X = dm.feature_matrix([dm.extract_features(t) for t, _l in questions])
    return X, [level for _t, level in questions]


class TestEvaluation:
    def test_compares_five_candidates_with_two_baselines(self):
        X, y = corpus_arrays()
        results, best, classes = dm.evaluate_candidates(X, y, [np.nan] * len(y), 4, 42, TREES)

        assert set(results) == {"majority_class", "verb_rule", "decision_tree",
                                "random_forest", "gradient_boosting"}
        assert {n for n, r in results.items() if r["is_baseline"]} == {"majority_class", "verb_rule"}
        assert classes == ["easy", "medium", "hard"] and best in results

    def test_the_rule_baseline_scores_the_same_in_cross_validation_as_when_used_directly(self):
        X, y = corpus_arrays()
        results, _b, _c = dm.evaluate_candidates(X, y, [np.nan] * len(y), 4, 42, TREES)

        direct = dm.VerbRule().fit(X, np.array(y)).predict(X)
        direct_accuracy = float(np.mean(direct == np.array(y)))

        # the rule has nothing to learn, so cross-validation must agree with direct use
        assert results["verb_rule"]["metrics"]["accuracy"] == pytest.approx(direct_accuracy, abs=0.001)
        assert results["verb_rule"]["metrics"]["accuracy"] > 0.6

    def test_majority_baseline_is_near_chance(self):
        X, y = corpus_arrays()
        results, _b, _c = dm.evaluate_candidates(X, y, [np.nan] * len(y), 4, 42, TREES)

        assert results["majority_class"]["metrics"]["accuracy"] == pytest.approx(1 / 3, abs=0.05)

    def test_learns_the_pattern_on_separable_data(self):
        X, y = corpus_arrays()
        results, best, _c = dm.evaluate_candidates(X, y, [np.nan] * len(y), 4, 42, TREES)

        assert results[best]["metrics"]["accuracy"] > 0.85
        assert results[best]["selection_score"] > results["majority_class"]["selection_score"] + 0.4

    def test_repeatable_with_the_same_seed(self):
        X, y = corpus_arrays()
        a = dm.evaluate_candidates(X, y, [np.nan] * len(y), 4, 42, TREES)
        b = dm.evaluate_candidates(X, y, [np.nan] * len(y), 4, 42, TREES)

        assert {n: r["selection_score"] for n, r in a[0].items()} == {n: r["selection_score"] for n, r in b[0].items()}
        assert a[1] == b[1]

    def test_ordinal_measures_are_consistent(self):
        X, y = corpus_arrays()
        results, best, _c = dm.evaluate_candidates(X, y, [np.nan] * len(y), 4, 42, TREES)
        m = results[best]["metrics"]

        assert m["within_one_level"] >= m["accuracy"] and m["mean_level_error"] >= 0

    def test_probability_columns_are_reordered_easy_to_hard(self):
        model_classes = ["easy", "hard", "medium"]                       # sklearn's alphabetical order
        proba = np.array([[0.1, 0.2, 0.7]])                              # easy .1  hard .2  medium .7

        ordered = dm.to_ordinal_proba(proba, model_classes, ["easy", "medium", "hard"])

        assert ordered.tolist() == [[0.1, 0.7, 0.2]]

    def test_expected_level(self):
        proba = np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 0.5]])

        assert dm.expected_level(proba, ["easy", "medium", "hard"]).tolist() == [2.0, 1.0]

    def test_agreement_with_observed_results_is_reported_when_available(self):
        X, y = corpus_arrays()
        rate = [{"easy": 0.1, "medium": 0.45, "hard": 0.85}[v] + (i % 3) * 0.01 for i, v in enumerate(y)]

        results, best, _c = dm.evaluate_candidates(X, y, rate, 4, 42, TREES)
        agreement = results[best]["agreement_with_observed"]

        assert agreement["samples"] == len(y) and agreement["spearman"] > 0.7

    def test_agreement_is_none_without_enough_observed_data(self):
        X, y = corpus_arrays()
        few = [np.nan] * (len(y) - 3) + [0.1, 0.5, 0.9]

        results, best, _c = dm.evaluate_candidates(X, y, few, 4, 42, TREES)

        assert results[best]["agreement_with_observed"] is None

    def test_only_tree_models_report_feature_importance(self):
        X, y = corpus_arrays()
        tree = dm.fit_final("random_forest", X, y, 42, TREES)
        dummy = dm.fit_final("majority_class", X, y, 42)

        ranked = dm.feature_importances(tree)

        assert ranked[0]["importance"] >= ranked[-1]["importance"] and {r["feature"] for r in ranked} <= set(dm.FEATURE_NAMES)
        assert dm.feature_importances(dummy) is None

    def test_a_model_trained_on_two_levels_still_reports_all_three(self):
        X, y = corpus_arrays()
        keep = [i for i, v in enumerate(y) if v != "hard"]
        model = dm.fit_final("random_forest", X[keep], [y[i] for i in keep], 42, TREES)

        result = dm.predict_levels(model, X[:1])[0]

        assert result["probabilities"]["hard"] == 0.0 and result["difficulty"] == "easy"


# ============================================================ training
class TestTraining:
    def test_trains_compares_and_stores_a_model(self, api, db, memory_store):
        seed_questions(db)

        r = train(api)
        assert r.status_code == 200
        card = r.json()

        assert card["version"] == 1 and card["is_active"] is True and card["activated"] is True
        assert card["model_type"] == "difficulty" and card["algorithm"] in dm.LABELS
        assert len(card["comparison"]) == 5 and sum(c["selected"] for c in card["comparison"]) == 1
        assert sum(c["is_baseline"] for c in card["comparison"]) == 2

        m = card["metrics"]
        assert m["cv_folds"] >= 2 and set(m["baseline_scores"]) == {"majority_class", "verb_rule"}
        assert m["improvement_over_best_baseline"] == round(
            m["selection_score"] - max(m["baseline_scores"].values()), 4)
        assert m["confusion_matrix"]["labels"] == ["easy", "medium", "hard"]
        assert sum(map(sum, m["confusion_matrix"]["matrix"])) == m["overall"]["samples"] == 42
        assert [p["level"] for p in m["per_level"]] == ["easy", "medium", "hard"]
        assert m["overall"]["within_one_level"] >= m["overall"]["accuracy"]

        data = m["training_data"]
        assert data["questions"] == 42 and data["gemini_labels"] == 42 and data["observed_labels"] == 0
        assert data["levels_left_out"] == [] and data["random_seed"] == settings.ml_random_seed
        assert "Gemini" in m["label_note"] and "imitate" in m["label_note"]

        row = db.rows("ml_models")[0]
        assert row["artifact_path"] in memory_store.files and row["is_active"] is True

    def test_reports_which_features_matter(self, api, db):
        seed_questions(db)

        card = train(api, force=True).json()

        importance = card["metrics"]["feature_importance"]
        if card["algorithm"] in ("decision_tree", "random_forest", "gradient_boosting"):
            assert importance and importance[0]["feature"] in dm.FEATURE_NAMES

    def test_saved_model_loads_and_predicts(self, api, db):
        seed_questions(db)
        train(api)

        row, bundle = registry.load_active_bundle(USER, COURSE, "difficulty")
        X = dm.feature_matrix([dm.extract_features("Define a stack."), dm.extract_features(PAPER_QUESTIONS["hard"][0])])
        out = dm.predict_levels(bundle["model"], X)

        assert [o["difficulty"] for o in out] == ["easy", "hard"] and bundle["feature_names"] == dm.FEATURE_NAMES

    def test_retraining_makes_version_2_and_one_active(self, api, db):
        seed_questions(db)
        first, second = train(api).json(), train(api).json()

        assert second["version"] == 2 and second["activated"] is True
        assert second["metrics"]["selection_score"] == first["metrics"]["selection_score"]
        assert [r["is_active"] for r in sorted(db.rows("ml_models"), key=lambda r: r["version"])] == [False, True]

    def test_worse_version_is_kept_out_unless_forced(self, api, db):
        seed_questions(db)
        train(api)
        db.rows("ml_models")[0]["metrics"]["selection_score"] = 2.0

        worse = train(api).json()
        assert worse["activated"] is False and "Kept version 1" in worse["activation_reason"]

        forced = train(api, force=True).json()
        assert forced["activated"] is True and {r["version"] for r in db.rows("ml_models") if r["is_active"]} == {3}

    def test_observed_results_replace_gemini_labels_after_enough_attempts(self, api, db):
        ids = seed_questions(db)
        easy_by_gemini = ids[0]                         # Gemini said easy
        add_attempts(db, easy_by_gemini, correct=1, wrong=3)         # students got 75% wrong -> hard
        add_attempts(db, ids[1], correct=1, wrong=1)                 # only 2 attempts: ignored

        rows, counts = ds.build_training_set(USER, COURSE)
        by_q = {r["question_id"]: r for r in rows}

        assert by_q[easy_by_gemini]["label"] == "hard" and by_q[easy_by_gemini]["source"] == "observed"
        assert by_q[easy_by_gemini]["wrong_rate"] == 0.75 and by_q[ids[1]]["source"] == "gemini"
        assert counts["observed_labels"] == 1 and counts["gemini_labels"] == 41

        note = train(api).json()["metrics"]["label_note"]
        assert "1 of 42 labels come from real student results" in note

    @pytest.mark.parametrize("correct,wrong,level", [(9, 1, "easy"), (2, 2, "medium"), (1, 4, "hard"), (0, 3, "hard")])
    def test_observed_level_thresholds(self, correct, wrong, level):
        assert ds.observed_level(wrong / (correct + wrong)) == level

    def test_not_enough_questions_returns_422_and_stores_nothing(self, api, db):
        seed_questions(db, per_level=5)                 # 15 questions < 20

        r = train(api)

        assert r.status_code == 422 and "Not enough practice questions" in r.json()["detail"]
        assert "command-verb rule" in r.json()["detail"] and db.rows("ml_models") == []

    def test_a_single_difficulty_level_cannot_be_trained(self, api, db):
        for text, _l in make_questions(10):
            db.seed("generated_questions", {"id": new_id(), "question_text": text, "difficulty": "medium",
                                            "status": "active", "user_id": USER, "course_id": COURSE})

        r = train(api)

        assert r.status_code == 422 and "30 medium" in r.json()["detail"]

    def test_levels_with_too_few_examples_are_left_out_and_reported(self, api, db):
        seed_questions(db, per_level=14)
        for i in range(2):                               # only 2 'hard' ... remove most hard ones
            pass
        hard = [q for q in db.rows("generated_questions") if q["difficulty"] == "hard"]
        db.tables["generated_questions"][:] = [q for q in db.rows("generated_questions")
                                               if q["difficulty"] != "hard"] + hard[:2]

        card = train(api).json()

        assert card["metrics"]["training_data"]["levels_left_out"] == [{"level": "hard", "examples": 2}]
        assert card["metrics"]["training_data"]["levels_trained"] == ["easy", "medium"]
        assert sum(map(sum, card["metrics"]["confusion_matrix"]["matrix"])) == 28

    def test_archived_and_foreign_questions_are_ignored(self, api, db):
        seed_questions(db)
        seed_questions(db, status="archived")
        seed_questions(db, user=OTHER_USER, course=OTHER_COURSE)

        assert train(api).json()["metrics"]["training_data"]["questions"] == 42

    def test_option_features_are_read_from_question_options(self, db):
        qid = new_id()
        db.seed("generated_questions", {"id": qid, "question_text": "Which is true?", "difficulty": "medium",
                                        "status": "active", "user_id": USER, "course_id": COURSE})
        for i, (text, ok) in enumerate([("A stack is LIFO", True), ("A stack is FIFO", False),
                                        ("A stack is LILO", False), ("A stack is FILO", False)], start=1):
            db.seed("question_options", {"question_id": qid, "option_text": text, "option_order": i, "is_correct": ok})

        rows, _c = ds.build_training_set(USER, COURSE)

        assert rows[0]["features"]["has_options"] == 1 and rows[0]["features"]["n_options"] == 4

    def test_requires_internal_key_and_valid_body(self, api, anon_api):
        assert anon_api.post(TRAIN, json={"user_id": USER}).status_code == 401
        assert api.post(TRAIN, json={}).status_code == 422
        assert api.post(TRAIN, json={"user_id": USER, "force": "maybe"}).status_code == 422


# ============================================================ status
class TestStatus:
    def test_before_training(self, api, db):
        seed_questions(db)

        body = api.get(STATUS, params={"user_id": USER}).json()

        assert body["using_fallback"] is True and body["active_model"] is None and body["history"] == []
        assert body["can_train"] is True and body["message"] is None
        assert body["training_data"]["levels_ready"] == ["easy", "medium", "hard"]
        assert body["training_data"]["level_counts"] == {"easy": 14, "medium": 14, "hard": 14}

    def test_explains_what_is_missing(self, api, db):
        seed_questions(db, per_level=4)

        body = api.get(STATUS, params={"user_id": USER}).json()

        assert body["can_train"] is False and "Generate more practice questions" in body["message"]

    def test_after_training(self, api, db):
        seed_questions(db)
        train(api); train(api)

        body = api.get(STATUS, params={"user_id": USER}).json()

        assert body["using_fallback"] is False and body["active_model"]["version"] == 2
        assert [h["version"] for h in body["history"]] == [2, 1]

    def test_user_id_is_required(self, api):
        assert api.get(STATUS).status_code == 422


# ============================================================ predict
class TestPredict:
    def test_without_a_model_it_uses_the_verb_rule_and_never_saves(self, api, db):
        q = add_paper_question(db, PAPER_QUESTIONS["hard"][0])

        body = predict(api, save=True, min_confidence=0).json()

        assert body["model_used"] == "fallback_rule" and body["model_version"] is None
        assert body["saved_count"] == 0 and paper_row(db, q)["predicted_difficulty"] is None
        assert body["items"][0]["predicted_difficulty"] == "hard"

    def test_predicts_easy_medium_and_hard_past_paper_questions(self, api, db):
        seed_questions(db)
        train(api)
        ids = {lvl: add_paper_question(db, text, marks=marks) for lvl, (text, marks) in PAPER_QUESTIONS.items()}

        body = predict(api).json()
        items = {i["question_id"]: i for i in body["items"]}

        assert body["model_used"] == "difficulty" and body["model_version"] == 1 and body["total"] == 3
        for level, qid in ids.items():
            assert items[qid]["predicted_difficulty"] == level
            assert sum(items[qid]["probabilities"].values()) == pytest.approx(1, abs=1e-2)
            assert items[qid]["confidence"] == max(items[qid]["probabilities"].values())
        assert items[ids["hard"]]["marks"] == 10 and body["by_level"] == {"easy": 1, "medium": 1, "hard": 1}

    def test_save_stores_confident_predictions_with_model_version(self, api, db):
        seed_questions(db)
        train(api)
        q = add_paper_question(db, PAPER_QUESTIONS["hard"][0])

        body = predict(api, save=True, min_confidence=0.0).json()
        row = paper_row(db, q)

        assert body["saved_count"] == 1 and body["items"][0]["saved"] is True
        assert row["predicted_difficulty"] == "hard" and 0 < row["difficulty_confidence"] <= 1
        assert row["difficulty_model_version"] == 1

    def test_low_confidence_predictions_are_flagged_not_saved(self, api, db):
        seed_questions(db)
        train(api)
        q = add_paper_question(db, PAPER_QUESTIONS["medium"][0])

        body = predict(api, save=True, min_confidence=1.0).json()

        assert body["needs_review_count"] == body["total"] and body["saved_count"] == 0
        assert paper_row(db, q)["predicted_difficulty"] is None

    def test_without_save_nothing_is_written(self, api, db):
        seed_questions(db)
        train(api)
        q = add_paper_question(db, PAPER_QUESTIONS["easy"][0])

        predict(api, min_confidence=0)

        assert paper_row(db, q)["predicted_difficulty"] is None

    def test_default_skips_questions_that_already_have_a_prediction(self, api, db):
        seed_questions(db)
        train(api)
        done = add_paper_question(db, "Define a heap.", predicted="easy")
        todo = add_paper_question(db, "Define a graph.")

        default = predict(api).json()
        by_id = predict(api, question_ids=[done]).json()

        assert [i["question_id"] for i in default["items"]] == [todo]
        assert by_id["items"][0]["previous_prediction"] == "easy"

    def test_works_for_pending_and_reviewed_questions_alike(self, api, db):
        seed_questions(db)
        train(api)
        a = add_paper_question(db, "Define a heap.", status="pending")
        b = add_paper_question(db, "Define a graph.", status="reviewed")

        body = predict(api, save=True, min_confidence=0).json()

        assert body["saved_count"] == 2 and all(paper_row(db, q)["predicted_difficulty"] for q in (a, b))

    def test_unknown_and_foreign_ids_are_reported(self, api, db):
        seed_questions(db)
        train(api)
        foreign = add_paper_question(db, "Not mine", user=OTHER_USER, course=OTHER_COURSE)
        ghost = new_id()

        body = predict(api, question_ids=[ghost, foreign]).json()

        assert body["total"] == 0 and set(body["not_found"]) == {ghost, foreign}

    def test_a_missing_model_file_falls_back_instead_of_failing(self, api, db, memory_store):
        seed_questions(db)
        train(api)
        memory_store.files.clear()
        add_paper_question(db, "Define a heap.")

        r = predict(api, save=True, min_confidence=0)

        assert r.status_code == 200 and r.json()["model_used"] == "fallback_rule" and r.json()["saved_count"] == 0

    def test_limit_caps_the_batch(self, api, db):
        seed_questions(db)
        train(api)
        for i in range(5):
            add_paper_question(db, f"Define concept number {i}.")

        assert predict(api, limit=2).json()["total"] == 2

    def test_validation(self, api, anon_api):
        assert anon_api.post(PREDICT, json={"user_id": USER}).status_code == 401
        assert predict(api, min_confidence=1.5).status_code == 422
        assert predict(api, limit=0).status_code == 422
        assert predict(api, limit=201).status_code == 422
        assert predict(api, question_ids=["nope"]).status_code == 422
        assert api.post(PREDICT, json={}).status_code == 422
