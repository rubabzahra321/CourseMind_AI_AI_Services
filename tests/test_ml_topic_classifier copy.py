"""ML step 1: topic classification (training, comparison, registry, suggestions)."""
import numpy as np
import pytest

from app.core.config import settings
from app.ml import registry, topic_classifier as tc
from app.ml.estimators import CosineCentroid, SoftmaxLinearSVC
from tests.conftest import COURSE, OTHER_COURSE, OTHER_USER, USER, new_id
from tests.ml_corpus import CORPUS, NEW_QUESTIONS

BASE = f"/internal/courses/{COURSE}/ml"
TRAIN = f"{BASE}/topic-classifier/train"
CLASSIFY = f"{BASE}/classify-questions"
STATUS = f"{BASE}/status"


# ============================================================ fixtures
@pytest.fixture(autouse=True)
def memory_store():
    store = registry.MemoryArtifactStore()
    registry.set_artifact_store(store)
    yield store
    registry.set_artifact_store(registry.SupabaseArtifactStore())


def seed_course(db, with_mappings=True, per_topic=8):
    """Three topics; every question hand-mapped to its topic."""
    topic_ids = {name: new_id() for name in CORPUS}

    db.seed("syllabus_topics", *[
        {"id": tid, "title": name, "description": None, "position": i,
         "parent_topic_id": None, "user_id": USER, "course_id": COURSE}
        for i, (name, tid) in enumerate(topic_ids.items(), start=1)])

    question_ids = {}

    for name, questions in CORPUS.items():
        for number, text in enumerate(questions[:per_topic], start=1):
            qid = new_id()
            question_ids[(name, number)] = qid
            db.seed("past_paper_questions", {
                "id": qid, "question_text": text, "question_number": str(number),
                "review_status": "reviewed", "user_id": USER, "course_id": COURSE})
            if with_mappings:
                db.seed("question_topic_mappings", {
                    "past_paper_question_id": qid, "topic_id": topic_ids[name],
                    "user_id": USER, "mapping_method": "manual"})

    return topic_ids, question_ids


def add_unmapped(db, text, status="pending", user=USER, course=COURSE):
    qid = new_id()
    db.seed("past_paper_questions", {
        "id": qid, "question_text": text, "question_number": "9",
        "review_status": status, "user_id": user, "course_id": course})
    return qid


def train(api, **extra):
    return api.post(TRAIN, json={"user_id": USER, **extra})


def classify(api, **extra):
    return api.post(CLASSIFY, json={"user_id": USER, **extra})


def mappings_for(db, qid):
    return [m for m in db.rows("question_topic_mappings") if m["past_paper_question_id"] == qid]


# ============================================================ estimators & core (no DB)
class TestEstimators:
    def test_cosine_centroid_predicts_closest_class(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = ["red apple fruit", "green apple fruit", "fast car engine", "car wheel engine"]
        X = TfidfVectorizer().fit_transform(texts)
        model = CosineCentroid().fit(X, np.array(["fruit", "fruit", "car", "car"]))

        assert list(model.predict(X)) == ["fruit", "fruit", "car", "car"]
        proba = model.predict_proba(X)
        assert proba.shape == (4, 2) and np.allclose(proba.sum(axis=1), 1)

    def test_softmax_svm_gives_valid_probabilities_for_two_and_three_classes(self):
        X = np.array([[0.0, 1], [0.1, 0.9], [1, 0], [0.9, 0.1], [0.5, 0.5], [0.45, 0.55]])
        two = SoftmaxLinearSVC(random_state=1).fit(X[:4], ["a", "a", "b", "b"])
        three = SoftmaxLinearSVC(random_state=1).fit(X, ["a", "a", "b", "b", "c", "c"])

        for model, columns in ((two, 2), (three, 3)):
            proba = model.predict_proba(X)
            assert proba.shape == (6, columns) and np.allclose(proba.sum(axis=1), 1)


class TestCore:
    def data(self):
        texts, labels, sources = [], [], []
        for name, questions in CORPUS.items():
            for q in questions:
                texts.append(q); labels.append(name); sources.append("question")
        return texts, labels, sources

    def test_compares_five_candidates_including_the_baseline(self):
        results, best, basis, classes = tc.evaluate_candidates(*self.data(), 3, 42, 10)

        assert set(results) == {"centroid_baseline", "logistic_regression", "linear_svm",
                                "naive_bayes", "knn"}
        assert results[tc.BASELINE]["is_baseline"] is True
        assert best in results and basis == "question_only"
        assert list(classes) == ["Graphs", "Sorting", "Trees"]
        for r in results.values():
            assert 0 <= r["selection_score"] <= 1

    def test_results_are_repeatable_with_the_same_seed(self):
        a = tc.evaluate_candidates(*self.data(), 3, 42, 10)
        b = tc.evaluate_candidates(*self.data(), 3, 42, 10)

        assert {n: r["selection_score"] for n, r in a[0].items()} == \
               {n: r["selection_score"] for n, r in b[0].items()}
        assert a[1] == b[1]

    def test_selection_uses_all_samples_when_too_few_questions(self):
        texts, labels, sources = self.data()
        sources = ["passage"] * len(sources)                   # no real exam questions at all

        _r, _best, basis, _c = tc.evaluate_candidates(texts, labels, sources, 3, 42, 10)

        assert basis == "all_samples"

    def test_trained_model_beats_chance_on_clear_topics(self):
        results, best, *_ = tc.evaluate_candidates(*self.data(), 3, 42, 10)

        assert results[best]["overall"]["accuracy"] > 0.6            # chance is 0.33

    def test_predict_top_k_returns_sorted_probabilities(self):
        texts, labels, _s = self.data()
        pipeline = tc.fit_final("logistic_regression", texts, labels, 3, 42)

        top = tc.predict_top_k(pipeline, [NEW_QUESTIONS["Graphs"]], k=3)[0]

        assert top[0][0] == "Graphs"
        assert [p for _t, p in top] == sorted((p for _t, p in top), reverse=True)
        assert sum(p for _t, p in top) <= 1.0 + 1e-9

    def test_keyword_fallback_needs_no_training(self):
        topics = [{"id": "g", "title": "Graphs", "description": "graph traversal search"},
                  {"id": "t", "title": "Trees", "description": None}]

        ranked = tc.keyword_fallback(topics, ["Explain depth first search on a graph"])[0]

        assert ranked and ranked[0][0] == "g"

    def test_keyword_fallback_handles_empty_input(self):
        assert tc.keyword_fallback([], ["x"]) == [[]]
        assert tc.keyword_fallback([{"id": "a", "title": "A"}], []) == []


# ============================================================ training
class TestTraining:
    def test_trains_compares_and_stores_a_model(self, api, db, memory_store):
        seed_course(db)

        r = train(api)
        assert r.status_code == 200
        card = r.json()

        assert card["version"] == 1 and card["is_active"] is True and card["activated"] is True
        assert card["model_type"] == "topic_classifier"
        assert card["algorithm"] in tc.LABELS

        comparison = card["comparison"]
        assert len(comparison) == 5 and sum(1 for c in comparison if c["selected"]) == 1
        assert any(c["is_baseline"] for c in comparison)
        assert comparison == sorted(comparison, key=lambda c: -c["selection_score"])

        m = card["metrics"]
        assert m["selection_basis"] == "question_only" and m["cv_folds"] >= 2
        assert m["improvement_over_baseline"] == round(m["selection_score"] - m["baseline_score"], 4)
        assert len(m["per_topic"]) == 3
        matrix = m["confusion_matrix"]
        assert matrix["labels"] == ["Graphs", "Sorting", "Trees"]
        assert np.array(matrix["matrix"]).shape == (3, 3)
        assert sum(map(sum, matrix["matrix"])) == m["overall"]["samples"]

        training = m["training_data"]
        assert training["labelled_questions"] == 24 and training["syllabus_topics"] == 3
        assert training["topics_trained"] == 3 and training["topics_left_out"] == []
        assert training["random_seed"] == settings.ml_random_seed

        row = db.rows("ml_models")[0]
        assert row["artifact_path"] in memory_store.files and row["is_active"] is True

    def test_saved_model_can_be_loaded_and_used(self, api, db):
        seed_course(db)
        train(api)

        row, bundle = registry.load_active_bundle(USER, COURSE, "topic_classifier")

        assert row["version"] == 1 and bundle["algorithm"] == row["algorithm"]
        top = tc.predict_top_k(bundle["pipeline"], [NEW_QUESTIONS["Sorting"]], 1)[0][0]
        assert bundle["titles"][top[0]] == "Sorting"

    def test_retraining_creates_version_2_and_keeps_one_active(self, api, db):
        seed_course(db)
        first = train(api).json()
        second = train(api).json()

        assert second["version"] == 2 and second["activated"] is True
        assert second["metrics"]["selection_score"] == first["metrics"]["selection_score"]   # repeatable
        rows = db.rows("ml_models")
        assert [r["is_active"] for r in sorted(rows, key=lambda r: r["version"])] == [False, True]

    def test_a_worse_version_does_not_replace_the_active_one(self, api, db):
        seed_course(db)
        train(api)
        db.rows("ml_models")[0]["metrics"]["selection_score"] = 2.0     # pretend v1 was excellent

        r = train(api).json()

        assert r["version"] == 2 and r["activated"] is False and r["is_active"] is False
        assert "Kept version 1" in r["activation_reason"]
        assert {x["version"] for x in db.rows("ml_models") if x["is_active"]} == {1}

    def test_force_activates_the_new_version_anyway(self, api, db):
        seed_course(db)
        train(api)
        db.rows("ml_models")[0]["metrics"]["selection_score"] = 2.0

        r = train(api, force=True).json()

        assert r["activated"] is True and r["activation_reason"] == "Activated by request."
        assert {x["version"] for x in db.rows("ml_models") if x["is_active"]} == {2}

    def test_not_enough_data_returns_422_and_stores_nothing(self, api, db):
        seed_course(db, with_mappings=False)           # topics and questions, but no labels

        r = train(api)

        assert r.status_code == 422
        assert "Not enough labelled data" in r.json()["detail"] and "keyword matching" in r.json()["detail"]
        assert db.rows("ml_models") == []

    def test_topics_with_too_few_examples_are_left_out_and_reported(self, api, db):
        topic_ids, _q = seed_course(db)
        db.seed("syllabus_topics", {"id": new_id(), "title": "Hashing", "description": None, "position": 9,
                                    "parent_topic_id": None, "user_id": USER, "course_id": COURSE})

        card = train(api).json()

        left_out = card["metrics"]["training_data"]["topics_left_out"]
        assert [t["topic_title"] for t in left_out] == ["Hashing"] and left_out[0]["examples"] == 1
        assert card["metrics"]["training_data"]["topics_trained"] == 3

    def test_only_hand_made_labels_are_used(self, api, db):
        topic_ids, q = seed_course(db)
        stray = add_unmapped(db, "Explain AVL rotations")
        db.seed("question_topic_mappings", {"past_paper_question_id": stray, "topic_id": topic_ids["Trees"],
                                            "user_id": USER, "mapping_method": "ai_suggested"})

        card = train(api).json()

        assert card["metrics"]["training_data"]["labelled_questions"] == 24      # AI guess ignored

    def test_ai_linked_passages_are_used_only_when_asked(self, api, db):
        topic_ids, _q = seed_course(db)
        doc = new_id()
        for name in ("Trees", "Graphs"):
            for i in range(2):
                cid = new_id()
                db.seed("document_chunks", {"id": cid, "document_id": doc, "course_id": COURSE, "user_id": USER,
                                            "chunk_index": i, "content": f"{name} lecture passage {i}"})
                db.seed("topic_chunk_links", {"topic_id": topic_ids[name], "chunk_id": cid, "document_id": doc,
                                              "course_id": COURSE, "user_id": USER, "method": "ai", "similarity": .8})

        default = train(api).json()["metrics"]["training_data"]
        with_ai = train(api, include_ai_links=True).json()["metrics"]["training_data"]

        assert default["linked_passages"] == 0 and default["includes_ai_links"] is False
        assert with_ai["linked_passages"] == 4 and with_ai["includes_ai_links"] is True

    def test_manual_passage_links_add_training_examples(self, api, db):
        topic_ids, _q = seed_course(db)
        doc = new_id()
        cid = new_id()
        db.seed("document_chunks", {"id": cid, "document_id": doc, "course_id": COURSE, "user_id": USER,
                                    "chunk_index": 0, "content": "Dijkstra and graph search lecture"})
        db.seed("topic_chunk_links", {"topic_id": topic_ids["Graphs"], "chunk_id": cid, "document_id": doc,
                                      "course_id": COURSE, "user_id": USER, "method": "manual", "similarity": 1})

        assert train(api).json()["metrics"]["training_data"]["linked_passages"] == 1

    def test_other_users_and_courses_are_never_used(self, api, db):
        seed_course(db)
        db.seed("syllabus_topics", {"id": new_id(), "title": "Secret", "position": 1, "parent_topic_id": None,
                                    "user_id": OTHER_USER, "course_id": OTHER_COURSE})
        add_unmapped(db, "Other user's question", user=OTHER_USER, course=OTHER_COURSE)

        card = train(api).json()

        assert "Secret" not in str(card["metrics"]["training_data"]["samples_per_topic"])

    def test_requires_internal_key_and_valid_body(self, api, anon_api):
        assert anon_api.post(TRAIN, json={"user_id": USER}).status_code == 401
        assert api.post(TRAIN, json={}).status_code == 422
        assert api.post(TRAIN, json={"user_id": USER, "force": "maybe"}).status_code == 422


# ============================================================ status
class TestStatus:
    def test_before_training_reports_fallback_and_data_readiness(self, api, db):
        seed_course(db)

        body = api.get(STATUS, params={"user_id": USER}).json()

        assert body["using_fallback"] is True and body["active_model"] is None and body["history"] == []
        assert body["can_train"] is True and body["message"] is None
        assert body["training_data"]["labelled_questions"] == 24
        assert body["training_data"]["topics_ready"] == 3

    def test_explains_what_is_missing(self, api, db):
        seed_course(db, with_mappings=False)

        body = api.get(STATUS, params={"user_id": USER}).json()

        assert body["can_train"] is False and "Map more past-paper questions" in body["message"]

    def test_after_training_shows_the_active_model_and_history(self, api, db):
        seed_course(db)
        train(api); train(api)

        body = api.get(STATUS, params={"user_id": USER}).json()

        assert body["using_fallback"] is False
        assert body["active_model"]["version"] == 2 and body["active_model"]["is_active"] is True
        assert [h["version"] for h in body["history"]] == [2, 1]
        assert all(h["selection_score"] is not None for h in body["history"])

    def test_user_id_is_required(self, api):
        assert api.get(STATUS).status_code == 422


# ============================================================ classify
class TestClassify:
    def test_without_a_model_it_uses_keyword_matching_and_never_saves(self, api, db):
        seed_course(db, with_mappings=False)
        qid = add_unmapped(db, NEW_QUESTIONS["Graphs"])

        body = classify(api, save=True).json()

        assert body["model_used"] == "fallback_keyword" and body["model_version"] is None
        assert body["saved_count"] == 0 and mappings_for(db, qid) == []
        assert body["total"] >= 1

    def test_suggests_the_right_topic_with_probabilities(self, api, db):
        seed_course(db)
        train(api)
        asked = {name: add_unmapped(db, text) for name, text in NEW_QUESTIONS.items()}

        body = classify(api).json()
        by_q = {i["question_id"]: i for i in body["items"]}

        assert body["model_used"] == "topic_classifier" and body["model_version"] == 1
        assert body["total"] == 3                                   # only the unmapped ones
        for name, qid in asked.items():
            item = by_q[qid]
            assert item["suggestions"][0]["topic_title"] == name
            probs = [s["probability"] for s in item["suggestions"]]
            assert probs == sorted(probs, reverse=True) and len(probs) == 3
            assert item["already_mapped"] is False and item["saved"] is False

    def test_save_stores_confident_suggestions_for_pending_questions_only(self, api, db):
        seed_course(db)
        train(api)
        pending = add_unmapped(db, NEW_QUESTIONS["Graphs"], status="pending")
        reviewed = add_unmapped(db, NEW_QUESTIONS["Trees"], status="reviewed")

        body = classify(api, save=True, min_confidence=0.0).json()
        items = {i["question_id"]: i for i in body["items"]}

        assert items[pending]["saved"] is True and body["saved_count"] == 1
        saved = mappings_for(db, pending)
        assert len(saved) == 1 and saved[0]["mapping_method"] == "ai_suggested"
        assert 0 < saved[0]["confidence"] <= 1

        # reviewed questions are never touched, so AI guesses cannot enter exam statistics
        assert items[reviewed]["saved"] is False and mappings_for(db, reviewed) == []
        assert items[reviewed]["suggestions"]                          # but the suggestion is shown

    def test_low_confidence_suggestions_need_review_and_are_not_saved(self, api, db):
        seed_course(db)
        train(api)
        qid = add_unmapped(db, NEW_QUESTIONS["Sorting"])

        body = classify(api, save=True, min_confidence=1.0).json()

        assert body["needs_review_count"] == body["total"] and body["saved_count"] == 0
        assert mappings_for(db, qid) == []

    def test_an_unrelated_question_gets_low_confidence(self, api, db):
        seed_course(db)
        train(api)
        add_unmapped(db, "Describe the causes of the French Revolution.")

        item = classify(api).json()["items"][0]

        assert item["suggestions"][0]["probability"] < 0.6

    def test_already_mapped_questions_are_skipped_unless_asked_for_by_id(self, api, db):
        topic_ids, q = seed_course(db)
        train(api)
        mapped_q = q[("Trees", 1)]

        default = classify(api).json()
        by_id = classify(api, question_ids=[mapped_q]).json()

        assert mapped_q not in {i["question_id"] for i in default["items"]}
        assert by_id["total"] == 1 and by_id["items"][0]["already_mapped"] is True
        assert by_id["saved_count"] == 0

    def test_never_overwrites_an_existing_mapping(self, api, db):
        topic_ids, q = seed_course(db)
        train(api)
        mapped_q = q[("Graphs", 2)]
        before = list(mappings_for(db, mapped_q))

        classify(api, question_ids=[mapped_q], save=True, min_confidence=0.0)

        assert mappings_for(db, mapped_q) == before

    def test_unknown_and_foreign_question_ids_are_reported(self, api, db):
        seed_course(db)
        train(api)
        foreign = add_unmapped(db, "Someone else's question", user=OTHER_USER, course=OTHER_COURSE)
        ghost = new_id()

        body = classify(api, question_ids=[ghost, foreign]).json()

        assert body["total"] == 0 and set(body["not_found"]) == {ghost, foreign}

    def test_deleted_topics_are_not_suggested(self, api, db):
        topic_ids, _q = seed_course(db)
        train(api)
        add_unmapped(db, NEW_QUESTIONS["Graphs"])
        db.tables["syllabus_topics"][:] = [t for t in db.rows("syllabus_topics") if t["title"] != "Graphs"]

        item = classify(api).json()["items"][0]

        assert "Graphs" not in {s["topic_title"] for s in item["suggestions"]}

    def test_a_missing_model_file_falls_back_instead_of_failing(self, api, db, memory_store):
        seed_course(db)
        train(api)
        memory_store.files.clear()
        add_unmapped(db, NEW_QUESTIONS["Graphs"])

        r = classify(api, save=True)

        assert r.status_code == 200 and r.json()["model_used"] == "fallback_keyword"
        assert r.json()["saved_count"] == 0

    def test_limit_caps_the_batch(self, api, db):
        seed_course(db)
        train(api)
        for i in range(5):
            add_unmapped(db, f"Explain graph search number {i}")

        assert classify(api, limit=2).json()["total"] == 2

    def test_validation(self, api, anon_api):
        assert anon_api.post(CLASSIFY, json={"user_id": USER}).status_code == 401
        assert classify(api, min_confidence=1.5).status_code == 422
        assert classify(api, limit=0).status_code == 422
        assert classify(api, limit=201).status_code == 422
        assert classify(api, question_ids=["nope"]).status_code == 422
        assert api.post(CLASSIFY, json={}).status_code == 422
