"""
Performance model orchestration.

  * TRAINING is pooled: attempts from all students are combined (anonymised -
    only numbers are kept, the model file holds no names, ids or question text)
    and one global model is stored.
  * READINESS is per student: the student's own history on each topic goes in,
    the probability of answering the next question correctly comes out.
"""

import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

from app.core.config import settings
from app.db.client import get_db
from app.ml import performance as pm
from app.ml import registry
from app.repositories import ml_repository
from app.services.insights_service import _all_rows, _topics
from app.services.ml_service import InsufficientTrainingData


MODEL_TYPE = "performance"
WEAKEST_COUNT = 5


def _now():
    """Current time in seconds (a function so tests can fix it)."""

    return time.time()


def _to_seconds(value):
    """ISO string or datetime -> seconds since 1970 (UTC)."""

    if isinstance(value, datetime):
        moment = value
    else:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)

    return moment.timestamp()


def _iso(seconds):
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()


def _attempt(row, question):
    return {
        "t": _to_seconds(row["answered_at"]),
        "correct": 1 if row["is_correct"] else 0,
        "rt": row.get("response_time_seconds"),
        "difficulty": pm.DIFFICULTY_INDEX.get(question.get("difficulty"), 1),
    }


# ---------------------------------------------------------------- pooled data
def load_pooled_histories():
    """
    Every student's attempts, grouped into one history per (student, topic).
    Ids are used only to group them in memory; nothing identifying is kept.
    """

    questions = {
        str(q["id"]): q
        for q in _all_rows(
            lambda: get_db().table("generated_questions")
            .select("id, topic_id, difficulty")
        )
        if q.get("topic_id")
    }

    groups = defaultdict(list)
    attempts = 0

    for row in _all_rows(
        lambda: get_db().table("practice_attempts")
        .select("user_id, question_id, is_correct, response_time_seconds, answered_at")
    ):
        question = questions.get(str(row["question_id"]))

        if not question or not row.get("answered_at"):
            continue

        groups[(str(row["user_id"]), str(question["topic_id"]))].append(
            _attempt(row, question))
        attempts += 1

    histories = list(groups.values())

    return histories, {
        "students": len({user for user, _topic in groups}),
        "topic_histories": len(histories),
        "attempts": attempts,
    }


def _training_arrays(histories):
    rows, y, t = pm.build_samples(
        histories,
        settings.ml_performance_min_history,
        settings.ml_performance_ewma_alpha,
    )

    return pm.feature_matrix(rows), y, t


def _data_problem(counts, samples, y, t):
    """Why training is not possible yet, or None."""

    if samples < settings.ml_performance_min_samples:
        return (
            f"Found {counts['attempts']} practice answers from "
            f"{counts['students']} student(s), which gives {samples} training "
            f"examples; at least {settings.ml_performance_min_samples} are needed."
        )

    train, test = pm.temporal_split(t, settings.ml_performance_test_fraction)

    if len(test) < settings.ml_performance_min_test:
        return (
            f"Only {len(test)} examples fall in the later test period; at "
            f"least {settings.ml_performance_min_test} are needed."
        )

    if len(set(y[train])) < 2 or len(set(y[test])) < 2:
        return (
            "The earlier or the later period contains only correct (or only "
            "wrong) answers, so there is nothing to learn or test."
        )

    return None


def _insufficient_message(problem):
    return (
        "Not enough practice history to train a performance model yet. "
        f"{problem} Keep practising (answers from many students are pooled) "
        "and train again. Until then readiness is estimated from each "
        "student's recent accuracy."
    )


# ---------------------------------------------------------------- cards
def _card(row, activated=None, reason=None):
    card = {
        "id": row["id"],
        "model_type": row["model_type"],
        "scope": "global",
        "algorithm": row["algorithm"],
        "algorithm_label": pm.LABELS.get(row["algorithm"], row["algorithm"]),
        "version": row["version"],
        "is_active": bool(row.get("is_active")),
        "n_training_samples": row["n_training_samples"],
        "trained_at": row.get("trained_at"),
        "metrics": row.get("metrics") or {},
        "comparison": row.get("comparison") or [],
    }

    if activated is not None:
        card["activated"] = activated
        card["activation_reason"] = reason

    return card


# ---------------------------------------------------------------- train
def train_performance_model(force=False):
    histories, counts = load_pooled_histories()
    X, y, t = _training_arrays(histories)

    problem = _data_problem(counts, len(y), y, t)

    if problem:
        raise InsufficientTrainingData(_insufficient_message(problem))

    results, best, split = pm.evaluate_candidates(
        X, y, t,
        settings.ml_performance_test_fraction,
        settings.ml_random_seed,
        settings.ml_performance_ensemble_size,
    )

    metrics = pm.detailed_metrics(results, best, split)
    model = pm.fit_final(
        best, X, y, settings.ml_random_seed, settings.ml_performance_ensemble_size)

    metrics["feature_importance"] = pm.feature_importance(model)
    metrics["split"]["split_time"] = _iso(split["split_time"])
    metrics["training_data"] = {
        **counts,
        "samples_used": int(len(y)),
        "min_history": settings.ml_performance_min_history,
        "test_fraction": settings.ml_performance_test_fraction,
        "period_start": _iso(float(t.min())),
        "period_end": _iso(float(t.max())),
        "random_seed": settings.ml_random_seed,
    }
    metrics["privacy_note"] = (
        "Trained on answers pooled from several students. Only numbers "
        "(accuracy, timing, difficulty) are used; the stored model contains "
        "no names, ids or question text."
    )

    bundle = {
        "model": model,
        "algorithm": best,
        "feature_names": pm.FEATURE_NAMES,
    }

    row = registry.save_version(
        None, None, MODEL_TYPE, best, int(len(y)),
        metrics, pm.comparison_table(results, best), bundle,
    )

    activated, reason = registry.activate_if_better(row, force=force)
    row["is_active"] = activated

    return _card(row, activated, reason)


# ---------------------------------------------------------------- status
def get_status():
    active = ml_repository.get_active(None, None, MODEL_TYPE)
    history = ml_repository.list_models(None, None, MODEL_TYPE)

    histories, counts = load_pooled_histories()
    X, y, t = _training_arrays(histories)
    problem = _data_problem(counts, len(y), y, t)

    return {
        "model_type": MODEL_TYPE,
        "scope": "global",
        "active_model": _card(active) if active else None,
        "using_fallback": active is None,
        "history": [
            {
                "version": row["version"],
                "algorithm": row["algorithm"],
                "algorithm_label": pm.LABELS.get(row["algorithm"], row["algorithm"]),
                "selection_score": (row.get("metrics") or {}).get("selection_score"),
                "n_training_samples": row["n_training_samples"],
                "is_active": bool(row.get("is_active")),
                "trained_at": row.get("trained_at"),
            }
            for row in history
        ],
        "training_data": {
            **counts,
            "training_examples": int(len(y)),
            "minimum_examples": settings.ml_performance_min_samples,
            "minimum_test_examples": settings.ml_performance_min_test,
        },
        "can_train": problem is None,
        "message": None if problem is None else _insufficient_message(problem),
    }


# ---------------------------------------------------------------- readiness
def _student_histories(user_id, course_id):
    """topic_id -> this student's attempts on it, oldest first."""

    questions = {
        str(q["id"]): q
        for q in _all_rows(
            lambda: get_db().table("generated_questions")
            .select("id, topic_id, difficulty")
            .eq("user_id", str(user_id))
            .eq("course_id", str(course_id))
        )
        if q.get("topic_id")
    }

    histories = defaultdict(list)

    for row in _all_rows(
        lambda: get_db().table("practice_attempts")
        .select("question_id, is_correct, response_time_seconds, answered_at")
        .eq("user_id", str(user_id))
    ):
        question = questions.get(str(row["question_id"]))

        if question and row.get("answered_at"):
            histories[str(question["topic_id"])].append(_attempt(row, question))

    return {
        topic: sorted(items, key=lambda a: a["t"])
        for topic, items in histories.items()
    }


def _readiness_item(topic, history, model, now, q_difficulty):
    n = len(history)
    base = {
        "topic_id": str(topic["id"]),
        "topic_title": topic["title"],
        "parent_topic_id": topic.get("parent_topic_id"),
        "attempts": n,
    }

    empty_trend = {"label": "not_enough_data", "slope": None, "attempts_used": 0}

    if n == 0:
        return {
            **base, "status": "not_assessed", "readiness": None, "band": None,
            "accuracy": None, "recent_accuracy": None,
            "days_since_last": None, "trend": empty_trend,
        }

    features = pm.history_features(
        history, now, q_difficulty, settings.ml_performance_ewma_alpha)

    if model is not None:
        readiness = float(pm.predict_probability(model, pm.feature_matrix([features]))[0])
    else:
        readiness = features["ewma"]                 # fallback: recent accuracy

    results = [a["correct"] for a in history]

    return {
        **base,
        "status": "assessed" if n >= 3 else "few_attempts",
        "readiness": round(readiness, 4),
        "band": pm.band(readiness),
        "accuracy": round(sum(results) / n, 4),
        "recent_accuracy": round(sum(results[-5:]) / len(results[-5:]), 4),
        "days_since_last": round(
            max(0.0, (now - history[-1]["t"]) / pm.SECONDS_PER_DAY), 1),
        "trend": pm.trend(
            history,
            settings.ml_performance_trend_window,
            settings.ml_performance_trend_threshold,
            settings.ml_performance_trend_min_attempts,
        ),
    }


def student_readiness(user_id, course_id, difficulty="medium"):
    """
    For every syllabus topic: predicted chance of answering the next question
    correctly, plus accuracy, how long ago it was practised, and the trend.
    Topics never practised are reported as not assessed (nothing is invented).
    """

    topics = _topics(user_id, course_id)
    histories = _student_histories(user_id, course_id)
    now = _now()

    row, bundle = registry.load_active_bundle(None, None, MODEL_TYPE)
    model = None

    if bundle and bundle.get("feature_names") == pm.FEATURE_NAMES:
        model = bundle["model"]

    q_difficulty = pm.DIFFICULTY_INDEX[difficulty]

    items = [
        _readiness_item(t, histories.get(str(t["id"]), []), model, now, q_difficulty)
        for t in topics
    ]

    assessed = [i for i in items if i["readiness"] is not None]
    weakest = sorted(assessed, key=lambda i: i["readiness"])[:WEAKEST_COUNT]

    return {
        "model_used": MODEL_TYPE if model is not None else "fallback_recent_accuracy",
        "model_version": row["version"] if (row and model is not None) else None,
        "algorithm": row["algorithm"] if (row and model is not None) else None,
        "model_skill": (
            (row.get("metrics") or {}).get("selection_score")
            if (row and model is not None) else None),
        "difficulty": difficulty,
        "as_of": _iso(now),
        "summary": {
            "topics": len(items),
            "assessed": len(assessed),
            "not_assessed": len(items) - len(assessed),
            "by_band": dict(Counter(i["band"] for i in assessed)),
            "by_trend": dict(Counter(
                i["trend"]["label"] for i in assessed)),
        },
        "weakest": [
            {"topic_id": i["topic_id"], "topic_title": i["topic_title"],
             "readiness": i["readiness"], "band": i["band"],
             "trend": i["trend"]["label"]}
            for i in weakest
        ],
        "topics": items,
    }
