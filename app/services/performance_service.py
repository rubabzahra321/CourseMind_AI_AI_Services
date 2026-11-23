"""
Performance model orchestration.

Training is pooled across students. Student/topic identifiers are used only
temporarily to group histories in memory; only numeric features and outcomes
are passed to the model and saved in the model bundle.

Readiness is calculated separately for each student's topics.
"""

import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

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
    """Current time in seconds; kept separate so tests can mock it."""
    return time.time()


def _to_seconds(value):
    """Convert an ISO timestamp or datetime to UTC epoch seconds."""
    if isinstance(value, datetime):
        moment = value
    else:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)

    return moment.timestamp()


def _iso(seconds):
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc).isoformat()


def _attempt(row, difficulty):
    """Convert a database attempt into the numeric history format used by pm."""
    return {
        "t": _to_seconds(row["answered_at"]),
        "correct": 1 if row["is_correct"] else 0,
        "rt": row.get("response_time_seconds"),
        "difficulty": pm.DIFFICULTY_INDEX.get(difficulty, 1),
    }


def _load_histories(user_id=None, course_id=None):
    """
    Load generated-question and manually mapped past-paper attempts.

    The returned dictionary is keyed by (user_id, topic_id). Identifiers are
    used only for in-memory grouping and are not included in model features.
    Past-paper attempts are included only when a human has manually mapped
    the question to a syllabus topic.
    """
    generated_questions = _all_rows(
        lambda: get_db().table("generated_questions")
        .select("id, user_id, course_id, topic_id, difficulty")
    )
    generated_by_id = {
        str(question["id"]): question
        for question in generated_questions
        if question.get("id") and question.get("topic_id")
    }

    paper_questions = _all_rows(
        lambda: get_db().table("past_paper_questions")
        .select("id, user_id, course_id, predicted_difficulty")
    )
    paper_by_id = {
        str(question["id"]): question
        for question in paper_questions
        if question.get("id")
    }

    manual_mappings = _all_rows(
        lambda: get_db().table("question_topic_mappings")
        .select("past_paper_question_id, topic_id, user_id")
        .eq("mapping_method", "manual")
    )
    paper_topics = defaultdict(list)
    for mapping in manual_mappings:
        paper_id = mapping.get("past_paper_question_id")
        topic_id = mapping.get("topic_id")
        if paper_id and topic_id:
            paper_topics[str(paper_id)].append(str(topic_id))

    sessions = _all_rows(
        lambda: get_db().table("practice_sessions")
        .select("id, course_id, user_id")
    )
    session_by_id = {
        str(session["id"]): session
        for session in sessions
        if session.get("id")
    }

    grouped = defaultdict(list)
    included_attempts = 0

    attempts = _all_rows(
        lambda: get_db().table("practice_attempts").select(
            "user_id, session_id, question_id, past_paper_question_id, "
            "is_correct, response_time_seconds, answered_at"
        )
    )

    for attempt in attempts:
        if attempt.get("is_correct") is None or not attempt.get("answered_at"):
            continue

        owner_id = str(attempt.get("user_id") or "")
        if not owner_id:
            continue
        if user_id is not None and owner_id != str(user_id):
            continue

        question_id = attempt.get("question_id")
        generated = (
            generated_by_id.get(str(question_id)) if question_id else None
        )

        if generated:
            # Generated question: its topic and course are already known.
            if str(generated.get("user_id")) != owner_id:
                continue
            topic_ids = [str(generated["topic_id"])]
            attempt_course_id = str(generated.get("course_id") or "")
            difficulty = generated.get("difficulty")
        else:
            # Past-paper question: require a manually reviewed topic mapping.
            paper_id = attempt.get("past_paper_question_id")
            paper = paper_by_id.get(str(paper_id)) if paper_id else None
            if not paper:
                continue

            session = session_by_id.get(str(attempt.get("session_id")))
            attempt_course_id = str(
                (session or {}).get("course_id")
                or paper.get("course_id")
                or ""
            )

            paper_owner = paper.get("user_id")
            if paper_owner and str(paper_owner) != owner_id:
                continue

            topic_ids = paper_topics.get(str(paper_id), [])
            difficulty = paper.get("predicted_difficulty")

        if course_id is not None and attempt_course_id != str(course_id):
            continue

        for topic_id in topic_ids:
            grouped[(owner_id, topic_id)].append(
                _attempt(attempt, difficulty)
            )
            included_attempts += 1

    for history in grouped.values():
        history.sort(key=lambda item: item["t"])

    counts = {
        "students": len({owner for owner, _topic in grouped}),
        "topic_histories": len(grouped),
        "attempts": included_attempts,
    }
    return grouped, counts


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def load_pooled_histories():
    """Return all students' histories, grouped by student and topic."""
    return _load_histories()


def _training_arrays(histories):
    rows, labels, timestamps = pm.build_samples(
        histories,
        settings.ml_performance_min_history,
        settings.ml_performance_ewma_alpha,
    )
    return pm.feature_matrix(rows), labels, timestamps


def _data_problem(counts, samples, labels, timestamps):
    """Return an explanation if the available data cannot support training."""
    if samples < settings.ml_performance_min_samples:
        return (
            f"Found {counts['attempts']} eligible practice answers from "
            f"{counts['students']} student(s), producing {samples} training "
            f"examples; at least {settings.ml_performance_min_samples} are needed."
        )

    train_indices, test_indices = pm.temporal_split(
        timestamps, settings.ml_performance_test_fraction
    )

    if len(test_indices) < settings.ml_performance_min_test:
        return (
            f"Only {len(test_indices)} examples fall in the later test period; "
            f"at least {settings.ml_performance_min_test} are needed."
        )

    if (
        len(set(labels[train_indices])) < 2
        or len(set(labels[test_indices])) < 2
    ):
        return (
            "The earlier training period or later test period contains only "
            "correct answers or only incorrect answers. Both outcomes are "
            "needed for training and evaluation."
        )

    return None


def _insufficient_message(problem):
    return (
        "Not enough practice history to train the performance model yet. "
        f"{problem} Keep practising; eligible answers from multiple students "
        "are pooled. Until enough data is available, readiness uses each "
        "student's exponentially weighted recent accuracy."
    )


# ---------------------------------------------------------------------------
# Model cards
# ---------------------------------------------------------------------------

def _card(row, activated=None, reason=None):
    card = {
        "id": row["id"],
        "model_type": row["model_type"],
        "scope": "global_anonymised",
        "algorithm": row["algorithm"],
        "algorithm_label": pm.LABELS.get(
            row["algorithm"], row["algorithm"]
        ),
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_performance_model(force=False):
    histories, counts = load_pooled_histories()
    features, labels, timestamps = _training_arrays(histories)

    problem = _data_problem(
        counts, len(labels), labels, timestamps
    )
    if problem:
        raise InsufficientTrainingData(_insufficient_message(problem))

    results, best, split = pm.evaluate_candidates(
        features,
        labels,
        timestamps,
        settings.ml_performance_test_fraction,
        settings.ml_random_seed,
        settings.ml_performance_ensemble_size,
    )

    metrics = pm.detailed_metrics(results, best, split)
    model = pm.fit_final(
        best,
        features,
        labels,
        settings.ml_random_seed,
        settings.ml_performance_ensemble_size,
    )

    metrics["feature_importance"] = pm.feature_importance(model)
    metrics["split"]["split_time"] = _iso(split["split_time"])
    metrics["training_data"] = {
        **counts,
        "samples_used": int(len(labels)),
        "min_history": settings.ml_performance_min_history,
        "test_fraction": settings.ml_performance_test_fraction,
        "period_start": _iso(timestamps.min()),
        "period_end": _iso(timestamps.max()),
        "random_seed": settings.ml_random_seed,
        "source": "pooled_anonymised_practice_attempts",
        "question_text_used": False,
        "personal_fields_in_features": False,
    }
    metrics["privacy_note"] = (
        "Training uses pooled numeric practice-history features and answer "
        "outcomes. Student and question identifiers are used only for "
        "in-memory grouping; names, identifiers and question text are not "
        "included in the model features or model bundle."
    )

    bundle = {
        "model": model,
        "algorithm": best,
        "feature_names": pm.FEATURE_NAMES,
    }

    row = registry.save_version(
        None,
        None,
        MODEL_TYPE,
        best,
        int(len(labels)),
        metrics,
        pm.comparison_table(results, best),
        bundle,
    )

    activated, reason = registry.activate_if_better(
        row, force=force
    )
    row["is_active"] = activated

    return _card(row, activated, reason)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def get_status():
    active = ml_repository.get_active(None, None, MODEL_TYPE)
    history = ml_repository.list_models(None, None, MODEL_TYPE)

    pooled_histories, counts = load_pooled_histories()
    _features, labels, timestamps = _training_arrays(pooled_histories)
    problem = _data_problem(
        counts, len(labels), labels, timestamps
    )

    return {
        "model_type": MODEL_TYPE,
        "scope": "global_anonymised",
        "active_model": _card(active) if active else None,
        "using_fallback": active is None,
        "history": [
            {
                "version": row["version"],
                "algorithm": row["algorithm"],
                "algorithm_label": pm.LABELS.get(
                    row["algorithm"], row["algorithm"]
                ),
                "selection_score": (
                    (row.get("metrics") or {}).get("selection_score")
                ),
                "n_training_samples": row["n_training_samples"],
                "is_active": bool(row.get("is_active")),
                "trained_at": row.get("trained_at"),
            }
            for row in history
        ],
        "training_data": {
            **counts,
            "training_examples": int(len(labels)),
            "minimum_examples": settings.ml_performance_min_samples,
            "minimum_test_examples": settings.ml_performance_min_test,
        },
        "can_train": problem is None,
        "message": (
            None if problem is None else _insufficient_message(problem)
        ),
    }


# ---------------------------------------------------------------------------
# Per-student readiness
# ---------------------------------------------------------------------------

def _student_histories(user_id, course_id):
    """Return this student's topic histories, oldest attempt first."""
    grouped, _counts = _load_histories(
        user_id=user_id,
        course_id=course_id,
    )

    histories = {}
    for (owner_id, topic_id), history in grouped.items():
        if owner_id == str(user_id):
            histories[topic_id] = history

    return histories


def _readiness_item(topic, history, model, now, question_difficulty):
    count = len(history)
    base = {
        "topic_id": str(topic["id"]),
        "topic_title": topic["title"],
        "parent_topic_id": topic.get("parent_topic_id"),
        "attempts": count,
    }

    empty_trend = {
        "label": "not_enough_data",
        "slope": None,
        "attempts_used": 0,
    }

    if count == 0:
        return {
            **base,
            "status": "not_assessed",
            "readiness": None,
            "band": None,
            "accuracy": None,
            "recent_accuracy": None,
            "days_since_last": None,
            "trend": empty_trend,
        }

    features = pm.history_features(
        history,
        now,
        question_difficulty,
        settings.ml_performance_ewma_alpha,
    )

    if model is not None:
        readiness = float(
            pm.predict_probability(
                model, pm.feature_matrix([features])
            )[0]
        )
    else:
        readiness = float(features["ewma"])

    outcomes = [attempt["correct"] for attempt in history]
    recent_outcomes = outcomes[-5:]

    return {
        **base,
        "status": "assessed" if count >= 3 else "few_attempts",
        "readiness": round(readiness, 4),
        "band": pm.band(readiness),
        "accuracy": round(sum(outcomes) / count, 4),
        "recent_accuracy": round(
            sum(recent_outcomes) / len(recent_outcomes), 4
        ),
        "days_since_last": round(
            max(
                0.0,
                (now - history[-1]["t"]) / pm.SECONDS_PER_DAY,
            ),
            1,
        ),
        "trend": pm.trend(
            history,
            settings.ml_performance_trend_window,
            settings.ml_performance_trend_threshold,
            settings.ml_performance_trend_min_attempts,
        ),
    }


def student_readiness(user_id, course_id, difficulty="medium"):
    """
    Predict each syllabus topic's probability of a correct next answer.

    A student's own history is used for readiness. Topics without attempts
    remain not assessed; no readiness score is invented for them.
    """
    if difficulty not in pm.DIFFICULTY_INDEX:
        raise ValueError(
            "difficulty must be one of: "
            + ", ".join(pm.DIFFICULTY_INDEX.keys())
        )

    topics = _topics(user_id, course_id)
    histories = _student_histories(user_id, course_id)
    now = _now()

    row, bundle = registry.load_active_bundle(
        None, None, MODEL_TYPE
    )
    model = None

    if (
        bundle
        and bundle.get("feature_names") == pm.FEATURE_NAMES
    ):
        model = bundle["model"]

    question_difficulty = pm.DIFFICULTY_INDEX[difficulty]

    items = [
        _readiness_item(
            topic,
            histories.get(str(topic["id"]), []),
            model,
            now,
            question_difficulty,
        )
        for topic in topics
    ]

    assessed = [
        item for item in items
        if item["readiness"] is not None
    ]
    weakest = sorted(
        assessed, key=lambda item: item["readiness"]
    )[:WEAKEST_COUNT]

    return {
        "model_used": (
            MODEL_TYPE if model is not None
            else "fallback_recent_accuracy"
        ),
        "model_version": (
            row["version"] if row and model is not None else None
        ),
        "algorithm": (
            row["algorithm"] if row and model is not None else None
        ),
        "model_skill": (
            (row.get("metrics") or {}).get("selection_score")
            if row and model is not None else None
        ),
        "difficulty": difficulty,
        "as_of": _iso(now),
        "summary": {
            "topics": len(items),
            "assessed": len(assessed),
            "not_assessed": len(items) - len(assessed),
            "by_band": dict(
                Counter(item["band"] for item in assessed)
            ),
            "by_trend": dict(
                Counter(
                    item["trend"]["label"]
                    for item in assessed
                )
            ),
        },
        "weakest": [
            {
                "topic_id": item["topic_id"],
                "topic_title": item["topic_title"],
                "readiness": item["readiness"],
                "band": item["band"],
                "trend": item["trend"]["label"],
            }
            for item in weakest
        ],
        "topics": items,
    }


# Backward-compatible name used by the earlier implementation.
def get_topic_readiness(user_id, course_id, difficulty="medium"):
    return student_readiness(user_id, course_id, difficulty)
