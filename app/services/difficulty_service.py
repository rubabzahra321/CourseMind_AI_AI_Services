"""
Difficulty model orchestration: build training data from the database, train
and compare models, save them, and predict difficulty for past-paper questions.

Labels (what the model learns from):
  * Gemini's difficulty rating on each generated question (weak labels: the
    model learns to imitate Gemini), and
  * observed difficulty - once a question has been attempted enough times, the
    real wrong-answer rate replaces Gemini's rating.
"""

from collections import Counter, defaultdict

import numpy as np

from app.core.config import settings
from app.db.client import get_db
from app.ml import difficulty as dm
from app.ml import registry
from app.repositories import ml_repository
from app.services.insights_service import _all_rows
from app.services.ml_service import InsufficientTrainingData


MODEL_TYPE = "difficulty"
BATCH = 50


# ---------------------------------------------------------------- training data
def _options_by_question(user_id, question_ids):
    options = defaultdict(list)

    for start in range(0, len(question_ids), BATCH):
        batch = question_ids[start:start + BATCH]

        rows = (
            get_db().table("question_options")
            .select("question_id, option_text, option_order, is_correct")
            .in_("question_id", batch)
            .execute().data or []
        )

        for row in rows:
            options[str(row["question_id"])].append(row)

    return options


def _attempt_counts(user_id, question_ids):
    """question_id -> (attempts, wrong)."""

    wanted = set(question_ids)
    totals = defaultdict(lambda: [0, 0])

    for row in _all_rows(
        lambda: get_db().table("practice_attempts")
        .select("question_id, is_correct")
        .eq("user_id", str(user_id))
    ):
        qid = str(row["question_id"])

        if qid in wanted:
            totals[qid][0] += 1
            totals[qid][1] += 0 if row["is_correct"] else 1

    return totals


def observed_level(wrong_rate):
    if wrong_rate < settings.ml_difficulty_easy_max_wrong:
        return "easy"

    if wrong_rate > settings.ml_difficulty_hard_min_wrong:
        return "hard"

    return "medium"


def build_training_set(user_id, course_id):
    """
    One row per active generated question:
    features, label, where the label came from, and the observed wrong rate.
    """

    questions = _all_rows(
        lambda: get_db().table("generated_questions")
        .select("id, question_text, difficulty")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
        .eq("status", "active")
    )

    ids = [str(q["id"]) for q in questions]
    options = _options_by_question(user_id, ids)
    attempts = _attempt_counts(user_id, ids)

    rows = []

    for question in questions:
        qid = str(question["id"])
        count, wrong = attempts.get(qid, (0, 0))

        label, source, rate = question["difficulty"], "gemini", None

        if count >= settings.ml_difficulty_min_attempts:
            rate = wrong / count
            label, source = observed_level(rate), "observed"

        rows.append({
            "question_id": qid,
            "features": dm.extract_features(
                question["question_text"], None, options.get(qid)),
            "label": label,
            "source": source,
            "wrong_rate": rate,
            "attempts": count,
        })

    sources = Counter(r["source"] for r in rows)

    return rows, {
        "questions": len(rows),
        "gemini_labels": sources.get("gemini", 0),
        "observed_labels": sources.get("observed", 0),
        "min_attempts_for_observed": settings.ml_difficulty_min_attempts,
        "level_counts": {
            level: n for level, n in
            sorted(Counter(r["label"] for r in rows).items(),
                   key=lambda i: dm.LEVEL_INDEX.get(i[0], 9))
        },
    }


def _usable(rows):
    """Keep only levels with enough examples to learn and cross-validate."""

    per_level = Counter(r["label"] for r in rows)
    kept = {
        level for level, n in per_level.items()
        if n >= settings.ml_difficulty_min_per_class
    }

    left_out = [
        {"level": level, "examples": n}
        for level, n in sorted(per_level.items(), key=lambda i: dm.LEVEL_INDEX.get(i[0], 9))
        if level not in kept
    ]

    return kept, left_out


def _insufficient_message(counts, kept_levels, total_usable):
    distribution = ", ".join(
        f"{n} {level}" for level, n in counts["level_counts"].items()
    ) or "no questions"

    return (
        "Not enough practice questions to train a difficulty model yet. "
        f"Found {counts['questions']} ({distribution}); training needs at "
        f"least {settings.ml_difficulty_min_samples} questions in total and "
        f"at least 2 difficulty levels with {settings.ml_difficulty_min_per_class}+ "
        f"questions each (currently {kept_levels} level(s) qualify, "
        f"{total_usable} usable questions). Generate more practice questions "
        "in different difficulty levels and train again. Until then difficulty "
        "is estimated with a simple command-verb rule."
    )


def _card(row, activated=None, reason=None):
    card = {
        "id": row["id"],
        "model_type": row["model_type"],
        "algorithm": row["algorithm"],
        "algorithm_label": dm.LABELS.get(row["algorithm"], row["algorithm"]),
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
def train_difficulty_model(user_id, course_id, force=False):
    rows, counts = build_training_set(user_id, course_id)

    kept, left_out = _usable(rows)
    usable = [r for r in rows if r["label"] in kept]

    if (
        len(rows) < settings.ml_difficulty_min_samples
        or len(kept) < 2
        or len(usable) < settings.ml_difficulty_min_samples
    ):
        raise InsufficientTrainingData(
            _insufficient_message(counts, len(kept), len(usable)))

    labels = [r["label"] for r in usable]
    n_splits = min(settings.ml_cv_max_folds, min(Counter(labels).values()))

    if n_splits < 2:
        raise InsufficientTrainingData(
            _insufficient_message(counts, len(kept), len(usable)))

    X = dm.feature_matrix([r["features"] for r in usable])
    rates = [
        np.nan if r["wrong_rate"] is None else r["wrong_rate"] for r in usable
    ]

    results, best, classes = dm.evaluate_candidates(
        X, labels, rates, n_splits, settings.ml_random_seed,
        settings.ml_difficulty_ensemble_size)

    metrics = dm.detailed_metrics(results, best, labels, classes, n_splits)

    model = dm.fit_final(
        best, X, labels, settings.ml_random_seed,
        settings.ml_difficulty_ensemble_size)

    metrics["feature_importance"] = dm.feature_importances(model)
    metrics["training_data"] = {
        **counts,
        "samples_used": len(usable),
        "levels_trained": [c for c in classes],
        "levels_left_out": left_out,
        "random_seed": settings.ml_random_seed,
    }
    metrics["label_note"] = (
        "Every label is Gemini's own difficulty rating, so the model learns "
        "to imitate Gemini. It improves as real practice attempts replace "
        "those ratings."
        if counts["observed_labels"] == 0 else
        f"{counts['observed_labels']} of {counts['questions']} labels come from "
        "real student results; the rest are Gemini's ratings."
    )

    bundle = {
        "model": model,
        "algorithm": best,
        "feature_names": dm.FEATURE_NAMES,
        "levels": list(classes),
    }

    row = registry.save_version(
        user_id, course_id, MODEL_TYPE, best,
        len(usable), metrics, dm.comparison_table(results, best), bundle,
    )

    activated, reason = registry.activate_if_better(row, force=force)
    row["is_active"] = activated

    return _card(row, activated, reason)


# ---------------------------------------------------------------- status
def get_status(user_id, course_id):
    active = ml_repository.get_active(user_id, course_id, MODEL_TYPE)
    history = ml_repository.list_models(user_id, course_id, MODEL_TYPE)

    rows, counts = build_training_set(user_id, course_id)
    kept, left_out = _usable(rows)
    usable = [r for r in rows if r["label"] in kept]

    can_train = (
        len(rows) >= settings.ml_difficulty_min_samples
        and len(kept) >= 2
        and len(usable) >= settings.ml_difficulty_min_samples
    )

    return {
        "model_type": MODEL_TYPE,
        "active_model": _card(active) if active else None,
        "using_fallback": active is None,
        "history": [
            {
                "version": row["version"],
                "algorithm": row["algorithm"],
                "algorithm_label": dm.LABELS.get(row["algorithm"], row["algorithm"]),
                "selection_score": (row.get("metrics") or {}).get("selection_score"),
                "n_training_samples": row["n_training_samples"],
                "is_active": bool(row.get("is_active")),
                "trained_at": row.get("trained_at"),
            }
            for row in history
        ],
        "training_data": {
            **counts,
            "levels_ready": sorted(kept, key=dm.LEVEL_INDEX.get),
            "levels_left_out": left_out,
            "minimum_questions": settings.ml_difficulty_min_samples,
            "minimum_per_level": settings.ml_difficulty_min_per_class,
        },
        "can_train": can_train,
        "message": (
            None if can_train
            else _insufficient_message(counts, len(kept), len(usable))
        ),
    }


# ---------------------------------------------------------------- predict
def predict_difficulty(user_id, course_id, question_ids=None, save=False,
                       min_confidence=None, limit=None):
    """
    Predict difficulty for PAST-PAPER questions (they have no difficulty yet).

    By default it looks at questions without a prediction. A prediction is only
    SAVED (predicted_difficulty on the question) when save=true, a trained model
    is in use and the confidence reaches the threshold. The rule fallback is
    never saved.
    """

    min_confidence = (
        settings.ml_min_confidence if min_confidence is None else min_confidence)
    limit = min(limit or settings.ml_max_classify_questions,
                settings.ml_max_classify_questions)

    questions = _all_rows(
        lambda: get_db().table("past_paper_questions")
        .select("id, question_number, question_text, marks, question_type, "
                "review_status, predicted_difficulty")
        .eq("user_id", str(user_id))
        .eq("course_id", str(course_id))
    )
    by_id = {str(q["id"]): q for q in questions}

    not_found = []

    if question_ids:
        wanted = [str(q) for q in question_ids]
        not_found = [q for q in wanted if q not in by_id]
        selected = [by_id[q] for q in wanted if q in by_id]
    else:
        selected = [q for q in questions if not q.get("predicted_difficulty")]

    selected = selected[:limit]

    row, bundle = registry.load_active_bundle(user_id, course_id, MODEL_TYPE)

    X = dm.feature_matrix([
        dm.extract_features(q["question_text"], q.get("marks"), None)
        for q in selected
    ])

    if bundle:
        predictions = dm.predict_levels(bundle["model"], X)
        model_used = MODEL_TYPE
    else:
        predictions = dm.rule_predict(X)
        model_used = "fallback_rule"

    version = row["version"] if (row and bundle) else None
    items, saved = [], 0

    for question, prediction in zip(selected, predictions):
        needs_review = prediction["probability"] < min_confidence
        was_saved = False

        if save and bundle and not needs_review:
            try:
                get_db().table("past_paper_questions").update({
                    "predicted_difficulty": prediction["difficulty"],
                    "difficulty_confidence": round(prediction["probability"], 4),
                    "difficulty_model_version": version,
                }).eq("id", str(question["id"])).eq("user_id", str(user_id)).execute()
                was_saved = True
                saved += 1
            except Exception:
                was_saved = False

        items.append({
            "question_id": str(question["id"]),
            "question_number": question.get("question_number"),
            "question_type": question.get("question_type"),
            "marks": question.get("marks"),
            "question_text": (question["question_text"] or "")[:200],
            "previous_prediction": question.get("predicted_difficulty"),
            "predicted_difficulty": prediction["difficulty"],
            "confidence": round(prediction["probability"], 4),
            "probabilities": prediction["probabilities"],
            "needs_review": needs_review,
            "saved": was_saved,
        })

    return {
        "model_used": model_used,
        "model_version": version,
        "algorithm": row["algorithm"] if (row and bundle) else None,
        "min_confidence": min_confidence,
        "total": len(items),
        "by_level": dict(Counter(i["predicted_difficulty"] for i in items)),
        "needs_review_count": sum(1 for i in items if i["needs_review"]),
        "saved_count": saved,
        "not_found": not_found,
        "items": items,
    }
