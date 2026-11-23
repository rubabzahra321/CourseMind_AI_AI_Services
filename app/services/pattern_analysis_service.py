"""Pooled anonymised clustering and per-student topic pattern assignment."""
from collections import defaultdict
from datetime import datetime, timezone

from app.core.config import settings
from app.ml import pattern_analysis as pa, registry
from app.repositories import ml_repository
from app.services import performance_service
from app.services.insights_service import _topics

MODEL_TYPE = "patterns"
MIN_TOPIC_ROWS = 8


def _all_topic_rows():
    grouped = performance_service._source_data()
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for (_user_id, _topic_id), history in grouped.items():
        if not history:
            continue
        rows.append(pa.topic_features(history, now))
    return rows


def train_patterns(force=False):
    rows = _all_topic_rows()
    if len(rows) < MIN_TOPIC_ROWS:
        raise ValueError(
            f"Not enough assessed topic histories. Found {len(rows)}; need at least {MIN_TOPIC_ROWS} pooled student-topic histories."
        )
    bundle, metrics = pa.fit_patterns(rows, settings.ml_random_seed)
    comparison = metrics.pop("candidate_comparison")
    row = registry.save_global_version(
        MODEL_TYPE, "kmeans", len(rows), metrics,
        [{"algorithm": "kmeans", "selected": True, "candidate_k": comparison}],
        bundle,
    )
    activated, reason = registry.activate_global_if_better(row, force=force)
    row["is_active"] = activated
    return {
        "model_type": MODEL_TYPE,
        "scope": "global_anonymised",
        "algorithm": "kmeans",
        "version": row["version"],
        "is_active": activated,
        "activated": activated,
        "activation_reason": reason,
        "n_training_samples": row["n_training_samples"],
        "metrics": row.get("metrics") or {},
    }


def get_status():
    active = ml_repository.get_global_active(MODEL_TYPE)
    history = ml_repository.list_global_models(MODEL_TYPE)
    count = len(_all_topic_rows())
    return {
        "model_type": MODEL_TYPE,
        "scope": "global_anonymised",
        "active_model": ({
            "algorithm": active["algorithm"],
            "version": active["version"],
            "n_training_samples": active["n_training_samples"],
            "metrics": active.get("metrics") or {},
            "is_active": True,
        } if active else None),
        "using_fallback": active is None,
        "training_data": {
            "assessed_topic_histories": count,
            "personal_fields_in_features": False,
            "question_text_used": False,
        },
        "can_train": count >= MIN_TOPIC_ROWS,
        "history": [{
            "version": row["version"], "algorithm": row["algorithm"],
            "n_training_samples": row["n_training_samples"],
            "metrics": row.get("metrics") or {}, "is_active": bool(row.get("is_active")),
        } for row in history],
    }


def get_learning_patterns(user_id, course_id):
    topics = _topics(user_id, course_id)
    grouped = performance_service._source_data(user_id=user_id, course_id=course_id)
    _model_row, bundle = registry.load_global_active_bundle(MODEL_TYPE)
    now = datetime.now(timezone.utc).isoformat()
    results = []
    for topic in topics:
        history = grouped.get((str(user_id), str(topic["id"])), [])
        if not history:
            results.append({
                "topic_id": str(topic["id"]), "topic_title": topic["title"],
                "pattern": "not_yet_assessed", "explanation": "There are no graded practice attempts for this topic.",
                "attempts": 0, "model_used": "not_assessed",
            })
            continue
        features = pa.topic_features(history, now)
        if bundle:
            pattern = pa.predict_pattern(bundle, features)
            model_used = "kmeans"
        else:
            pattern = pa.fallback_pattern(features)
            model_used = "fallback_rules"
        explanations = {
            "weak_and_not_improving": "Accuracy is low and the recent trend is not improving.",
            "weak_but_improving": "Accuracy is currently low, but the recent trend is improving.",
            "strong_but_going_stale": "Accuracy is strong, but this topic has not been practised recently.",
            "strong_and_active": "Accuracy is strong and the topic has been practised recently.",
            "developing_and_improving": "Performance is developing and the recent trend is improving.",
            "mixed_performance": "The measured indicators show a mixed learning pattern.",
        }
        results.append({
            "topic_id": str(topic["id"]), "topic_title": topic["title"],
            "pattern": pattern, "explanation": explanations.get(pattern, "Pattern assigned from topic practice history."),
            "attempts": int(features["attempt_count"]),
            "accuracy": round(features["accuracy"], 4),
            "trend_slope": round(features["trend_slope"], 6),
            "days_since_practice": round(features["days_since_practice"], 2),
            "repeated_mistake_rate": round(features["repeated_mistake_rate"], 4),
            "model_used": model_used,
        })
    return {
        "model_used": "kmeans" if bundle else "fallback_rules",
        "model_version": _model_row.get("version") if _model_row else None,
        "topics": results,
    }
