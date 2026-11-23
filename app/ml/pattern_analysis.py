"""Clustering utilities for per-topic learning-pattern analysis."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

FEATURE_NAMES = [
    "accuracy", "trend_slope", "average_response_time",
    "attempt_count", "days_since_practice", "repeated_mistake_rate",
]


def topic_features(history, now):
    """Create one numeric vector from a learner's history for one topic."""
    from app.ml.performance import extract_features
    base = extract_features(history, now)
    wrong = [row for row in history if not row.get("is_correct")]
    seen_wrong = set()
    repeated = 0
    for row in wrong:
        question_id = str(row.get("question_id") or "")
        if question_id and question_id in seen_wrong:
            repeated += 1
        elif question_id:
            seen_wrong.add(question_id)
    repeated_rate = repeated / len(wrong) if wrong else 0.0
    return {
        "accuracy": base["overall_accuracy"],
        "trend_slope": base["trend_slope"],
        "average_response_time": base["average_response_time"],
        "attempt_count": base["attempt_count"],
        "days_since_practice": base["days_since_practice"],
        "repeated_mistake_rate": repeated_rate,
    }


def matrix(rows):
    return np.asarray([[float(row.get(name, 0.0)) for name in FEATURE_NAMES] for row in rows], dtype=float)


def _cluster_name(center):
    accuracy, slope, _response, _attempts, days, _repeated = center
    if accuracy < 0.5 and slope <= 0.015:
        return "weak_and_not_improving"
    if accuracy < 0.5 and slope > 0.015:
        return "weak_but_improving"
    if accuracy >= 0.65 and days >= 7:
        return "strong_but_going_stale"
    if accuracy >= 0.65:
        return "strong_and_active"
    if slope > 0.015:
        return "developing_and_improving"
    return "mixed_performance"


def fit_patterns(rows, random_seed=42):
    """Fit k=3..5, select by silhouette, then fit a small explanation tree."""
    if len(rows) < 8:
        raise ValueError("At least 8 assessed student-topic histories are required across the pooled dataset.")
    X = matrix(rows)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    candidates = []
    for k in range(3, min(5, len(rows) - 1) + 1):
        model = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        labels = model.fit_predict(scaled)
        if len(set(labels.tolist())) < 2 or len(set(labels.tolist())) >= len(rows):
            continue
        score = float(silhouette_score(scaled, labels))
        candidates.append((score, k, model, labels))
    if not candidates:
        raise ValueError("Not enough distinct topic behaviour to form stable clusters.")
    score, k, model, labels = max(candidates, key=lambda item: (item[0], -item[1]))
    names = {}
    for cluster_id, center in enumerate(model.cluster_centers_):
        original_center = scaler.inverse_transform([center])[0]
        names[int(cluster_id)] = _cluster_name(original_center)
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=random_seed)
    tree.fit(scaled, labels)
    rules = export_text(tree, feature_names=FEATURE_NAMES)
    counts = {names[int(cid)]: int((labels == cid).sum()) for cid in range(k)}
    return {
        "scaler": scaler,
        "kmeans": model,
        "cluster_names": names,
        "explanation_tree": tree,
        "feature_names": FEATURE_NAMES,
    }, {
        "algorithm": "kmeans",
        "selected_k": int(k),
        "silhouette_score": round(score, 6),
        "cluster_sizes": counts,
        "candidate_comparison": [
            {"k": int(item[1]), "silhouette_score": round(float(item[0]), 6)}
            for item in candidates
        ],
        "explanation_rules": rules,
        "n_training_samples": len(rows),
    }


def predict_pattern(bundle, row):
    X = matrix([row])
    scaled = bundle["scaler"].transform(X)
    cluster_id = int(bundle["kmeans"].predict(scaled)[0])
    label = bundle["cluster_names"].get(cluster_id, "mixed_performance")
    return label


def fallback_pattern(row):
    """Transparent rule fallback when no global clustering model is available."""
    if row["attempt_count"] <= 0:
        return "not_yet_assessed"
    return _cluster_name([
        row["accuracy"], row["trend_slope"], row["average_response_time"],
        row["attempt_count"], row["days_since_practice"], row["repeated_mistake_rate"],
    ])
