"""
Performance analysis: how likely is a student to answer the NEXT question on a
topic correctly (their "readiness"), and is the topic improving or declining?

Pure scikit-learn / numpy logic with no database access.

How training examples are made
------------------------------
For every attempt a student made on a topic, the features describe only what
happened BEFORE that attempt (earlier answers, how recently they practised ...)
and the target is whether that attempt was correct. Nothing from the future is
ever used, so the model can be tested honestly on a later period of time.

Candidates (compared on a time-based split, earlier = train, later = test):

    baseline_overall    the student's overall accuracy on the topic     (baseline)
    baseline_ewma       exponentially weighted recent accuracy          (baseline)
    logistic_regression
    gradient_boosting
"""

import math

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DIFFICULTY_INDEX = {"easy": 0, "medium": 1, "hard": 2}

BASELINES = ("baseline_overall", "baseline_ewma")

LABELS = {
    "baseline_overall": "Overall accuracy (baseline)",
    "baseline_ewma": "Recent accuracy, moving average (baseline)",
    "logistic_regression": "Logistic Regression",
    "gradient_boosting": "Gradient Boosting",
}

# Used only to break exact ties: prefer the simpler, steadier model.
TIE_PRIORITY = [
    "logistic_regression",
    "gradient_boosting",
    "baseline_ewma",
    "baseline_overall",
]

BAND_LOW = 0.50          # readiness below this = "low"
BAND_HIGH = 0.75         # readiness at or above this = "high"

SECONDS_PER_DAY = 86400.0
MAX_DAYS = 60.0          # gaps longer than this are treated the same
MAX_RESPONSE_SECONDS = 120.0
MAX_STREAK = 10

FEATURE_NAMES = [
    "n_prev",                # earlier attempts on this topic
    "acc_all",               # overall accuracy so far (smoothed)
    "acc_recent",            # accuracy over the last 5 attempts (smoothed)
    "ewma",                  # exponentially weighted accuracy
    "slope",                 # accuracy trend over the last attempts
    "days_since_last",       # time since the previous attempt (forgetting)
    "mean_response_time",    # average seconds per answer
    "has_response_time",
    "mean_prev_difficulty",  # average difficulty of earlier questions (0-2)
    "q_difficulty",          # difficulty of THIS question (0-2)
    "last_correct",
    "streak",                # correct answers in a row
]

_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


# ---------------------------------------------------------------- features
def _smoothed(correct_count, total):
    """Accuracy with a +1/+2 prior, so 1 of 1 is 0.67 and not a certain 1.0."""

    return (correct_count + 1.0) / (total + 2.0)


def _slope(values):
    """Least-squares slope of 0/1 results per attempt (0 if too few)."""

    n = len(values)

    if n < 3:
        return 0.0

    x = np.arange(n, dtype=float)
    y = np.asarray(values, dtype=float)
    x_centered = x - x.mean()
    denominator = float((x_centered ** 2).sum())

    return float((x_centered * (y - y.mean())).sum() / denominator)


def history_features(history, now_t, q_difficulty, alpha=0.3, recent=5, slope_window=10):
    """
    Features for predicting the NEXT attempt.

    history: earlier attempts on this topic, oldest first, each
             {"t": seconds, "correct": 0/1, "rt": seconds or None, "difficulty": 0-2}
    now_t:   time of the attempt being predicted (seconds)
    """

    n = len(history)
    results = [int(a["correct"]) for a in history]

    ewma = 0.5

    for result in results:
        ewma = alpha * result + (1 - alpha) * ewma

    times = [a["rt"] for a in history if a.get("rt") is not None]
    streak = 0

    for result in reversed(results):
        if result != 1 or streak >= MAX_STREAK:
            break

        streak += 1

    days = (
        min(MAX_DAYS, max(0.0, (now_t - history[-1]["t"]) / SECONDS_PER_DAY))
        if n else 0.0
    )

    return {
        "n_prev": float(n),
        "acc_all": _smoothed(sum(results), n),
        "acc_recent": _smoothed(sum(results[-recent:]), len(results[-recent:])),
        "ewma": ewma,
        "slope": _slope(results[-slope_window:]),
        "days_since_last": days,
        "mean_response_time": (
            min(MAX_RESPONSE_SECONDS, float(np.mean(times))) if times else 0.0),
        "has_response_time": 1.0 if times else 0.0,
        "mean_prev_difficulty": (
            float(np.mean([a["difficulty"] for a in history])) if n else 1.0),
        "q_difficulty": float(q_difficulty),
        "last_correct": float(results[-1]) if n else 0.0,
        "streak": float(streak),
    }


def feature_matrix(rows):
    return np.array(
        [[row[name] for name in FEATURE_NAMES] for row in rows],
        dtype=float,
    ).reshape(-1, len(FEATURE_NAMES))


def build_samples(histories, min_history=2, alpha=0.3):
    """
    histories: one list of attempts per (student, topic).
    Returns (feature dicts, targets, times), one entry per usable attempt.
    """

    rows, targets, times = [], [], []

    for history in histories:
        ordered = sorted(history, key=lambda a: a["t"])

        for i in range(min_history, len(ordered)):
            attempt = ordered[i]

            rows.append(history_features(
                ordered[:i], attempt["t"], attempt["difficulty"], alpha))
            targets.append(int(attempt["correct"]))
            times.append(float(attempt["t"]))

    return rows, np.array(targets, dtype=int), np.array(times, dtype=float)


# ---------------------------------------------------------------- trend
def trend(history, window=10, threshold=0.03, min_attempts=5):
    """Is accuracy on this topic improving, flat or declining?"""

    n = len(history)

    if n < min_attempts:
        return {"label": "not_enough_data", "slope": None, "attempts_used": n}

    ordered = sorted(history, key=lambda a: a["t"])
    results = [int(a["correct"]) for a in ordered][-window:]
    slope = _slope(results)

    if slope > threshold:
        label = "improving"
    elif slope < -threshold:
        label = "declining"
    else:
        label = "flat"

    return {"label": label, "slope": round(slope, 4), "attempts_used": len(results)}


def band(readiness):
    if readiness is None:
        return None

    if readiness < BAND_LOW:
        return "low"

    return "high" if readiness >= BAND_HIGH else "medium"


# ---------------------------------------------------------------- baselines
class FeatureProbability(ClassifierMixin, BaseEstimator):
    """No-training baseline: the probability IS one of the features."""

    def __init__(self, feature_index=0):
        self.feature_index = feature_index

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.clip(np.asarray(X, dtype=float)[:, self.feature_index], 0.001, 0.999)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def candidate_models(seed, ensemble=100):
    """name -> function that builds a fresh, untrained model."""

    return {
        "baseline_overall": lambda: FeatureProbability(_INDEX["acc_all"]),
        "baseline_ewma": lambda: FeatureProbability(_INDEX["ewma"]),
        "logistic_regression": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=2000)),
        ]),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=ensemble, max_depth=2, learning_rate=0.1,
            subsample=0.8, random_state=seed),
    }


# ---------------------------------------------------------------- evaluation
def _round(value, digits=4):
    return round(float(value), digits)


def temporal_split(times, test_fraction):
    """
    Earlier attempts train, the latest `test_fraction` tests. The model is never
    tested on the past it learned from.
    """

    order = np.argsort(times, kind="stable")
    cut = int(round(len(order) * (1 - test_fraction)))

    return order[:cut], order[cut:]


def calibration_table(y_true, p, bins=5):
    """Do predicted chances come true? (e.g. 70% predicted -> ~70% observed)"""

    y_true = np.asarray(y_true)
    p = np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    which = np.minimum(np.digitize(p, edges[1:-1]), bins - 1)

    table, ece = [], 0.0

    for b in range(bins):
        mask = which == b

        if not mask.any():
            continue

        predicted, observed = float(p[mask].mean()), float(y_true[mask].mean())
        ece += mask.mean() * abs(predicted - observed)

        table.append({
            "range": [_round(edges[b], 2), _round(edges[b + 1], 2)],
            "samples": int(mask.sum()),
            "mean_predicted": _round(predicted),
            "observed_rate": _round(observed),
        })

    return table, _round(ece)


def score(y_true, p, base_rate):
    """Log loss, AUC, Brier score and accuracy on the test period."""

    y_true = np.asarray(y_true)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)

    reference = log_loss(y_true, np.full(len(y_true), base_rate), labels=[0, 1])
    loss = log_loss(y_true, p, labels=[0, 1])

    return {
        "samples": int(len(y_true)),
        "log_loss": _round(loss),
        "auc": _round(roc_auc_score(y_true, p)) if len(set(y_true)) == 2 else None,
        "brier": _round(brier_score_loss(y_true, p)),
        "accuracy": _round(accuracy_score(y_true, p >= 0.5)),
        # 1 = perfect, 0 = no better than always guessing the average, < 0 = worse
        "skill": _round(1 - loss / reference),
        "reference_log_loss": _round(reference),
    }


def evaluate_candidates(X, y, times, test_fraction, seed, ensemble=100):
    """
    Train every candidate on the earlier period and score it on the later one.

    Returns (results, best_name, split). Raises ValueError when either period
    has only one outcome (nothing to learn or test).
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    train, test = temporal_split(times, test_fraction)

    if len(set(y[train])) < 2 or len(set(y[test])) < 2:
        raise ValueError(
            "the earlier or the later period contains only correct (or only "
            "wrong) answers")

    base_rate = float(y[train].mean())
    results = {}

    for name, build in candidate_models(seed, ensemble).items():
        model = build().fit(X[train], y[train])
        p = model.predict_proba(X[test])[:, 1]
        metrics = score(y[test], p, base_rate)
        table, ece = calibration_table(y[test], p)

        results[name] = {
            "algorithm": name,
            "label": LABELS[name],
            "is_baseline": name in BASELINES,
            "metrics": metrics,
            "calibration": table,
            "calibration_error": ece,
            "selection_score": metrics["skill"],
            "_p": p,
        }

    best = max(
        results,
        key=lambda n: (
            results[n]["selection_score"],
            results[n]["metrics"]["auc"] or 0,
            -TIE_PRIORITY.index(n),
        ),
    )

    return results, best, {
        "train": int(len(train)),
        "test": int(len(test)),
        "base_rate_train": _round(base_rate),
        "split_time": float(times[test[0]]) if len(test) else None,
    }


def detailed_metrics(results, best, split):
    """Metrics stored with the winning model (the 'model card')."""

    winner = results[best]
    baseline_skill = {n: results[n]["selection_score"] for n in BASELINES}
    best_baseline = max(BASELINES, key=lambda n: results[n]["selection_score"])

    return {
        "selection_metric": "log_loss_skill",
        "selection_note": (
            "Skill = 1 - (model log loss / log loss of always predicting the "
            "average). 0 means no better than guessing the average."),
        "selection_score": winner["selection_score"],
        "baseline_scores": baseline_skill,
        "best_baseline": best_baseline,
        "improvement_over_best_baseline": _round(
            winner["selection_score"] - results[best_baseline]["selection_score"]),
        "auc_improvement_over_best_baseline": (
            _round(winner["metrics"]["auc"] - results[best_baseline]["metrics"]["auc"])
            if winner["metrics"]["auc"] is not None
            and results[best_baseline]["metrics"]["auc"] is not None else None),
        "test_metrics": winner["metrics"],
        "calibration": winner["calibration"],
        "calibration_error": winner["calibration_error"],
        "split": split,
    }


def comparison_table(results, best):
    return [
        {
            "algorithm": r["algorithm"],
            "label": r["label"],
            "is_baseline": r["is_baseline"],
            "selected": name == best,
            "selection_score": r["selection_score"],
            "metrics": r["metrics"],
            "calibration_error": r["calibration_error"],
        }
        for name, r in sorted(
            results.items(), key=lambda item: -item[1]["selection_score"])
    ]


def fit_final(name, X, y, seed, ensemble=100):
    """Train the chosen algorithm on ALL data (earlier and later periods)."""

    return candidate_models(seed, ensemble)[name]().fit(
        np.asarray(X, dtype=float), np.asarray(y))


def feature_importance(model):
    """Which features move the prediction (direction shown for the linear model)."""

    clf = model.named_steps["clf"] if isinstance(model, Pipeline) else model

    if hasattr(clf, "coef_"):
        values = clf.coef_[0]
        ranked = sorted(zip(FEATURE_NAMES, values), key=lambda i: -abs(i[1]))
        return [
            {"feature": n, "weight": _round(v),
             "effect": "raises readiness" if v > 0 else "lowers readiness"}
            for n, v in ranked if abs(v) > 1e-9
        ]

    if hasattr(clf, "feature_importances_"):
        ranked = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda i: -i[1])
        return [{"feature": n, "importance": _round(v)} for n, v in ranked if v > 0]

    return None


# ---------------------------------------------------------------- prediction
def predict_probability(model, X):
    """P(next answer is correct) for each row."""

    X = np.asarray(X, dtype=float).reshape(-1, len(FEATURE_NAMES))

    if len(X) == 0:
        return np.array([])

    return model.predict_proba(X)[:, 1]
