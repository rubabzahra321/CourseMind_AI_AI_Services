"""
Difficulty prediction: question -> easy / medium / hard.

Pure scikit-learn logic with no database access. The model uses hand-made
numeric features (no raw text), so it works for any subject and can be
explained feature by feature.

Candidates (compared with stratified cross-validation):

    majority_class   always the most common level            (baseline)
    verb_rule        command-verb + length rule, no training (baseline)
    decision_tree
    random_forest
    gradient_boosting
"""

import itertools
import math
import re

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier


LEVELS = ["easy", "medium", "hard"]
LEVEL_INDEX = {name: i for i, name in enumerate(LEVELS)}

BASELINES = ("majority_class", "verb_rule")

LABELS = {
    "majority_class": "Most common level (baseline)",
    "verb_rule": "Command-verb rule (baseline)",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}

# Used only to break exact ties: prefer the sturdier model.
TIE_PRIORITY = [
    "random_forest",
    "gradient_boosting",
    "decision_tree",
    "verb_rule",
    "majority_class",
]


# ---------------------------------------------------------------- features
# Command verbs grouped by the thinking they ask for (a Bloom-style scale).
VERB_LEVELS = {
    # 1  recall
    **dict.fromkeys(
        ["define", "list", "state", "name", "identify", "mention", "give",
         "recall", "label", "outline", "what"], 1),
    # 2  understand
    **dict.fromkeys(
        ["explain", "describe", "discuss", "summarise", "summarize",
         "differentiate", "distinguish", "illustrate", "interpret",
         "classify", "why"], 2),
    # 3  apply
    **dict.fromkeys(
        ["solve", "calculate", "compute", "apply", "implement", "trace",
         "demonstrate", "show", "construct", "find", "perform", "sort",
         "insert", "delete", "write", "use", "draw"], 3),
    # 4  analyse / create
    **dict.fromkeys(
        ["compare", "derive", "analyse", "analyze", "evaluate", "justify",
         "prove", "design", "critique", "assess", "optimise", "optimize",
         "develop", "contrast", "propose"], 4),
}

NEGATIONS = {"not", "except", "never", "incorrect", "false", "cannot", "neither"}
MATH_SYMBOLS = set("+*/=<>^%√∑∫≤≥≠")
SUBPART = re.compile(r"\(\s*[a-hivx]{1,4}\s*\)|(?<!\w)[a-h]\)|\bpart\s+[a-h]\b", re.I)
WORD = re.compile(r"[A-Za-z']+")

FEATURE_NAMES = [
    "n_words", "n_chars", "avg_word_len", "n_sentences", "n_commas",
    "n_digits", "n_math_symbols", "n_subparts", "verb_level", "has_negation",
    "has_options", "n_options", "option_len_mean", "option_overlap",
    "correct_is_longest", "marks", "has_marks",
]

_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _jaccard(a, b):
    if not a and not b:
        return 0.0

    return len(a & b) / len(a | b)


def _option_features(options):
    """Features of a multiple-choice question's options (zeros if none)."""

    texts = [(o.get("option_text") or "").strip() for o in options or []]
    texts = [t for t in texts if t]

    if len(texts) < 2:
        return {"has_options": 0, "n_options": 0, "option_len_mean": 0,
                "option_overlap": 0, "correct_is_longest": 0}

    tokens = [set(w.lower() for w in WORD.findall(t)) for t in texts]
    overlaps = [_jaccard(a, b) for a, b in itertools.combinations(tokens, 2)]
    lengths = [len(t) for t in texts]

    correct = [
        len((o.get("option_text") or "").strip())
        for o in options if o.get("is_correct")
    ]

    longest = bool(correct) and max(correct) == max(lengths) and lengths.count(max(lengths)) == 1

    return {
        "has_options": 1,
        "n_options": len(texts),
        "option_len_mean": float(np.mean(lengths)),
        "option_overlap": float(np.mean(overlaps)),
        "correct_is_longest": int(longest),
    }


def extract_features(text, marks=None, options=None):
    """One question -> {feature name: number}."""

    text = (text or "").strip()
    words = WORD.findall(text)
    lowered = [w.lower() for w in words]
    sentences = [s for s in re.split(r"[.?!]+", text) if s.strip()]

    # The FIRST command verb near the start decides (so a noun later on, such
    # as "sort" in "quick sort", is not mistaken for a command).
    verb_level = next(
        (VERB_LEVELS[w] for w in lowered[:8] if w in VERB_LEVELS), 0)

    features = {
        "n_words": len(words),
        "n_chars": len(text),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "n_sentences": max(1, len(sentences)) if text else 0,
        "n_commas": text.count(","),
        "n_digits": sum(c.isdigit() for c in text),
        "n_math_symbols": sum(c in MATH_SYMBOLS for c in text),
        "n_subparts": len(SUBPART.findall(text)),
        "verb_level": verb_level,
        "has_negation": int(any(w in NEGATIONS for w in lowered)),
        "marks": float(marks) if marks is not None else 0.0,
        "has_marks": int(marks is not None),
    }

    features.update(_option_features(options))

    return features


def feature_matrix(rows):
    """rows: iterable of feature dicts -> numpy array in FEATURE_NAMES order."""

    return np.array(
        [[row[name] for name in FEATURE_NAMES] for row in rows],
        dtype=float,
    ).reshape(-1, len(FEATURE_NAMES))


# ---------------------------------------------------------------- baseline rule
class VerbRule(ClassifierMixin, BaseEstimator):
    """
    No-training baseline: the command verb sets the level, and a long question
    counts as one level harder. This is what a person might guess by hand, so
    a trained model has to beat it.
    """

    LONG_QUESTION_WORDS = 35

    def __init__(self, class_names=None):
        # Names of the levels in the same order as `classes_`. Needed because
        # cross-validation trains on integer-coded labels.
        self.class_names = class_names

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def _names(self):
        if self.class_names is not None:
            return list(self.class_names)

        return [str(c) for c in self.classes_]

    def _levels(self, X):
        X = np.asarray(X, dtype=float)
        verb = X[:, _INDEX["verb_level"]]
        words = X[:, _INDEX["n_words"]]

        base = np.select(
            [verb >= 4, verb >= 2, verb == 1],
            [2, 1, 0],                       # analyse -> hard, understand/apply -> medium, recall -> easy
            default=1,                       # no recognised verb -> medium
        )

        return np.minimum(2, base + (words >= self.LONG_QUESTION_WORDS))

    def predict_proba(self, X):
        levels = self._levels(X)
        names = self._names()
        proba = np.full((len(levels), len(names)), 0.1)

        for row, level in enumerate(levels):
            name = LEVELS[int(level)]

            if name in names:
                proba[row, names.index(name)] = 0.8

        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------- candidates
def candidate_models(seed, trees=100, class_names=None):
    """
    name -> function that builds a fresh, untrained model.

    `trees` is the ensemble size (Random Forest trees / boosting rounds).
    """

    return {
        "majority_class": lambda: DummyClassifier(strategy="most_frequent"),
        "verb_rule": lambda: VerbRule(class_names=class_names),
        "decision_tree": lambda: DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=2,
            class_weight="balanced", random_state=seed),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=trees, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=seed, n_jobs=1),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=trees, max_depth=2, learning_rate=0.1,
            random_state=seed),
    }


# ---------------------------------------------------------------- ordinal helpers
def ordinal_classes(labels):
    """The levels present in the data, easy -> hard."""

    return sorted(set(labels), key=LEVEL_INDEX.get)


def to_ordinal_proba(proba, model_classes, classes):
    """Reorder probability columns from the model's order to easy -> hard."""

    model_classes = list(model_classes)
    return np.column_stack([proba[:, model_classes.index(c)] for c in classes])


def _round(value, digits=4):
    return round(float(value), digits)


# ---------------------------------------------------------------- metrics
def score_block(y_true, proba, classes):
    """
    Accuracy and macro-F1, plus ordinal measures: the levels are ordered, so
    calling a hard question 'medium' is a smaller mistake than calling it 'easy'.
    """

    y_true = np.asarray(y_true)
    classes = list(classes)

    if len(y_true) == 0:
        return None

    predicted = np.array(classes)[np.argmax(proba, axis=1)]
    true_idx = np.array([LEVEL_INDEX[v] for v in y_true])
    pred_idx = np.array([LEVEL_INDEX[v] for v in predicted])
    gap = np.abs(true_idx - pred_idx)

    return {
        "samples": int(len(y_true)),
        "accuracy": _round(accuracy_score(y_true, predicted)),
        "macro_f1": _round(
            f1_score(y_true, predicted, labels=classes,
                     average="macro", zero_division=0)),
        "within_one_level": _round(np.mean(gap <= 1)),
        "mean_level_error": _round(np.mean(gap)),
    }


def expected_level(proba, classes):
    """Probability-weighted level (0 = easy ... 2 = hard) for each question."""

    index = np.array([LEVEL_INDEX[c] for c in classes], dtype=float)
    return np.asarray(proba) @ index


def evaluate_candidates(X, labels, observed_rate, n_splits, seed, trees=100):
    """
    Cross-validate every candidate on the same folds.

    Returns (results, best_name, classes). `observed_rate` is each question's
    real wrong-answer rate (NaN when there are not enough attempts); it is only
    used to report how well predictions agree with what students actually did.
    """

    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    observed_rate = np.asarray(observed_rate, dtype=float)
    classes = ordinal_classes(labels)

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = {}

    names = sorted(set(labels))

    for name, build in candidate_models(seed, trees, names).items():
        model = build()
        proba = cross_val_predict(model, X, labels, cv=folds, method="predict_proba")

        # cross_val_predict orders columns like sorted(classes) (alphabetical)
        proba = to_ordinal_proba(proba, names, classes)

        block = score_block(labels, proba, classes)

        agreement = None
        has_rate = ~np.isnan(observed_rate)

        expected = expected_level(proba[has_rate], classes)

        # A correlation needs both sides to vary (a constant model has none).
        if (
            has_rate.sum() >= 8
            and np.ptp(observed_rate[has_rate]) > 0
            and np.ptp(expected) > 0
        ):
            rho, _p = spearmanr(expected, observed_rate[has_rate])

            if not math.isnan(rho):
                agreement = {"spearman": _round(rho), "samples": int(has_rate.sum())}

        results[name] = {
            "algorithm": name,
            "label": LABELS[name],
            "is_baseline": name in BASELINES,
            "metrics": block,
            "agreement_with_observed": agreement,
            "selection_score": block["macro_f1"],
            "_proba": proba,
        }

    best = max(
        results,
        key=lambda n: (
            results[n]["selection_score"],
            -results[n]["metrics"]["mean_level_error"],
            -TIE_PRIORITY.index(n),
        ),
    )

    return results, best, classes


def detailed_metrics(results, best, labels, classes, n_splits):
    """Metrics stored with the winning model (the 'model card')."""

    labels = np.asarray(labels)
    proba = results[best]["_proba"]
    predicted = np.array(classes)[np.argmax(proba, axis=1)]

    matrix = confusion_matrix(labels, predicted, labels=classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=classes, zero_division=0)

    winner = results[best]["selection_score"]
    baseline_scores = {n: results[n]["selection_score"] for n in BASELINES}
    best_baseline = max(baseline_scores.values())

    return {
        "selection_metric": "macro_f1",
        "selection_score": winner,
        "baseline_scores": baseline_scores,
        "improvement_over_majority": _round(winner - baseline_scores["majority_class"]),
        "improvement_over_rule": _round(winner - baseline_scores["verb_rule"]),
        "improvement_over_best_baseline": _round(winner - best_baseline),
        "cv_folds": int(n_splits),
        "overall": results[best]["metrics"],
        "agreement_with_observed": results[best]["agreement_with_observed"],
        "per_level": [
            {"level": c, "precision": _round(p), "recall": _round(r),
             "f1": _round(f), "support": int(s)}
            for c, p, r, f, s in zip(classes, precision, recall, f1, support)
        ],
        "confusion_matrix": {"labels": list(classes), "matrix": matrix.tolist()},
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
            "agreement_with_observed": r["agreement_with_observed"],
        }
        for name, r in sorted(
            results.items(), key=lambda item: -item[1]["selection_score"])
    ]


def fit_final(name, X, labels, seed, trees=100):
    model = candidate_models(seed, trees)[name]()
    model.fit(np.asarray(X, dtype=float), np.asarray(labels))
    return model


def feature_importances(model):
    """Which features drive the prediction (tree models only)."""

    importances = getattr(model, "feature_importances_", None)

    if importances is None:
        return None

    ranked = sorted(
        zip(FEATURE_NAMES, importances), key=lambda item: -item[1])

    return [
        {"feature": name, "importance": _round(value)}
        for name, value in ranked if value > 0
    ]


# ---------------------------------------------------------------- prediction
def predict_levels(model, X):
    """For each row: {difficulty, probability, probabilities{easy,medium,hard}}."""

    X = np.asarray(X, dtype=float).reshape(-1, len(FEATURE_NAMES))

    if len(X) == 0:
        return []

    proba = model.predict_proba(X)
    known = list(model.classes_)
    out = []

    for row in proba:
        by_level = {level: 0.0 for level in LEVELS}

        for level, p in zip(known, row):
            by_level[level] = float(p)

        best = max(by_level, key=by_level.get)

        out.append({
            "difficulty": best,
            "probability": by_level[best],
            "probabilities": {k: round(v, 4) for k, v in by_level.items()},
        })

    return out


def rule_predict(X):
    """No trained model yet: the command-verb rule."""

    rule = VerbRule().fit(X, np.array(LEVELS))
    return predict_levels(rule, X)
