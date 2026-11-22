"""
Topic classification: question text -> syllabus topic.

Pure scikit-learn logic with no database access, so it can be tested and
explained on its own. Data loading and saving live in services/ml_service.py.

Five candidates are trained and compared with stratified cross-validation:

    centroid_baseline    cosine nearest-centroid (what the old matching did)
    logistic_regression
    linear_svm
    naive_bayes
    knn

All use TF-IDF features (words and 2-word phrases).
"""

import math

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from app.ml.estimators import CosineCentroid, SoftmaxLinearSVC


BASELINE = "centroid_baseline"

LABELS = {
    "centroid_baseline": "Nearest centroid (baseline)",
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "naive_bayes": "Multinomial Naive Bayes",
    "knn": "k-Nearest Neighbours",
}

# Used only to break exact ties: prefer the more dependable model.
TIE_PRIORITY = [
    "logistic_regression",
    "linear_svm",
    "naive_bayes",
    "knn",
    "centroid_baseline",
]


def _vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )


def candidate_pipelines(n_samples, n_splits, seed):
    """name -> function that builds a fresh, untrained pipeline."""

    # Smallest training fold, so kNN never asks for more neighbours than exist.
    neighbours = max(1, min(3, n_samples - math.ceil(n_samples / n_splits)))

    return {
        "centroid_baseline": lambda: Pipeline([
            ("tfidf", _vectorizer()),
            ("clf", CosineCentroid()),
        ]),
        "logistic_regression": lambda: Pipeline([
            ("tfidf", _vectorizer()),
            ("clf", LogisticRegression(
                C=10.0, max_iter=2000, class_weight="balanced")),
        ]),
        "linear_svm": lambda: Pipeline([
            ("tfidf", _vectorizer()),
            ("clf", SoftmaxLinearSVC(
                C=1.0, class_weight="balanced", random_state=seed)),
        ]),
        "naive_bayes": lambda: Pipeline([
            ("tfidf", _vectorizer()),
            ("clf", MultinomialNB(alpha=0.5)),
        ]),
        "knn": lambda: Pipeline([
            ("tfidf", _vectorizer()),
            ("clf", KNeighborsClassifier(
                n_neighbors=neighbours, metric="cosine", weights="distance")),
        ]),
    }


# ---------------------------------------------------------------- metrics
def _round(value, digits=4):
    return round(float(value), digits)


def score_block(y_true, proba, classes):
    """Accuracy, macro-F1 and top-3 accuracy of out-of-fold predictions."""

    y_true = np.asarray(y_true)
    classes = np.asarray(classes)

    if len(y_true) == 0:
        return None

    predicted = classes[np.argmax(proba, axis=1)]
    top3 = np.argsort(-proba, axis=1)[:, :3]
    hits = [
        label in set(classes[row])
        for label, row in zip(y_true, top3)
    ]

    return {
        "samples": int(len(y_true)),
        "accuracy": _round(accuracy_score(y_true, predicted)),
        "macro_f1": _round(
            f1_score(y_true, predicted, labels=classes,
                     average="macro", zero_division=0)),
        "top3_accuracy": _round(np.mean(hits)),
    }


def _per_topic_report(y_true, predicted, classes, titles):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, labels=classes, zero_division=0)

    return [
        {
            "topic_id": str(label),
            "topic_title": titles.get(str(label), str(label)),
            "precision": _round(p),
            "recall": _round(r),
            "f1": _round(f),
            "support": int(s),
        }
        for label, p, r, f, s in zip(classes, precision, recall, f1, support)
    ]


def evaluate_candidates(texts, labels, sources, n_splits, seed, min_question_samples):
    """
    Cross-validate every candidate on the same folds.

    Returns (results, best_name, basis). `basis` says which samples the winner
    was chosen on: "question_only" when there are enough real exam questions
    (that is what the model will predict in use), otherwise "all_samples".
    """

    texts = np.asarray(texts, dtype=object)
    labels = np.asarray(labels)
    sources = np.asarray(sources)
    classes = np.unique(labels)

    question_mask = sources == "question"
    basis = (
        "question_only"
        if int(question_mask.sum()) >= min_question_samples
        and len(np.unique(labels[question_mask])) >= 2
        else "all_samples"
    )

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    factories = candidate_pipelines(len(labels), n_splits, seed)

    results = {}

    for name, build in factories.items():
        proba = cross_val_predict(
            build(), texts, labels, cv=folds, method="predict_proba")

        overall = score_block(labels, proba, classes)
        questions = score_block(
            labels[question_mask], proba[question_mask], classes)

        chosen = questions if basis == "question_only" else overall

        results[name] = {
            "algorithm": name,
            "label": LABELS[name],
            "is_baseline": name == BASELINE,
            "overall": overall,
            "question_only": questions,
            "selection_score": chosen["macro_f1"],
            "_proba": proba,
        }

    best = max(
        results,
        key=lambda n: (
            results[n]["selection_score"],
            -TIE_PRIORITY.index(n),
        ),
    )

    return results, best, basis, classes


def detailed_metrics(results, best, labels, titles, classes, basis, n_splits):
    """Metrics stored with the winning model (the 'model card')."""

    labels = np.asarray(labels)
    classes = np.asarray(classes)
    proba = results[best]["_proba"]
    predicted = classes[np.argmax(proba, axis=1)]

    # Report in topic-title order (not random UUID order) so it reads well.
    order = sorted(
        range(len(classes)),
        key=lambda i: titles.get(str(classes[i]), str(classes[i])).lower(),
    )
    ordered = classes[order]

    matrix = confusion_matrix(labels, predicted, labels=ordered)

    baseline = results[BASELINE]["selection_score"]
    winner = results[best]["selection_score"]

    return {
        "selection_basis": basis,
        "selection_metric": "macro_f1",
        "selection_score": winner,
        "baseline_score": baseline,
        "improvement_over_baseline": _round(winner - baseline),
        "cv_folds": int(n_splits),
        "overall": results[best]["overall"],
        "question_only": results[best]["question_only"],
        "per_topic": _per_topic_report(labels, predicted, ordered, titles),
        "confusion_matrix": {
            "labels": [titles.get(str(c), str(c)) for c in ordered],
            "topic_ids": [str(c) for c in ordered],
            "matrix": matrix.tolist(),
        },
    }


def comparison_table(results, best):
    return [
        {
            "algorithm": r["algorithm"],
            "label": r["label"],
            "is_baseline": r["is_baseline"],
            "selected": name == best,
            "selection_score": r["selection_score"],
            "overall": r["overall"],
            "question_only": r["question_only"],
        }
        for name, r in sorted(
            results.items(),
            key=lambda item: -item[1]["selection_score"],
        )
    ]


def fit_final(name, texts, labels, n_splits, seed):
    """Train the chosen algorithm on ALL data."""

    build = candidate_pipelines(len(labels), n_splits, seed)[name]
    pipeline = build()
    pipeline.fit(np.asarray(texts, dtype=object), np.asarray(labels))
    return pipeline


# ---------------------------------------------------------------- prediction
def predict_top_k(pipeline, texts, k=3):
    """For each text: [(topic_id, probability), ...] best first."""

    if not texts:
        return []

    proba = pipeline.predict_proba(np.asarray(texts, dtype=object))
    classes = np.asarray(pipeline.classes_)
    order = np.argsort(-proba, axis=1)[:, :k]

    return [
        [(str(classes[i]), float(row[i])) for i in indices]
        for row, indices in zip(proba, order)
    ]


def keyword_fallback(topics, texts, k=3):
    """
    No trained model yet: rank topics by TF-IDF cosine similarity between
    the question and each topic's title and description. Needs no labels.
    """

    if not topics or not texts:
        return [[] for _ in texts]

    topic_texts = [
        f"{t['title']} {t.get('description') or ''}".strip() for t in topics
    ]

    try:
        vectorizer = _vectorizer().fit(topic_texts + list(texts))
    except ValueError:                           # no usable words at all
        return [[] for _ in texts]

    topic_matrix = vectorizer.transform(topic_texts)
    question_matrix = vectorizer.transform(texts)
    similarity = (question_matrix @ topic_matrix.T).toarray()

    results = []

    for row in similarity:
        order = np.argsort(-row)[:k]
        results.append([
            (str(topics[i]["id"]), float(row[i]))
            for i in order
            if row[i] > 0
        ])

    return results
