"""Small scikit-learn compatible classifiers used by the topic classifier."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC


def softmax(scores, temperature=1.0):
    z = np.asarray(scores, dtype=float) * temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class CosineCentroid(ClassifierMixin, BaseEstimator):
    """
    Nearest-centroid classifier using cosine similarity.

    This is the BASELINE: it behaves like the old "closest passage to the
    topic" matching, so a trained model has to beat it to be worth using.
    """

    def __init__(self, temperature=10.0):
        self.temperature = temperature

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        centroids = [
            np.asarray(X[y == label].mean(axis=0)).ravel()
            for label in self.classes_
        ]

        self.centroids_ = normalize(np.vstack(centroids))
        return self

    def decision_function(self, X):
        return np.asarray(normalize(X) @ self.centroids_.T)

    def predict_proba(self, X):
        return softmax(self.decision_function(X), self.temperature)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


class SoftmaxLinearSVC(LinearSVC):
    """Linear SVM that also reports probabilities (softmax of its scores)."""

    def predict_proba(self, X):
        scores = np.asarray(self.decision_function(X))

        if scores.ndim == 1:                     # two classes
            scores = np.column_stack([-scores, scores])

        return softmax(scores, 3.0)
