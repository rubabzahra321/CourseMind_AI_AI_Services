"""
Demo of the difficulty model: compares five algorithms with cross-validation
and prints a results table, the confusion matrix and the most useful features.

Run from the ai-service folder (no database or API keys needed):

    python -m scripts.ml_difficulty_demo

It uses SYNTHETIC questions from tests/difficulty_corpus.py. The numbers show
that the pipeline works; they are NOT results on your real course. For those,
train from the app and read GET /courses/:id/ml/difficulty/status.
"""

import numpy as np

from app.ml import difficulty as dm
from tests.difficulty_corpus import PAPER_QUESTIONS, make_questions

SEED = 42
FOLDS = 4


def main():
    questions = make_questions(14)
    X = dm.feature_matrix([dm.extract_features(text) for text, _ in questions])
    labels = [level for _, level in questions]

    results, best, classes = dm.evaluate_candidates(
        X, labels, [np.nan] * len(labels), FOLDS, SEED)
    metrics = dm.detailed_metrics(results, best, labels, classes, FOLDS)

    print(f"\n{len(labels)} synthetic questions, {FOLDS}-fold stratified "
          f"cross-validation, seed {SEED}\n")
    print(f"{'Algorithm':<32}{'Macro-F1':>10}{'Accuracy':>10}{'Within 1':>10}")
    print("-" * 62)

    for row in dm.comparison_table(results, best):
        m = row["metrics"]
        mark = "  <- selected" if row["selected"] else ("  (baseline)" if row["is_baseline"] else "")
        print(f"{row['label']:<32}{m['macro_f1']:>10.3f}{m['accuracy']:>10.3f}"
              f"{m['within_one_level']:>10.3f}{mark}")

    print(f"\nImprovement over majority baseline: {metrics['improvement_over_majority']:+.3f}")
    print(f"Improvement over verb-rule baseline: {metrics['improvement_over_rule']:+.3f}")

    matrix = metrics["confusion_matrix"]
    print("\nConfusion matrix (rows = true level, columns = predicted):")
    print(" " * 10 + "".join(f"{name:>10}" for name in matrix["labels"]))
    for name, row in zip(matrix["labels"], matrix["matrix"]):
        print(f"{name:<10}" + "".join(f"{value:>10}" for value in row))

    model = dm.fit_final(best, X, labels, SEED)
    importance = dm.feature_importances(model)

    if importance:
        print("\nMost useful features:")
        for item in importance[:6]:
            print(f"  {item['feature']:<18}{item['importance']:.3f}")

    print("\nNew past-paper style questions:")
    for expected, (text, marks) in PAPER_QUESTIONS.items():
        features = dm.feature_matrix([dm.extract_features(text, marks)])
        out = dm.predict_levels(model, features)[0]
        print(f"  expected {expected:<7} -> predicted {out['difficulty']:<7}"
              f"({out['probability']:.2f})  {text[:58]}")


if __name__ == "__main__":
    main()
