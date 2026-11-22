"""
Demo of the topic classifier: compares the five algorithms with
cross-validation and prints a results table plus the confusion matrix.

Run from the ai-service folder (no database or API keys needed):

    python -m scripts.ml_demo

It uses the small demo question set in tests/ml_corpus.py. These are demo
numbers, NOT results on your real course data: for those, train from the app
and read the "model card" from GET /courses/:id/ml/status.
"""

from app.ml import topic_classifier as tc
from tests.ml_corpus import CORPUS, NEW_QUESTIONS

SEED = 42
FOLDS = 4


def main():
    texts, labels, sources = [], [], []

    for topic, questions in CORPUS.items():
        for question in questions:
            texts.append(question)
            labels.append(topic)
            sources.append("question")

    results, best, basis, classes = tc.evaluate_candidates(
        texts, labels, sources, FOLDS, SEED, 10)

    titles = {c: c for c in classes}
    metrics = tc.detailed_metrics(results, best, labels, titles, classes, basis, FOLDS)

    print(f"\n{len(texts)} labelled questions, {len(classes)} topics, "
          f"{FOLDS}-fold stratified cross-validation, seed {SEED}\n")
    print(f"{'Algorithm':<30}{'Macro-F1':>10}{'Accuracy':>10}{'Top-3':>8}")
    print("-" * 58)

    for row in tc.comparison_table(results, best):
        block = row["question_only"]
        mark = "  <- selected" if row["selected"] else ("  (baseline)" if row["is_baseline"] else "")
        print(f"{row['label']:<30}{block['macro_f1']:>10.3f}{block['accuracy']:>10.3f}"
              f"{block['top3_accuracy']:>8.3f}{mark}")

    print(f"\nImprovement over baseline: {metrics['improvement_over_baseline']:+.3f}")

    matrix = metrics["confusion_matrix"]
    print("\nConfusion matrix (rows = true topic, columns = predicted):")
    print(" " * 12 + "".join(f"{name:>10}" for name in matrix["labels"]))
    for name, row in zip(matrix["labels"], matrix["matrix"]):
        print(f"{name:<12}" + "".join(f"{value:>10}" for value in row))

    pipeline = tc.fit_final(best, texts, labels, FOLDS, SEED)

    print("\nNew questions:")
    for expected, question in NEW_QUESTIONS.items():
        topic, probability = tc.predict_top_k(pipeline, [question], 1)[0][0]
        print(f"  expected {expected:<8} -> predicted {topic:<8} ({probability:.2f})  {question[:60]}")


if __name__ == "__main__":
    main()
