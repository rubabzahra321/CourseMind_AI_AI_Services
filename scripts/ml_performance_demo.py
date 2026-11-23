"""
Demo of the performance model: trains on SYNTHETIC practice histories, tests on
the LATER period, and prints the comparison, calibration and feature weights.

Run from the ai-service folder (no database or API keys needed):

    python -m scripts.ml_performance_demo

The simulated students learn, forget and find some questions harder (see
tests/performance_corpus.py). The numbers show the pipeline works; they are NOT
results on your real students. For those, train from the app (admin route
POST /admin/ml/performance/train) and read the model card from
GET /admin/ml/performance/status.
"""

from app.ml import performance as pm
from tests.performance_corpus import simulate, to_histories

SEED = 42
TEST_FRACTION = 0.30


def main():
    attempts = simulate(n_students=12, topics_per_student=3, attempts_per_topic=40, seed=7)
    rows, y, t = pm.build_samples(to_histories(attempts))
    X = pm.feature_matrix(rows)

    results, best, split = pm.evaluate_candidates(X, y, t, TEST_FRACTION, SEED)
    metrics = pm.detailed_metrics(results, best, split)

    print(f"\n{len(attempts)} simulated answers -> {len(y)} training examples")
    print(f"Trained on the earlier {split['train']}, tested on the later {split['test']} "
          f"(base rate of correct answers: {split['base_rate_train']:.2f})\n")
    print(f"{'Algorithm':<44}{'Skill':>8}{'AUC':>8}{'LogLoss':>9}{'Brier':>8}{'CalErr':>8}")
    print("-" * 85)

    for row in pm.comparison_table(results, best):
        m = row["metrics"]
        mark = "  <- selected" if row["selected"] else ("  (baseline)" if row["is_baseline"] else "")
        print(f"{row['label']:<44}{m['skill']:>8.3f}{m['auc']:>8.3f}{m['log_loss']:>9.3f}"
              f"{m['brier']:>8.3f}{row['calibration_error']:>8.3f}{mark}")

    print("\nSkill: 0 = no better than always guessing the average, higher is better.")
    print(f"Improvement over the best baseline: {metrics['improvement_over_best_baseline']:+.3f} "
          f"(AUC {metrics['auc_improvement_over_best_baseline']:+.3f})")

    print("\nCalibration of the selected model (do predicted chances come true?):")
    print(f"{'Predicted range':<18}{'Samples':>8}{'Mean predicted':>16}{'Observed':>10}")
    for bin_ in metrics["calibration"]:
        low, high = bin_["range"]
        print(f"{low:.1f} - {high:.1f}{'':<8}{bin_['samples']:>8}"
              f"{bin_['mean_predicted']:>16.3f}{bin_['observed_rate']:>10.3f}")

    model = pm.fit_final(best, X, y, SEED)
    importance = pm.feature_importance(model)

    if importance:
        print("\nWhat moves the prediction:")
        for item in importance[:6]:
            value = item.get("weight", item.get("importance"))
            print(f"  {item['feature']:<22}{value:>8.3f}  {item.get('effect', '')}")


if __name__ == "__main__":
    main()
