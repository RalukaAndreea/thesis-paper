

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, permutation_test_score
)
from sklearn.metrics import accuracy_score

from pipeline import (
    _load_pathogenicity_training_data,
)


def test_significance(name, X_arr, y_arr, model, seed, n_permutations=100):
    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print(f"{'=' * 55}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=seed, stratify=y_arr
    )

    baseline = max((y_arr == 0).mean(), (y_arr == 1).mean())
    print(f"\n  Majority-class baseline: {baseline:.4f}")

    # 1. 5-fold CV accuracy
    cv_accs = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"  Model 5-fold CV:        {cv_accs.mean():.4f} ± {cv_accs.std():.4f}")

    # 2. T-test: model > baseline?
    t, p = stats.ttest_1samp(cv_accs, baseline)
    p_one = p / 2
    print(f"\n  T-test (model > baseline):")
    print(f"    t = {t:.4f}, p = {p_one:.6f}  {'✓ significant' if p_one < 0.05 else '✗ not significant'}")

    # 3. Binomial test on hold-out
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    n_correct = (y_pred == y_test).sum()
    n_total = len(y_test)
    binom = stats.binomtest(n_correct, n_total, baseline, alternative="greater")
    print(f"\n  Binomial test (hold-out {n_correct}/{n_total}):")
    print(f"    accuracy = {n_correct/n_total:.4f}, p = {binom.pvalue:.2e}  {'✓ significant' if binom.pvalue < 0.05 else '✗ not significant'}")

    probabilities = model1.predict_proba(X1.values.astype(float))

    # Example: Print the predicted probabilities
    print(probabilities)

    # 4. Permutation test
    print(f"\n  Permutation test ({n_permutations} permutations)...")
    score, perm_scores, perm_p = permutation_test_score(
        model, X_train, y_train, cv=cv, scoring="accuracy",
        n_permutations=n_permutations, random_state=seed, n_jobs=1
    )
    print(f"    true score = {score:.4f}, random = {perm_scores.mean():.4f} ± {perm_scores.std():.4f}")
    print(f"    p = {perm_p:.4f}  {'✓ significant' if perm_p < 0.05 else '✗ not significant'}")


if __name__ == "__main__":
    # Stage 1
    returned_values = _load_pathogenicity_training_data()
    print(len(returned_values))  # Check the number of returned values

    X1, y1, _ = _load_pathogenicity_training_data()
    model1 = RandomForestClassifier(
        max_depth=12, min_samples_leaf=4, min_samples_split=2,
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    )
    test_significance("Stage 1: Pathogenicity", X1.values.astype(float), y1.values, model1, seed=42)

    print()
