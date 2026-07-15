# Dependencies: numpy>=1.24, scipy>=1.11, matplotlib>=3.7, scikit-learn>=1.3
# Node: 1.2.1 — Distributions & Bayesian Statistics
# Run: python starter.py

import numpy as np
from scipy.stats import norm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Implemented helpers — provided for you, do not modify
# ---------------------------------------------------------------------------


def load_and_split_data(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the Iris dataset and return stratified train/test splits."""
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def compute_class_stats(
    X: np.ndarray, y: np.ndarray
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Compute per-class, per-feature (mean, std) pairs from training data.

    Returns: {class_id: [(mean_f0, std_f0), (mean_f1, std_f1), ...]}
    Uses sample standard deviation (ddof=1) to match scipy assumptions.
    """
    classes = np.unique(y)
    stats: Dict[int, List[Tuple[float, float]]] = {}
    for c in classes:
        X_c = X[y == c]
        stats[int(c)] = [
            (float(np.mean(X_c[:, f])), float(np.std(X_c[:, f], ddof=1)))
            for f in range(X.shape[1])
        ]
    return stats


def plot_class_distributions(
    class_stats: Dict[int, List[Tuple[float, float]]],
    feature_names: List[str],
    output_path: str = "class_distributions.png",
) -> None:
    """Plot class-conditional Gaussian PDFs for the first two features and save to disk."""
    class_colors = ["#2196F3", "#4CAF50", "#FF5722"]
    class_names = ["Setosa", "Versicolor", "Virginica"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for feat_idx in range(2):
        ax = axes[feat_idx]
        for class_id, feat_list in class_stats.items():
            mu, sigma = feat_list[feat_idx]
            x_range = np.linspace(mu - 3.5 * sigma, mu + 3.5 * sigma, 300)
            ax.plot(
                x_range,
                norm.pdf(x_range, loc=mu, scale=sigma),
                label=class_names[class_id],
                color=class_colors[class_id],
                linewidth=2,
            )
        ax.set_title(f"Class-conditional PDF — {feature_names[feat_idx]}")
        ax.set_xlabel(feature_names[feat_idx])
        ax.set_ylabel("p(x | class)")
        ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"  Plot saved → {output_path}")


# ---------------------------------------------------------------------------
# Student stubs — implement each function below
# ---------------------------------------------------------------------------


def gaussian_log_likelihood(x: float, mean: float, std: float) -> float:
    """
    Return the log probability density of x under N(mean, std^2).

    Use scipy.stats.norm.logpdf — it handles the log internally with
    better numerical stability than computing log(norm.pdf(...)).

    Args:
        x    : observed feature value
        mean : class-conditional mean  (μ_c,f)
        std  : class-conditional std   (σ_c,f)

    Expected:
        gaussian_log_likelihood(0.0, 0.0, 1.0) → -0.9189
        gaussian_log_likelihood(1.0, 0.0, 1.0) → -1.4189
    """
    raise NotImplementedError("TODO: implement this")


def naive_bayes_predict(
    X: np.ndarray,
    class_stats: Dict[int, List[Tuple[float, float]]],
    log_class_priors: Dict[int, float],
) -> np.ndarray:
    """
    Classify each row of X using log-space Gaussian Naive Bayes.

    Algorithm (for each sample):
      1. For each class c:
             log_score[c] = log_class_priors[c]
                          + sum of gaussian_log_likelihood(x_f, mean_c_f, std_c_f)
                            over all features f
      2. Return argmax over classes.

    Working in log-space avoids floating-point underflow when multiplying
    many small probabilities — the same reason logits are preferred over
    raw softmax probabilities in deep learning frameworks.

    Args:
        X               : shape (n_samples, n_features) — test feature matrix
        class_stats     : {class_id: [(mean_f0, std_f0), ...]}
        log_class_priors: {class_id: log(P(class))}

    Returns:
        np.ndarray of shape (n_samples,) with predicted integer class labels
    """
    raise NotImplementedError("TODO: implement this")


def evaluate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Return the fraction of predictions that exactly match the true labels.

    One line: np.mean(y_true == y_pred).
    Expected range on Iris test set: 0.93–1.00.

    Args:
        y_true : ground-truth integer labels, shape (n_samples,)
        y_pred : predicted integer labels, shape (n_samples,)

    Returns:
        Accuracy as a float in [0.0, 1.0]
    """
    raise NotImplementedError("TODO: implement this")


def sklearn_gnb_accuracy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """
    Fit sklearn's GaussianNB on the training set and return test-set accuracy.

    Steps:
      1. Instantiate sklearn.naive_bayes.GaussianNB()
      2. Call .fit(X_train, y_train)
      3. Call .score(X_test, y_test) — returns accuracy directly
      4. Return the score as a float

    This is the reference baseline. Your scratch implementation (Task 2 & 3)
    must score within 0.10 of this value.
    """
    raise NotImplementedError("TODO: implement this")


def sequential_bayes_update(
    prior: float,
    likelihoods_given_h: List[float],
    likelihoods_given_not_h: List[float],
) -> List[float]:
    """
    Apply Bayes' theorem iteratively across multiple independent observations.
    Each posterior becomes the prior for the next step.

    Per-step formula:
        marginal  = P(E|H)*P(H) + P(E|¬H)*(1 - P(H))
        posterior = P(E|H)*P(H) / marginal

    Args:
        prior                  : initial P(H) before any evidence
        likelihoods_given_h    : [P(E_1|H), P(E_2|H), ...] — one per observation
        likelihoods_given_not_h: [P(E_1|¬H), P(E_2|¬H), ...] — one per observation

    Returns:
        List of posteriors [P(H|E_1), P(H|E_1,E_2), ...]

    Hint: after computing posterior, set prior = posterior before the next loop
    iteration — this chains the updates sequentially.
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# main — runs all tasks in order, prints labeled results
# ---------------------------------------------------------------------------


def main() -> None:
    iris = load_iris()
    feature_names: List[str] = iris.feature_names

    X_train, X_test, y_train, y_test = load_and_split_data()
    class_stats = compute_class_stats(X_train, y_train)

    # Compute log class priors from training label frequencies
    classes = np.unique(y_train)
    class_counts = {int(c): int(np.sum(y_train == c)) for c in classes}
    total = len(y_train)
    log_class_priors = {c: float(np.log(count / total)) for c, count in class_counts.items()}

    # ── Task 1: gaussian_log_likelihood sanity check ──────────────────────
    print("=" * 60)
    print("Task 1: gaussian_log_likelihood — sanity check")
    print("=" * 60)
    try:
        ll_0 = gaussian_log_likelihood(0.0, 0.0, 1.0)
        ll_1 = gaussian_log_likelihood(1.0, 0.0, 1.0)
        print(f"  log p(x=0.0 | N(0, 1))  : {ll_0:.4f}   (expect -0.9189)")
        print(f"  log p(x=1.0 | N(0, 1))  : {ll_1:.4f}   (expect -1.4189)")
    except NotImplementedError:
        print("  [NOT IMPLEMENTED] gaussian_log_likelihood")

    # ── Task 2 & 3: Scratch Gaussian Naive Bayes ─────────────────────────
    print()
    print("=" * 60)
    print("Task 2 & 3: Scratch Gaussian Naive Bayes — fit and predict")
    print("=" * 60)
    print("  Class-conditional statistics (training set):")
    for class_id, feat_list in class_stats.items():
        mu, sigma = feat_list[0]
        print(
            f"    Class {class_id} — {feature_names[0]}: "
            f"mean={mu:.3f}  std={sigma:.3f}"
        )
    try:
        y_pred = naive_bayes_predict(X_test, class_stats, log_class_priors)
        try:
            acc = evaluate_accuracy(y_test, y_pred)
            print(f"  Scratch GNB accuracy     : {acc:.4f}   (expect 0.93–1.00)")
        except NotImplementedError:
            print("  [NOT IMPLEMENTED] evaluate_accuracy")
    except NotImplementedError:
        print("  [NOT IMPLEMENTED] naive_bayes_predict")

    # ── Task 4: sklearn GaussianNB comparison ────────────────────────────
    print()
    print("=" * 60)
    print("Task 4: sklearn GaussianNB comparison")
    print("=" * 60)
    try:
        sklearn_acc = sklearn_gnb_accuracy(X_train, y_train, X_test, y_test)
        print(f"  sklearn GNB accuracy     : {sklearn_acc:.4f}   (expect 0.93–1.00)")
    except NotImplementedError:
        print("  [NOT IMPLEMENTED] sklearn_gnb_accuracy")

    # ── Task 5: Sequential Bayes update ──────────────────────────────────
    print()
    print("=" * 60)
    print("Task 5: Sequential Bayes — P(Setosa | petal_length > 2.45)")
    print("=" * 60)
    # Feature index 2 is petal length.
    # Threshold 2.45 cm is the canonical first split in the Iris decision tree.
    # Binary encoding: 1 → petal_length > 2.45 (evidence against Setosa)
    #                  0 → petal_length ≤ 2.45 (evidence for Setosa)
    observations = X_test[:5, 2]
    evidence_flags = [float(obs > 2.45) for obs in observations]
    # P(evidence=1 | Setosa) ≈ 0.02 — almost no Setosa has petal_length > 2.45
    # P(evidence=1 | ¬Setosa)≈ 0.98 — almost all non-Setosa have petal_length > 2.45
    p_e_given_h     = [0.02 if flag == 1.0 else 0.98 for flag in evidence_flags]
    p_e_given_not_h = [0.98 if flag == 1.0 else 0.02 for flag in evidence_flags]
    prior_setosa = float(class_counts[0] / total)
    print(f"  Initial prior P(Setosa)  : {prior_setosa:.4f}")
    print(f"  Petal lengths observed   : {[round(o, 2) for o in observations]}")
    try:
        posteriors = sequential_bayes_update(prior_setosa, p_e_given_h, p_e_given_not_h)
        prev = prior_setosa
        for i, post in enumerate(posteriors):
            print(f"  After obs {i + 1}:  P(Setosa) {prev:.4f} → {post:.4f}")
            prev = post
        delta = abs(posteriors[-1] - prior_setosa)
        print(f"\n  Total shift in P(Setosa) : {delta:.4f}   (expect ≥ 0.30)")
    except NotImplementedError:
        print("  [NOT IMPLEMENTED] sequential_bayes_update")

    # ── Bonus: class-distribution plot ───────────────────────────────────
    print()
    print("=" * 60)
    print("Bonus: Plotting class-conditional distributions")
    print("=" * 60)
    plot_class_distributions(class_stats, list(feature_names))


if __name__ == "__main__":
    main()
