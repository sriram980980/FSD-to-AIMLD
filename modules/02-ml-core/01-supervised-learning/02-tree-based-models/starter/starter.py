# Dependencies: scikit-learn>=1.3, xgboost>=2.0, pandas>=2.0
# Node: 2.1.2 — Tree-Based Models & Ensembles
# Run: python starter.py

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from typing import Tuple
import os


# ── Implemented helpers ──────────────────────────────────────────────────────


def load_wine_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load Wine dataset, stratify-split 80/20, print class distribution, return splits."""
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )
    print(f"Classes     : {list(data.target_names)}")
    print(f"Features    : {len(data.feature_names)}")
    print(f"Train size  : {len(X_train)} | Test size: {len(X_test)}")
    counts = np.bincount(y_train)
    for i, name in enumerate(data.target_names):
        print(f"  Train class '{name}': {counts[i]} samples")
    return X_train, X_test, y_train, y_test, list(data.feature_names)


def evaluate_model(
    name: str,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Print labeled accuracy + classification report + confusion matrix; return accuracy."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n{'─' * 50}")
    print(f"Model       : {name}")
    print(f"Accuracy    : {accuracy:.4f}")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["class_0", "class_1", "class_2"],
        )
    )
    cm = confusion_matrix(y_test, predictions)
    print(f"Confusion Matrix:\n{cm}")
    return accuracy


def print_top_features(
    model,
    feature_names: list[str],
    top_n: int = 5,
    model_name: str = "Model",
) -> None:
    """Print top_n features by Gini/gain importance in descending order."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    print(f"\nTop {top_n} features — {model_name}:")
    for rank, idx in enumerate(indices, start=1):
        print(f"  {rank}. {feature_names[idx]:<30} importance = {importances[idx]:.4f}")


def export_wine_csv(output_path: str = "data/wine.csv") -> None:
    """Export the Wine dataset to CSV so the Java starter can load it via Tribuo."""
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = [data.target_names[t] for t in data.target]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wine CSV exported → {output_path} ({len(df)} rows)")


# ── Student stubs ─────────────────────────────────────────────────────────────


def train_decision_tree_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> DecisionTreeClassifier:
    """Train a single, unpruned Decision Tree and return the fitted model.

    Requirements:
    - Use DecisionTreeClassifier with random_state=42 and no depth limit.
    - Fit on (X_train, y_train) and return the fitted estimator.
    """
    raise NotImplementedError("TODO: implement this")


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """Train a Random Forest with 200 trees and OOB scoring enabled.

    Requirements:
    - Use RandomForestClassifier with n_estimators=200, oob_score=True,
      random_state=42, and n_jobs=-1.
    - Fit on (X_train, y_train).
    - After fitting, print the OOB score with a label:
        print(f"OOB Score   : {model.oob_score_:.4f}")
    - Return the fitted estimator.
    """
    raise NotImplementedError("TODO: implement this")


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> XGBClassifier:
    """Train an XGBoost classifier with 300 rounds and learning_rate=0.05.

    Requirements:
    - Use XGBClassifier with n_estimators=300, learning_rate=0.05, max_depth=4,
      random_state=42, and verbosity=0.
    - Fit on (X_train, y_train) and return the fitted estimator.
    """
    raise NotImplementedError("TODO: implement this")


def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[RandomForestClassifier, dict]:
    """Run RandomizedSearchCV over RF hyperparameters; return (best_estimator, best_params).

    Search space:
        n_estimators     : [100, 200, 300, 500]
        max_depth        : [None, 5, 10, 20]
        min_samples_leaf : [1, 2, 4]
        max_features     : ["sqrt", "log2"]

    Requirements:
    - Use RandomizedSearchCV with n_iter=20, cv=5, scoring="accuracy", random_state=42,
      and n_jobs=-1.
    - Fit the search on (X_train, y_train).
    - After fitting, print best params and best CV accuracy with labels:
        print(f"Best params : {search.best_params_}")
        print(f"Best CV accuracy : {search.best_score_:.4f}")
    - Return (search.best_estimator_, search.best_params_).
    """
    raise NotImplementedError("TODO: implement this")


# ── Entrypoint ────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 50)
    print("Node 2.1.2 — Tree-Based Models & Ensembles")
    print("=" * 50)

    # Task 1 — Load and inspect
    X_train, X_test, y_train, y_test, feature_names = load_wine_dataset()

    # Export CSV for Java starter (runs silently if data/ already exists)
    export_wine_csv()

    # Task 2 — Baseline: single decision tree
    print("\n[Task 2] Decision Tree Baseline")
    dt_model = train_decision_tree_baseline(X_train, y_train)
    dt_accuracy = evaluate_model("Decision Tree (baseline)", dt_model, X_test, y_test)

    # Task 3 — Random Forest with OOB score
    print("\n[Task 3] Random Forest")
    rf_model = train_random_forest(X_train, y_train)
    rf_accuracy = evaluate_model("Random Forest (200 trees)", rf_model, X_test, y_test)
    print_top_features(rf_model, feature_names, model_name="Random Forest")

    # Task 4 — XGBoost + feature importance comparison
    print("\n[Task 4] XGBoost")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_accuracy = evaluate_model("XGBoost (300 rounds)", xgb_model, X_test, y_test)
    print_top_features(xgb_model, feature_names, model_name="XGBoost")

    # Task 5 — Hyperparameter tuning
    print("\n[Task 5] Tuning Random Forest with RandomizedSearchCV")
    tuned_rf, best_params = tune_random_forest(X_train, y_train)
    tuned_accuracy = evaluate_model("Tuned Random Forest", tuned_rf, X_test, y_test)

    # Task 6 — Final comparison table
    print(f"\n{'=' * 50}")
    print("Final Accuracy Comparison")
    print(f"  Decision Tree  : {dt_accuracy:.4f}")
    print(f"  Random Forest  : {rf_accuracy:.4f}")
    print(f"  XGBoost        : {xgb_accuracy:.4f}")
    print(f"  Tuned RF       : {tuned_accuracy:.4f}")
    ensemble_wins = rf_accuracy > dt_accuracy and xgb_accuracy > dt_accuracy
    print(f"  Ensembles beat baseline : {ensemble_wins}")


if __name__ == "__main__":
    main()
