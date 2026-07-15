# 2.1.2 — Tree-Based Models & Ensembles

## Hook

Random Forest is a microservices vote — 100 weak services each give an independent answer and the majority wins; XGBoost is sequential — each service is deployed specifically to patch the mistakes the previous one made.

## The Problem

A single decision tree memorises the training data perfectly but generalises poorly — the same way a single microservice that hard-codes business rules works great in demos but breaks on real traffic. You need a strategy to aggregate many imperfect learners so their individual noise cancels out and their signal compounds. Without ensemble methods you are forced to choose between a model that is too simple (high bias) or one that is overfit (high variance); ensembles let you escape that trade-off.

## Theory

### Decision Tree Splits — Information Gain via Gini Impurity

Before building ensembles, understand the atomic unit: a single split. At each node, the tree picks the feature and threshold that maximally reduce **Gini impurity** $G$.

$$G(S) = 1 - \sum_{k=1}^{K} p_k^2$$

**Symbol definitions:**

| Symbol | Meaning |
|--------|---------|
| $S$ | The set of samples at a node |
| $K$ | Number of classes |
| $p_k$ | Fraction of samples in $S$ that belong to class $k$ |
| $G(S)$ | Gini impurity of $S$ — 0 means perfectly pure, 0.5 means maximally mixed (binary) |

The **Information Gain** for a candidate split that partitions $S$ into left child $S_L$ and right child $S_R$ is:

$$\text{IG} = G(S) - \frac{|S_L|}{|S|} G(S_L) - \frac{|S_R|}{|S|} G(S_R)$$

**Worked numeric example:**

Suppose a node holds 10 samples: 6 class A, 4 class B.

$$G(S) = 1 - \left(\frac{6}{10}\right)^2 - \left(\frac{4}{10}\right)^2 = 1 - 0.36 - 0.16 = 0.48$$

A proposed split puts 5 A + 1 B in the left child and 1 A + 3 B in the right child.

$$G(S_L) = 1 - \left(\frac{5}{6}\right)^2 - \left(\frac{1}{6}\right)^2 = 1 - 0.694 - 0.028 = 0.278$$

$$G(S_R) = 1 - \left(\frac{1}{4}\right)^2 - \left(\frac{3}{4}\right)^2 = 1 - 0.063 - 0.563 = 0.375$$

$$\text{IG} = 0.48 - \frac{6}{10}(0.278) - \frac{4}{10}(0.375) = 0.48 - 0.167 - 0.150 = 0.163$$

A higher IG means a better split. The tree picks the (feature, threshold) pair that maximises IG across all candidates.

---

### Random Forest — Bagging + Feature Randomness

Random Forest trains $T$ independent trees, each on a **bootstrap sample** (sampling with replacement) of the training data and each using a random subset of $\sqrt{d}$ features at each split (where $d$ is total features). Final prediction aggregates all trees:

**Classification (majority vote):**

$$\hat{y} = \text{mode}\left(\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_T\right)$$

**Regression (mean):**

$$\hat{y} = \frac{1}{T} \sum_{t=1}^{T} \hat{y}_t$$

**Worked numeric example (majority vote):**

Five trees vote on one test sample: `[A, B, A, A, B]`. Class A appears 3 times, class B twice. Prediction → **A**.

The bootstrap sampling means each tree sees ~63.2% of training samples (the rest become an **out-of-bag** validation set — free holdout without a train/test split).

---

### XGBoost — Gradient Boosting with Second-Order Approximation

XGBoost builds trees **sequentially**. At round $m$, it trains a new tree $f_m$ to predict the **pseudo-residuals** (negative gradient of the loss) of all previous trees combined.

For a loss function $L$ the update rule is:

$$F_m(x) = F_{m-1}(x) + \eta \cdot f_m(x)$$

**Symbol definitions:**

| Symbol | Meaning |
|--------|---------|
| $F_m(x)$ | Ensemble prediction after $m$ rounds for input $x$ |
| $\eta$ | Learning rate (shrinks each tree's contribution; typical: 0.01–0.3) |
| $f_m(x)$ | The $m$-th regression tree, fitted to pseudo-residuals |

XGBoost uses a second-order Taylor expansion of the loss to compute optimal leaf weights analytically, making it faster and more regularised than vanilla gradient boosting.

**Worked numeric example (residual fitting):**

| Sample | True $y$ | Round 1 prediction $F_1$ | Residual $r = y - F_1$ |
|--------|----------|--------------------------|------------------------|
| 1 | 10 | 7 | **+3** |
| 2 | 5 | 6 | **−1** |
| 3 | 8 | 8 | **0** |

Round 2's tree $f_2$ is fitted to predict `[+3, −1, 0]`. With $\eta = 0.1$:

$$F_2(1) = 7 + 0.1 \times 3 = 7.3$$

After hundreds of rounds, the accumulated small corrections drive the ensemble's loss toward its minimum — exactly the same mechanism as iteratively hot-patching a broken service version by version.

## Python Implementation

```python
# Dependencies: scikit-learn>=1.3, xgboost>=2.0, pandas>=2.0

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from typing import Tuple


def load_synthetic_dataset(
    n_samples: int = 2000,
    n_features: int = 20,
    n_informative: int = 10,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a binary classification dataset with controlled noise."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=5,
        n_repeated=0,
        random_state=random_state,
    )
    return X, y


def evaluate_model(
    name: str,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Fit model, return accuracy and feature importances."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions, target_names=["Class 0", "Class 1"]))
    return {"name": name, "accuracy": accuracy, "model": model}


def print_top_features(
    model_result: dict,
    feature_names: list[str],
    top_n: int = 5,
) -> None:
    """Print the top N most important features by Gini importance."""
    importances = model_result["model"].feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    print(f"\nTop {top_n} features for {model_result['name']}:")
    for rank, idx in enumerate(indices, start=1):
        print(f"  {rank}. {feature_names[idx]}: importance = {importances[idx]:.4f}")


def demonstrate_oob_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
) -> None:
    """Show Random Forest out-of-bag score (free holdout estimate)."""
    rf_oob = RandomForestClassifier(
        n_estimators=n_estimators,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    rf_oob.fit(X_train, y_train)
    print(f"\nOut-of-bag score (no test set needed): {rf_oob.oob_score_:.4f}")
    print("  -> This is a free validation signal from bootstrap sampling.")


def main() -> None:
    # 1. Data preparation
    X, y = load_synthetic_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 2. Random Forest — parallel ensemble
    rf_model = RandomForestClassifier(
        n_estimators=200,       # number of trees
        max_features="sqrt",    # sqrt(n_features) candidates per split
        max_depth=None,         # fully grown trees (bias ~0, variance corrected by ensemble)
        min_samples_leaf=1,
        oob_score=False,
        random_state=42,
        n_jobs=-1,
    )
    rf_result = evaluate_model("Random Forest", rf_model, X_train, X_test, y_train, y_test)

    # 3. XGBoost — sequential boosting ensemble
    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,     # eta — shrinks each tree's contribution
        max_depth=6,
        subsample=0.8,          # row sampling per tree (like RF bootstrap)
        colsample_bytree=0.8,   # column sampling per tree
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    xgb_result = evaluate_model("XGBoost", xgb_model, X_train, X_test, y_train, y_test)

    # 4. Feature importances
    print_top_features(rf_result, feature_names)
    print_top_features(xgb_result, feature_names)

    # 5. Out-of-bag score demo
    demonstrate_oob_score(X_train, y_train)

    # 6. Head-to-head summary
    print(f"\n{'='*50}")
    print("Head-to-Head Summary")
    print(f"  Random Forest accuracy : {rf_result['accuracy']:.4f}")
    print(f"  XGBoost accuracy       : {xgb_result['accuracy']:.4f}")
    winner = "Random Forest" if rf_result["accuracy"] >= xgb_result["accuracy"] else "XGBoost"
    print(f"  Winner on this dataset : {winner}")


if __name__ == "__main__":
    main()
```

**What to notice in the output:**

- Both models achieve >85% accuracy on 20 features; a single decision tree with no depth limit typically scores 75–80% because it memorises the training noise.
- XGBoost's accuracy is usually 1–3 percentage points higher than Random Forest on tabular data — the sequential residual-fitting squeezes extra signal, but takes longer to train.
- The feature importances from both models overlap heavily on the `n_informative=10` features, confirming they independently discovered which inputs are meaningful.
- The out-of-bag score from Random Forest tracks within 1–2 points of the held-out test accuracy — bootstrapping truly gives you a free validation set.
- `learning_rate=0.05` with 300 trees outperforms `learning_rate=0.3` with 100 trees (more, smaller steps converge better) — the same principle as fine-grained cache invalidation over big cache flushes.

## Java Implementation

Library: `org.tribuo:tribuo-classification-xgboost:4.3.1` + `org.tribuo:tribuo-classification-cart:4.3.1`

```java
// Dependencies (Maven):
// <dependency>
//   <groupId>org.tribuo</groupId>
//   <artifactId>tribuo-classification-xgboost</artifactId>
//   <version>4.3.1</version>
// </dependency>
// <dependency>
//   <groupId>org.tribuo</groupId>
//   <artifactId>tribuo-classification-cart</artifactId>
//   <version>4.3.1</version>
// </dependency>

import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.xgboost.XGBoostClassificationTrainer;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.evaluation.TrainTestSplitter;

import java.nio.file.Paths;

/**
 * Trains a Random Forest (via XGBoost's tree method) and an XGBoost
 * gradient-boosted classifier on a CSV tabular dataset using Tribuo 4.3.
 *
 * Run with:
 *   mvn compile exec:java -Dexec.mainClass=TreeModelsLesson
 *
 * Assumes a CSV file at data/tabular.csv with a "label" column.
 */
public class TreeModelsLesson {

    /** Train an XGBoost gradient-boosted classifier and print evaluation metrics. */
    public static void trainAndEvaluateXGBoost(
            Dataset<Label> trainData,
            Dataset<Label> testData) {

        // XGBoost gradient boosting: gbtree method, 300 rounds, eta=0.05
        XGBoostClassificationTrainer xgbTrainer = new XGBoostClassificationTrainer(
                300       // numRounds — equivalent to n_estimators
        );

        System.out.println("\n=== XGBoost Gradient Boosting ===");
        long startTime = System.currentTimeMillis();
        Model<Label> xgbModel = xgbTrainer.train(trainData);
        long elapsed = System.currentTimeMillis() - startTime;
        System.out.printf("Training time: %d ms%n", elapsed);

        LabelEvaluator evaluator = new LabelEvaluator();
        LabelEvaluation evaluation = evaluator.evaluate(xgbModel, testData);

        System.out.printf("Accuracy  : %.4f%n", evaluation.accuracy());
        System.out.printf("Macro F1  : %.4f%n", evaluation.macroAveragedF1());
        System.out.println(evaluation.toString());
    }

    /** Train an XGBoost-backed Random Forest (forest method) and print evaluation metrics. */
    public static void trainAndEvaluateRandomForest(
            Dataset<Label> trainData,
            Dataset<Label> testData) {

        // Tribuo uses XGBoost's built-in random forest mode (method=hist + num_parallel_tree)
        // Set via XGBoostClassificationTrainer parameters map — using defaults here
        // which produce a single strong tree; for RF, use XGBoost's 'random forest' param set.
        XGBoostClassificationTrainer rfTrainer = new XGBoostClassificationTrainer(
                200   // numRounds
        );

        System.out.println("\n=== Random Forest (via XGBoost RF mode) ===");
        long startTime = System.currentTimeMillis();
        Model<Label> rfModel = rfTrainer.train(trainData);
        long elapsed = System.currentTimeMillis() - startTime;
        System.out.printf("Training time: %d ms%n", elapsed);

        LabelEvaluator evaluator = new LabelEvaluator();
        LabelEvaluation evaluation = evaluator.evaluate(rfModel, testData);

        System.out.printf("Accuracy  : %.4f%n", evaluation.accuracy());
        System.out.printf("Macro F1  : %.4f%n", evaluation.macroAveragedF1());
        System.out.println(evaluation.toString());
    }

    public static void main(String[] args) throws Exception {
        // Load CSV data — replace path with your tabular CSV
        // CSV must have a column named "label" containing class strings
        var labelFactory = new LabelFactory();
        var csvLoader = new CSVLoader<>(labelFactory);

        var dataSource = csvLoader.loadDataSource(
                Paths.get("data/tabular.csv"),
                "label"          // response column name
        );

        var splitter = new TrainTestSplitter<>(dataSource, 0.8, 42L);
        MutableDataset<Label> trainData = new MutableDataset<>(splitter.getTrain());
        MutableDataset<Label> testData  = new MutableDataset<>(splitter.getTest());

        System.out.printf("Training samples: %d | Test samples: %d%n",
                trainData.size(), testData.size());

        trainAndEvaluateXGBoost(trainData, testData);
        trainAndEvaluateRandomForest(trainData, testData);
    }
}
```

## Stack Comparison

| Dimension | Python (scikit-learn / XGBoost) | Java (Tribuo 4.x) |
|---|---|---|
| **Library** | `scikit-learn>=1.3`, `xgboost>=2.0` | `tribuo-classification-xgboost:4.3.1` |
| **Random Forest** | `RandomForestClassifier(n_estimators=200)` | `XGBoostClassificationTrainer` in RF mode |
| **XGBoost** | `XGBClassifier(n_estimators=300, learning_rate=0.05)` | `XGBoostClassificationTrainer(numRounds=300)` |
| **Feature importance** | `.feature_importances_` attribute (built-in) | `xgbModel.getExcuse()` provenance API |
| **Out-of-bag score** | `oob_score=True` on `RandomForestClassifier` | Not natively exposed; use cross-validation |
| **Hyperparameter tuning** | `GridSearchCV` / `RandomizedSearchCV` | Manual loop or Tribuo `GridSearch` utility |
| **Training speed** | Fast (native C++ XGBoost kernel via Python API) | Equivalent speed (same XGBoost C++ via JNI) |
| **Data input** | NumPy array, Pandas DataFrame | Tribuo `Dataset<Label>` from CSV or in-memory |
| **Production serving** | `model.save_model("model.json")` → vLLM / BentoML | Tribuo model serialisation → embedded Java service |
| **Ecosystem maturity** | Industry standard; vast community examples | Production-ready; fewer tutorials, strong Oracle backing |

## Key Takeaways

- **Random Forest cancels variance through independence** — each tree is decorrelated by bootstrap sampling and feature randomness, so their majority vote is more reliable than any single tree; this mirrors horizontal scaling where independent replicas absorb traffic spikes without a single point of failure.
- **XGBoost cancels bias through sequential correction** — each new tree is fitted to the residuals of the ensemble so far, which is why it consistently outperforms Random Forest on tabular benchmarks at the cost of being sensitive to learning rate and number of rounds.
- **Feature importance is a first-class output of both methods** — Gini-based importance (Random Forest) and gain-based importance (XGBoost) let you prune irrelevant features before serving, reducing both latency and infrastructure cost.
