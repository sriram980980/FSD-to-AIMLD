# Assignment 2.1.2 — Train/Tune Classifier on Tabular Dataset

## Objective

Train a Decision Tree baseline, a Random Forest, and an XGBoost classifier on the Wine dataset, then use `RandomizedSearchCV` to tune the Random Forest — proving that ensemble methods outperform a single tree and that hyperparameter search pushes accuracy even further.

## Background

The lesson showed that a single decision tree overfits by memorising training noise, while Random Forest corrects this through bootstrap sampling and majority vote, and XGBoost corrects it through sequential residual fitting. Both ensemble strategies consistently outperform a single tree on real tabular data — this assignment makes you measure that gap directly. Refer to the **Theory** and **Python Implementation** sections of `lesson.md` before starting.

## Setup

```bash
pip install scikit-learn>=1.3 xgboost>=2.0 pandas>=2.0
```

For the Java path, generate `data/wine.csv` first by running the Python starter, then compile with Maven:

```bash
# Java dependency (pom.xml)
# org.tribuo:tribuo-classification-xgboost:4.3.1
mvn compile exec:java -Dexec.mainClass=StarterAssignment
```

## Tasks

1. **Load and inspect the dataset.** Call `load_wine_dataset()`, which stratifies an 80/20 train/test split and prints class distribution. Confirm the output shows 3 classes, 13 features, 142 training samples, and 36 test samples.

2. **Train a Decision Tree baseline.** Implement `train_decision_tree_baseline()` using `DecisionTreeClassifier(random_state=42)` with no depth limit. Fit it on training data, then call `evaluate_model()` to print accuracy and the classification report. Record this number — every subsequent model must beat it.

3. **Train a Random Forest.** Implement `train_random_forest()` using `RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)`. After fitting, print the OOB score with a label (`OOB Score: <value>`). Confirm the test accuracy is higher than the Decision Tree baseline.

4. **Train XGBoost.** Implement `train_xgboost()` using `XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42, verbosity=0)`. After evaluating, call `print_top_features()` for both the Random Forest and XGBoost models. Confirm that at least 3 of the top-5 features overlap between the two models — both must discover the same informative signals.

5. **Tune the Random Forest.** Implement `tune_random_forest()` using `RandomizedSearchCV` with the following search space, `n_iter=20`, `cv=5`, `scoring="accuracy"`, and `random_state=42`:

   | Parameter | Values |
   |---|---|
   | `n_estimators` | `[100, 200, 300, 500]` |
   | `max_depth` | `[None, 5, 10, 20]` |
   | `min_samples_leaf` | `[1, 2, 4]` |
   | `max_features` | `["sqrt", "log2"]` |

   Print best params and best CV accuracy with labels. Confirm the tuned model's test accuracy is ≥ the untuned Random Forest's test accuracy.

6. **Print the final comparison table.** `main()` already handles this step. Verify your console output shows all four model accuracies and the line `Ensembles beat baseline : True`.

## Expected Output

```
==================================================
Node 2.1.2 — Tree-Based Models & Ensembles
==================================================
Classes     : ['class_0', 'class_1', 'class_2']
Features    : 13
Train size  : 142 | Test size: 36
  Train class 'class_0': 47 samples
  Train class 'class_1': 57 samples
  Train class 'class_2': 38 samples

[Task 2] Decision Tree Baseline
──────────────────────────────────────────────────
Model       : Decision Tree (baseline)
Accuracy    : <value between 0.86 and 0.97>
...

[Task 3] Random Forest
OOB Score   : <value between 0.95 and 1.00>
──────────────────────────────────────────────────
Model       : Random Forest (200 trees)
Accuracy    : <value between 0.97 and 1.00>
...

[Task 4] XGBoost
──────────────────────────────────────────────────
Model       : XGBoost (300 rounds)
Accuracy    : <value between 0.97 and 1.00>
...

Top 5 features — Random Forest:
  1. <feature_name>                    importance = <value>
  ...

[Task 5] Tuning Random Forest with RandomizedSearchCV
Best params : {'n_estimators': <n>, 'min_samples_leaf': <n>, ...}
Best CV accuracy : <value between 0.95 and 1.00>
──────────────────────────────────────────────────
Model       : Tuned Random Forest
Accuracy    : <value between 0.97 and 1.00>
...

==================================================
Final Accuracy Comparison
  Decision Tree  : <value>
  Random Forest  : <value>
  XGBoost        : <value>
  Tuned RF       : <value>
  Ensembles beat baseline : True
```

## Evaluation Criteria

- [ ] `load_wine_dataset()` prints 3 classes, 13 features, train size 142, test size 36
- [ ] `train_decision_tree_baseline()` returns a fitted `DecisionTreeClassifier`; accuracy printed with label
- [ ] `train_random_forest()` prints OOB score with label; test accuracy exceeds Decision Tree baseline
- [ ] `train_xgboost()` fits and evaluates successfully; `print_top_features()` called for both RF and XGBoost
- [ ] Top-5 feature lists for RF and XGBoost share at least 3 features in common
- [ ] `tune_random_forest()` runs `RandomizedSearchCV`, prints best params and best CV accuracy with labels
- [ ] Final comparison table shows `Ensembles beat baseline : True`
- [ ] All four models evaluated on the same held-out test set (no data leakage)
- [ ] `starter.py` runs end-to-end with `python starter.py` and no errors

## Extension Challenge

Replace the Wine dataset with the [UCI Adult Income dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html) (`fetch_openml('adult', version=2)`), which has a mix of numeric and categorical features and a binary target (`<=50K` / `>50K`). Build a full `sklearn.pipeline.Pipeline` that includes:

1. A `ColumnTransformer` to one-hot encode categorical columns and scale numeric ones
2. An `XGBClassifier` as the final estimator
3. A `RandomizedSearchCV` tuning loop over at least 3 hyperparameters
4. A calibration step using `CalibratedClassifierCV` so predicted probabilities are well-calibrated (check with a reliability diagram)

Report ROC-AUC alongside accuracy. No starter code is provided for this challenge.
