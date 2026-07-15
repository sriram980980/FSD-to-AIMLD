# Assignment 1.2.1 — Build SciPy Statistical Model

## Objective

Build a Gaussian Naive Bayes classifier from scratch using SciPy and NumPy, proving you can apply the normal distribution and Bayes' theorem to a real multi-class classification problem.

## Background

The lesson covered two connected ideas: the normal distribution describes how feature values spread around a class mean, and Bayes' theorem converts that spread into a posterior classification score. Gaussian Naive Bayes stitches them together — it models each feature as $\mathcal{N}(\mu_c, \sigma_c^2)$ per class $c$, then applies Bayes to pick the most probable class. This is one of the fastest classifiers to implement from first principles and gives you direct hands-on experience with `scipy.stats.norm`. See `lesson.md` for the full derivation of both equations.

## Setup

```bash
pip install "numpy>=1.24" "scipy>=1.11" "matplotlib>=3.7" "scikit-learn>=1.3"
```

> `scikit-learn` is used only in Task 4 as a reference baseline; all classification logic in Tasks 1–3 must be implemented from scratch.

## Tasks

### Task 1 — Implement `gaussian_log_likelihood`

Open `starter/starter.py` and implement `gaussian_log_likelihood(x, mean, std)`.

- Use `scipy.stats.norm.logpdf(x, loc=mean, scale=std)`.
- Return a single `float`.
- Verify your implementation by running the script; the sanity-check block must print:

```
log p(x=0.0 | N(0, 1))  : -0.9189   (expect -0.9189)
log p(x=1.0 | N(0, 1))  : -1.4189   (expect -1.4189)
```

### Task 2 — Implement `naive_bayes_predict`

Implement `naive_bayes_predict(X, class_stats, log_class_priors)`.

For each sample and each class $c$, compute:

$$\log \hat{y}_c = \log P(c) + \sum_{f} \log p(x_f \mid \mu_{c,f},\, \sigma_{c,f})$$

Return `np.argmax` across classes for every sample. Use `gaussian_log_likelihood` from Task 1 — do not call `scipy` directly here.

### Task 3 — Implement `evaluate_accuracy`

Implement `evaluate_accuracy(y_true, y_pred)` as `np.mean(y_true == y_pred)`.

When combined with Task 2, the printed accuracy must fall between **0.93 and 1.00** on the Iris test set.

### Task 4 — Implement `sklearn_gnb_accuracy`

Implement `sklearn_gnb_accuracy(X_train, y_train, X_test, y_test)` using `sklearn.naive_bayes.GaussianNB`.

- Fit on the training set, predict on the test set, return accuracy as a `float`.
- Your scratch implementation (Task 3) must score within **0.10** of this baseline.

### Task 5 — Implement `sequential_bayes_update`

Implement `sequential_bayes_update(prior, likelihoods_given_h, likelihoods_given_not_h)`.

Apply Bayes' theorem iteratively — each posterior becomes the prior for the next step:

$$P(H \mid E_n) = \frac{P(E_n \mid H) \cdot P(H \mid E_{n-1})}{P(E_n \mid H) \cdot P(H \mid E_{n-1}) + P(E_n \mid \neg H) \cdot (1 - P(H \mid E_{n-1}))}$$

Return a list of posteriors, one per observation. After all 5 observations the total shift in $P(\text{Setosa})$ from the initial prior must be **≥ 0.30**.

### Task 6 — Generate the class-distribution plot

Run the completed script end-to-end. The call to `plot_class_distributions()` is already wired in `main()` — it just needs Tasks 1–5 to succeed first. Confirm the file `class_distributions.png` is written to disk and visually inspect that the three Gaussian curves per feature are non-overlapping for petal measurements and heavily overlapping for sepal measurements.

## Expected Output

```
============================================================
Task 1: gaussian_log_likelihood — sanity check
============================================================
  log p(x=0.0 | N(0, 1))  : -0.9189   (expect -0.9189)
  log p(x=1.0 | N(0, 1))  : -1.4189   (expect -1.4189)

============================================================
Task 2 & 3: Scratch Gaussian Naive Bayes — fit and predict
============================================================
  Class-conditional statistics (training set):
    Class 0 — sepal length (cm): mean=4.985  std=0.308
    Class 1 — sepal length (cm): mean=5.930  std=0.477
    Class 2 — sepal length (cm): mean=6.610  std=0.685
  Scratch GNB accuracy     : 0.XXXX   (expect 0.93–1.00)

============================================================
Task 4: sklearn GaussianNB comparison
============================================================
  sklearn GNB accuracy     : 0.XXXX   (expect 0.93–1.00)

============================================================
Task 5: Sequential Bayes — P(Setosa | petal_length > 2.45)
============================================================
  Initial prior P(Setosa)  : 0.3333
  After obs 1:  P(Setosa) 0.3333 → 0.XXXX
  After obs 2:  P(Setosa) 0.XXXX → 0.XXXX
  After obs 3:  P(Setosa) 0.XXXX → 0.XXXX
  After obs 4:  P(Setosa) 0.XXXX → 0.XXXX
  After obs 5:  P(Setosa) 0.XXXX → 0.XXXX

  Total shift in P(Setosa) : 0.XXXX   (expect ≥ 0.30)

============================================================
Bonus: Plotting class-conditional distributions
============================================================
Plot saved → class_distributions.png
```

> Exact accuracy values depend on the train/test split. The bounds above are the acceptance criteria.

## Evaluation Criteria

- [ ] `gaussian_log_likelihood(0.0, 0.0, 1.0)` returns a value within 0.001 of **-0.9189**
- [ ] `naive_bayes_predict` returns an array of shape `(n_test_samples,)` with integer class labels
- [ ] Scratch GNB accuracy on Iris test set is **between 0.93 and 1.00**
- [ ] sklearn GNB accuracy is **within 0.10** of the scratch accuracy
- [ ] `sequential_bayes_update` final posterior differs from initial prior by **≥ 0.30**
- [ ] `class_distributions.png` is generated and contains 6 Gaussian curves (2 features × 3 classes)

## Extension Challenge

Implement **full Bayesian model selection** between two competing hypotheses about the Iris dataset:

- $H_1$: The data for class 0 (Setosa) petal length is drawn from $\mathcal{N}(\mu=1.462, \sigma=0.174)$
- $H_2$: The data for class 0 petal length is drawn from $\mathcal{N}(\mu=2.0, \sigma=0.5)$ (a misspecified model)

For each sample in the class-0 test subset, compute the **Bayes factor** $K = P(\text{data} \mid H_1) / P(\text{data} \mid H_2)$ and accumulate the log evidence using `gaussian_log_likelihood`. Plot log $K$ as observations accumulate. Show that the log Bayes factor grows linearly with the number of samples under the correct model — this is the statistical argument for why well-calibrated priors win with more data.

No starter code provided. Use only NumPy, SciPy, and Matplotlib.
