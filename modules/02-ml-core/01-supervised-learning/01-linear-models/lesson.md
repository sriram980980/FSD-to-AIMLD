# 2.1.1 — Linear Models — Regression & Classification

## Hook

Logistic regression is a single middleware function — it takes your feature inputs, applies learned
weights, and squashes the result into a probability score, exactly like an auth middleware that
scores a token's validity before deciding whether to pass the request through.

## The Problem

You need a baseline predictor before reaching for deep learning. Without one, you cannot tell
whether your neural network is actually learning anything, or just adding complexity over a
trivially solvable problem. Linear models give you interpretable weights (which features matter),
bounded training time (seconds, not hours), and a loss function you can reason about — **MSE** for
continuous targets and **Cross-Entropy** for classification. Everything downstream in this course
builds on these two loss functions.

## Theory

### Linear Regression and Mean Squared Error

The model predicts a continuous value $\hat{y}$ as a weighted sum of inputs:

$$\hat{y} = \mathbf{w}^{T}\mathbf{x} + b$$

- $\mathbf{w} \in \mathbb{R}^d$ — the weight vector (one scalar per feature), the knobs the model learns
- $\mathbf{x} \in \mathbb{R}^d$ — one input sample (a row of your feature matrix $\mathbf{X}$)
- $b \in \mathbb{R}$ — the bias (intercept), the prediction when all features are zero
- $\hat{y} \in \mathbb{R}$ — the predicted value

The loss measures average squared residual across $n$ samples:

$$L_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- $y_i$ — true label for sample $i$
- $\hat{y}_i$ — model prediction for sample $i$

**Numeric worked example:**

| Sample | $y_i$ | $\hat{y}_i$ | $(y_i - \hat{y}_i)^2$ |
|--------|--------|------------|------------------------|
| 1      | 3.0    | 2.5        | $0.25$                 |
| 2      | 5.0    | 4.5        | $0.25$                 |

$$L_{\text{MSE}} = \frac{0.25 + 0.25}{2} = 0.2500$$

The closed-form solution that minimises $L_{\text{MSE}}$ in one step (no iteration needed):

$$\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

This is **Ordinary Least Squares (OLS)**. In practice you use `np.linalg.pinv` for numerical
stability when features are collinear.

---

### Logistic Regression and Cross-Entropy Loss

Linear regression outputs any real number. For binary classification you need a probability in
$[0, 1]$. The **sigmoid** function does exactly that:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- $z \in \mathbb{R}$ — the raw linear score $\mathbf{w}^T \mathbf{x} + b$, called the **logit**
- $\sigma(z) \in (0, 1)$ — probability that sample $\mathbf{x}$ belongs to the positive class

**Numeric worked example:**

$$\mathbf{w} = [0.5,\ {-0.3}], \quad \mathbf{x} = [2.0,\ 1.0], \quad b = 0.1$$

$$z = 0.5 \times 2.0 + (-0.3) \times 1.0 + 0.1 = 1.0 - 0.3 + 0.1 = 0.8$$

$$\sigma(0.8) = \frac{1}{1 + e^{-0.8}} = \frac{1}{1 + 0.4493} \approx 0.6900 \quad (69\%\ \text{probability of positive class})$$

The loss for binary classification is **Binary Cross-Entropy**:

$$L_{\text{CE}} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

- $y_i \in \{0, 1\}$ — true binary label
- $\hat{p}_i = \sigma(z_i)$ — predicted probability

**Why this loss?** It is the negative log-likelihood of a Bernoulli distribution. A confident wrong
prediction (e.g., $\hat{p} = 0.99$ when $y = 0$) incurs a very large loss, far more than MSE
would.

**Continuing the worked example** ($y = 1$, $\hat{p} \approx 0.6900$):

$$L_{\text{CE}} = -\log(0.6900) \approx 0.3711$$

The gradient with respect to weights is:

$$\nabla_{\mathbf{w}} L_{\text{CE}} = \frac{1}{n} \mathbf{X}^T (\hat{\mathbf{p}} - \mathbf{y})$$

This is what the gradient descent update uses each epoch.

## Python Implementation

```python
# Dependencies: numpy>=1.24, scikit-learn>=1.3, matplotlib>=3.7
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from typing import Tuple


# ── math primitives ────────────────────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Squash any real-valued array into (0, 1)."""
    return 1.0 / (1.0 + np.exp(-z))


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error — average squared residual."""
    return float(np.mean((y_true - y_pred) ** 2))


def cross_entropy_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    """Binary Cross-Entropy — clips probabilities to avoid log(0)."""
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))


# ── linear regression (closed-form OLS) ────────────────────────────────────────

def fit_linear_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form OLS: w* = (X^T X)^{-1} X^T y.  Returns [bias, w1, w2, ...]."""
    X_b = np.c_[np.ones(len(X)), X]          # prepend bias column of 1s
    return np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y


def predict_linear(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply learned weights to produce continuous predictions."""
    X_b = np.c_[np.ones(len(X)), X]
    return X_b @ weights


# ── logistic regression (gradient descent) ─────────────────────────────────────

def fit_logistic_gd(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.1,
    epochs: int = 200,
) -> np.ndarray:
    """Train binary logistic regression via batch gradient descent."""
    X_b = np.c_[np.ones(len(X)), X]
    weights = np.zeros(X_b.shape[1])          # start at zero — symmetric initialisation

    for epoch in range(epochs):
        y_prob = sigmoid(X_b @ weights)
        gradient = (1.0 / len(y)) * X_b.T @ (y_prob - y)
        weights -= learning_rate * gradient

        if epoch % 50 == 0:
            loss = cross_entropy_loss(y, y_prob)
            print(f"  Epoch {epoch:>3d} | Cross-Entropy Loss: {loss:.4f}")

    return weights


def predict_logistic(
    X: np.ndarray, weights: np.ndarray, threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (predicted_probabilities, binary_labels)."""
    X_b = np.c_[np.ones(len(X)), X]
    y_prob = sigmoid(X_b @ weights)
    y_pred = (y_prob >= threshold).astype(int)
    return y_prob, y_pred


# ── demos ──────────────────────────────────────────────────────────────────────

def demo_linear_regression() -> None:
    print("=" * 60)
    print("LINEAR REGRESSION — MSE Loss")
    print("=" * 60)

    # Numeric worked example from Theory
    y_true_ex = np.array([3.0, 5.0])
    y_pred_ex = np.array([2.5, 4.5])
    print(f"Worked example | y_true={y_true_ex}, y_pred={y_pred_ex}")
    print(f"MSE            : {mse_loss(y_true_ex, y_pred_ex):.4f}  (expected 0.2500)\n")

    # Synthetic dataset
    X, y = make_regression(n_samples=300, n_features=3, noise=20.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scratch OLS
    weights = fit_linear_ols(X_train, y_train)
    y_pred_scratch = predict_linear(X_test, weights)
    mse_scratch = mse_loss(y_test, y_pred_scratch)

    # scikit-learn reference
    sk = LinearRegression().fit(X_train, y_train)
    mse_sk = mean_squared_error(y_test, sk.predict(X_test))

    print(f"Dataset        : n=300, 3 features, noise=20")
    print(f"Scratch OLS    | Test MSE : {mse_scratch:.4f}")
    print(f"scikit-learn   | Test MSE : {mse_sk:.4f}")
    print(f"Difference     : {abs(mse_scratch - mse_sk):.6f}  (should be ≈ 0)")


def demo_logistic_regression() -> None:
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION — Cross-Entropy Loss")
    print("=" * 60)

    # Numeric worked example from Theory
    w = np.array([0.5, -0.3])
    x = np.array([2.0, 1.0])
    b = 0.1
    z = float(w @ x) + b
    prob = float(sigmoid(np.array([z]))[0])
    ce = cross_entropy_loss(np.array([1.0]), np.array([prob]))
    print(f"Worked example | w={w}, x={x}, b={b}")
    print(f"  z = w·x + b  : {z:.4f}")
    print(f"  σ(z)         : {prob:.4f}  (expected ≈ 0.6900)")
    print(f"  CE (y=1)     : {ce:.4f}  (expected ≈ 0.3711)\n")

    # Synthetic binary classification dataset
    X, y = make_classification(
        n_samples=600, n_features=2, n_redundant=0,
        n_informative=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Gradient descent training log:")
    weights = fit_logistic_gd(X_train, y_train, learning_rate=0.1, epochs=200)
    y_prob_scratch, y_pred_scratch = predict_logistic(X_test, weights)

    # scikit-learn reference
    sk = LogisticRegression(max_iter=500).fit(X_train, y_train)
    y_prob_sk = sk.predict_proba(X_test)[:, 1]
    y_pred_sk = sk.predict(X_test)

    print(f"\nDataset        : n=600, 2 informative features")
    print(f"Scratch LR     | Accuracy: {accuracy_score(y_test, y_pred_scratch):.4f} | "
          f"Log-Loss: {log_loss(y_test, y_prob_scratch):.4f}")
    print(f"scikit-learn   | Accuracy: {accuracy_score(y_test, y_pred_sk):.4f} | "
          f"Log-Loss: {log_loss(y_test, y_prob_sk):.4f}")


def main() -> None:
    demo_linear_regression()
    demo_logistic_regression()


if __name__ == "__main__":
    main()
```

**What to notice in the output:**

- The scratch OLS MSE and scikit-learn MSE differ by less than `0.000001` — both solve the same
  closed-form equation; any gap is floating-point rounding only.
- The gradient descent Cross-Entropy drops sharply in the first 50 epochs, then slows — this is
  typical: the model corrects large errors fast and fine-tunes later.
- Your scratch logistic regression accuracy should land within 1–2 percentage points of
  scikit-learn. The gap exists because scikit-learn uses L2 regularisation by default (`C=1.0`),
  which your scratch version omits.
- Both accuracy values should exceed 0.85 on this linearly separable dataset — a strong signal
  that a linear model is sufficient before reaching for anything more complex.

## Java Implementation

```java
// Maven dependencies:
// <dependency>
//   <groupId>org.tribuo</groupId>
//   <artifactId>tribuo-classification-liblinear</artifactId>
//   <version>4.3.1</version>
// </dependency>

import org.tribuo.Feature;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.liblinear.LibLinearClassificationTrainer;
import org.tribuo.datasource.ListDataSource;
import org.tribuo.impl.ListExample;
import org.tribuo.provenance.SimpleDataSourceProvenance;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class LinearModelsLesson {

    // ── pure math helpers (no dependencies) ─────────────────────────────────

    public static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public static double mseLoss(double[] yTrue, double[] yPred) {
        double sum = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            double diff = yTrue[i] - yPred[i];
            sum += diff * diff;
        }
        return sum / yTrue.length;
    }

    public static double crossEntropyLoss(double[] yTrue, double[] yProb) {
        double eps = 1e-12;
        double sum = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            double p = Math.max(eps, Math.min(1.0 - eps, yProb[i]));
            sum += yTrue[i] * Math.log(p) + (1.0 - yTrue[i]) * Math.log(1.0 - p);
        }
        return -sum / yTrue.length;
    }

    // ── numeric worked examples (mirrors Theory section) ─────────────────────

    static void workedExamples() {
        System.out.println("=".repeat(60));
        System.out.println("WORKED NUMERIC EXAMPLES");
        System.out.println("=".repeat(60));

        // MSE
        double[] yTrue = {3.0, 5.0};
        double[] yPred = {2.5, 4.5};
        System.out.printf("MSE | y_true=[3.0, 5.0] y_pred=[2.5, 4.5]%n");
        System.out.printf("    | MSE = %.4f  (expected 0.2500)%n%n", mseLoss(yTrue, yPred));

        // Sigmoid / Cross-Entropy
        double z = 0.5 * 2.0 + (-0.3) * 1.0 + 0.1;   // w·x + b
        double prob = sigmoid(z);
        double ce = crossEntropyLoss(new double[]{1.0}, new double[]{prob});
        System.out.printf("Sigmoid | w=[0.5,-0.3], x=[2.0,1.0], b=0.1%n");
        System.out.printf("        | z = %.4f%n", z);
        System.out.printf("        | σ(z) = %.4f  (expected ≈ 0.6900)%n", prob);
        System.out.printf("        | Cross-Entropy (y=1) = %.4f  (expected ≈ 0.3711)%n", ce);
    }

    // ── Tribuo: L2-regularised logistic regression ───────────────────────────

    static void tribuoLogisticRegression() throws Exception {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TRIBUO — LibLinear Logistic Regression (L2R_LR)");
        System.out.println("=".repeat(60));

        LabelFactory labelFactory = new LabelFactory();
        Random rng = new Random(42);

        // Build synthetic binary classification data programmatically
        List<org.tribuo.Example<Label>> trainExamples = new ArrayList<>();
        List<org.tribuo.Example<Label>> testExamples = new ArrayList<>();

        for (int i = 0; i < 500; i++) {
            boolean positive = (i % 2 == 0);
            double offset = positive ? 1.5 : -1.5;
            double x1 = rng.nextGaussian() + offset;
            double x2 = rng.nextGaussian() + offset;
            String labelStr = positive ? "positive" : "negative";

            ListExample<Label> example = new ListExample<>(new Label(labelStr));
            example.add(new Feature("x1", x1));
            example.add(new Feature("x2", x2));

            if (i < 400) {
                trainExamples.add(example);
            } else {
                testExamples.add(example);
            }
        }

        // Wrap in Tribuo datasets
        var trainSource = new ListDataSource<>(trainExamples, labelFactory,
            new SimpleDataSourceProvenance("synthetic-train", labelFactory));
        var testSource = new ListDataSource<>(testExamples, labelFactory,
            new SimpleDataSourceProvenance("synthetic-test", labelFactory));

        MutableDataset<Label> trainData = new MutableDataset<>(trainSource);
        MutableDataset<Label> testData  = new MutableDataset<>(testSource);

        System.out.printf("Train samples: %d | Test samples: %d%n",
            trainData.size(), testData.size());

        // LibLinear default trainer: L2-regularised logistic regression
        LibLinearClassificationTrainer trainer = new LibLinearClassificationTrainer();
        Model<Label> model = trainer.train(trainData);

        // Evaluate
        LabelEvaluator evaluator = new LabelEvaluator();
        LabelEvaluation evaluation = evaluator.evaluate(model, testData);
        System.out.printf("Accuracy : %.4f%n", evaluation.accuracy());
        System.out.printf("Macro-F1 : %.4f%n", evaluation.macroAveragedF1());

        // Inspect a single prediction
        ListExample<Label> sampleExample = new ListExample<>(new Label("?"));
        sampleExample.add(new Feature("x1", 1.5));
        sampleExample.add(new Feature("x2", 1.5));
        Prediction<Label> prediction = model.predict(sampleExample);
        System.out.printf("%nSample (x1=1.5, x2=1.5)%n");
        System.out.printf("Predicted label : %s%n", prediction.getOutput().getLabel());
        System.out.printf("Score map       : %s%n", prediction.getOutputScores());
    }

    public static void main(String[] args) throws Exception {
        workedExamples();
        tribuoLogisticRegression();
    }
}
```

**Expected output (abridged):**

```
============================================================
WORKED NUMERIC EXAMPLES
============================================================
MSE | y_true=[3.0, 5.0] y_pred=[2.5, 4.5]
    | MSE = 0.2500  (expected 0.2500)

Sigmoid | w=[0.5,-0.3], x=[2.0,1.0], b=0.1
        | z = 0.8000
        | σ(z) = 0.6900  (expected ≈ 0.6900)
        | Cross-Entropy (y=1) = 0.3711  (expected ≈ 0.3711)

============================================================
TRIBUO — LibLinear Logistic Regression (L2R_LR)
============================================================
Train samples: 400 | Test samples: 100
Accuracy : 0.9700
Macro-F1 : 0.9700
```

## Stack Comparison

| Dimension | Python | Java |
|-----------|--------|------|
| **Library** | `scikit-learn>=1.3` | `org.tribuo:tribuo-classification-liblinear:4.3.1` |
| **Model class** | `LogisticRegression` | `LibLinearClassificationTrainer` |
| **Fit call** | `model.fit(X_train, y_train)` | `trainer.train(trainDataset)` |
| **Probabilities** | `model.predict_proba(X)[:, 1]` | `prediction.getOutputScores()` |
| **Evaluation** | `sklearn.metrics` functions | `LabelEvaluator.evaluate()` |
| **Data format** | `np.ndarray` (rows = samples) | `MutableDataset<Label>` (typed examples) |
| **Regularisation default** | L2, `C=1.0` (inverse strength) | L2, `cost=1.0` (same convention) |
| **Solver** | LBFGS (default) | LibLinear L2R_LR |
| **Training speed** | Fast on arrays | Comparable — LibLinear is a C extension |
| **Production fit** | Industry standard | Suitable for JVM-only microservices |

**When to choose Java:** Your model is embedded in an existing JVM service (Spring Boot, Micronaut)
and you need inference without a Python sidecar. For exploratory training, Python wins on
ecosystem breadth.

## Key Takeaways

- **MSE for regression, Cross-Entropy for classification** — the choice of loss function defines
  what "wrong" means and shapes every gradient; use MSE when residuals are continuous and
  cross-entropy when outputs are probabilities.
- **Logistic regression is a linear model with a non-linear output** — the sigmoid adds no hidden
  layers; the decision boundary is still a hyperplane, making the weights directly interpretable as
  feature importance scores.
- **Always train a linear baseline first** — if logistic regression hits 90% accuracy on your
  problem, a deep network that hits 91% is not worth the operational cost, and you now have a
  benchmark every future model must beat.
