# 1.2.1 — Distributions & Bayesian Statistics

## Hook

Bayes' theorem is a confidence score updater — like updating a feature-flag rollout percentage as you collect real-user data: you start with a prior belief (1% rollout), observe evidence (error rate drops), and compute a revised posterior probability (raise to 20%), replacing gut feel with arithmetic on evidence.

## The Problem

Model outputs are rarely a hard 0 or 1 — they are probability scores, and interpreting them correctly requires knowing what a probability distribution actually means. Without the normal distribution you cannot understand confidence intervals, model calibration, or why standardising your input features makes training faster. Without Bayes' theorem you cannot reason about false-positive rates, understand why a 99%-accurate cancer test is mostly wrong on a rare disease, or build any generative model from scratch. These two concepts underpin loss functions, regularisation, Gaussian noise augmentation, and the entire probabilistic ML stack.

## Theory

### The Normal (Gaussian) Distribution

$$p(x \mid \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}}\, e^{-\frac{(x - \mu)^2}{2\sigma^2}}$$

- $x$ — the value being evaluated
- $\mu$ — the **mean**: the centre of the bell curve
- $\sigma$ — the **standard deviation**: controls the width of the bell; larger $\sigma$ → flatter curve
- $\sigma^2$ — the **variance**: average squared distance from the mean
- $p(x \mid \mu, \sigma)$ — the **probability density** at $x$; integrates to 1 over all $x$

**Worked example** — standard normal ($\mu = 0$, $\sigma = 1$):

$$p(x = 0 \mid 0, 1) = \frac{1}{1 \cdot \sqrt{2\pi}}\, e^{0} = \frac{1}{\sqrt{6.2832}} \approx \frac{1}{2.5066} \approx 0.3989$$

The **68-95-99.7 rule**: 68% of all observations fall within $\mu \pm 1\sigma$, 95% within $\mu \pm 2\sigma$, and 99.7% within $\mu \pm 3\sigma$. A model prediction score of 0.05 on a feature with mean 0 and σ 1 sits more than 1.6 standard deviations away — uncommon, but not impossible.

---

### Bayes' Theorem

$$P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)}$$

- $H$ — the **hypothesis** you want to evaluate (e.g., "this email is spam")
- $E$ — the **evidence** you have observed (e.g., the word "buy" appears)
- $P(H)$ — the **prior**: your belief in $H$ before seeing any evidence
- $P(E \mid H)$ — the **likelihood**: probability of observing $E$ if $H$ is true
- $P(H \mid E)$ — the **posterior**: your updated belief after observing $E$
- $P(E)$ — the **marginal evidence** (normalisation constant):

$$P(E) = P(E \mid H) \cdot P(H) + P(E \mid \neg H) \cdot P(\neg H)$$

**Worked example** — spam filter:

| Quantity | Value | Source |
|---|---|---|
| $P(\text{spam})$ | 0.30 | Prior: 30% of all emails are spam |
| $P(\text{"buy"} \mid \text{spam})$ | 0.80 | 80% of spam emails contain "buy" |
| $P(\text{"buy"} \mid \neg\text{spam})$ | 0.10 | 10% of legitimate emails contain "buy" |

Step 1 — compute marginal evidence:
$$P(\text{"buy"}) = (0.80)(0.30) + (0.10)(0.70) = 0.24 + 0.07 = 0.31$$

Step 2 — apply Bayes' theorem:
$$P(\text{spam} \mid \text{"buy"}) = \frac{(0.80)(0.30)}{0.31} = \frac{0.24}{0.31} \approx 0.7742$$

Seeing the word "buy" raises spam probability from 30% → 77%. That updated score is your posterior — and it becomes the new prior the moment you observe the next word.

---

### Why This Pairs With the Normal Distribution in ML

Features of real datasets follow (approximately) normal distributions. When you standardise a feature — subtract the mean, divide by the standard deviation — you are transforming it to $\mathcal{N}(0, 1)$. Gradient descent converges faster on standardised inputs because all features live on the same scale, so no single weight receives disproportionately large gradient steps. Bayes' theorem then lets you frame classification as posterior inference: Naive Bayes classifiers assume each feature is drawn from $\mathcal{N}(\mu_c, \sigma_c^2)$ for each class $c$ and apply Bayes to score which class is most likely.

## Python Implementation

```python
# Dependencies: numpy>=1.24, scipy>=1.11, matplotlib>=3.7
import numpy as np
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Tuple


def normal_pdf(x: float, mean: float, std: float) -> float:
    """Evaluate the Gaussian probability density at x."""
    return float(norm.pdf(x, loc=mean, scale=std))


def normal_cdf(x: float, mean: float, std: float) -> float:
    """Return P(X <= x) for X ~ N(mean, std^2)."""
    return float(norm.cdf(x, loc=mean, scale=std))


def standardise(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Return (z_scores, mean, std). Transforms data to N(0,1)."""
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1))          # ddof=1 → sample std
    z_scores = (data - mean) / std
    return z_scores, mean, std


def bayes_update(
    prior: float,
    likelihood_given_h: float,
    likelihood_given_not_h: float,
) -> Tuple[float, float]:
    """
    Return (posterior, marginal_evidence) using Bayes' theorem.

    Args:
        prior: P(H) — initial belief that hypothesis H is true
        likelihood_given_h: P(E|H) — probability of evidence if H is true
        likelihood_given_not_h: P(E|¬H) — probability of evidence if H is false
    """
    marginal = likelihood_given_h * prior + likelihood_given_not_h * (1 - prior)
    posterior = (likelihood_given_h * prior) / marginal
    return posterior, marginal


def sequential_bayes(
    initial_prior: float,
    likelihoods_h: list[float],
    likelihoods_not_h: list[float],
) -> list[float]:
    """
    Apply Bayes' theorem sequentially for multiple pieces of evidence.
    Each posterior becomes the prior for the next step.
    """
    posteriors: list[float] = []
    prior = initial_prior
    for lh, lnh in zip(likelihoods_h, likelihoods_not_h):
        posterior, _ = bayes_update(prior, lh, lnh)
        posteriors.append(posterior)
        prior = posterior                       # posterior becomes next prior
    return posteriors


def plot_normal_curves(output_path: str = "normal_curves.png") -> None:
    """Save a plot of N(0,1), N(2,1), and N(0,4) on the same axes."""
    x = np.linspace(-6, 8, 400)
    curves = [
        (0, 1,  "N(μ=0, σ=1)  — standard normal"),
        (2, 1,  "N(μ=2, σ=1)  — shifted mean"),
        (0, 2,  "N(μ=0, σ=2)  — wider spread"),
    ]
    plt.figure(figsize=(9, 4))
    for mu, sigma, label in curves:
        plt.plot(x, norm.pdf(x, loc=mu, scale=sigma), label=label, linewidth=2)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.title("Normal Distribution — Effect of μ and σ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Plot saved → {output_path}")


def main() -> None:
    print("=" * 55)
    print("1. Normal Distribution — PDF and CDF")
    print("=" * 55)
    mu, sigma = 0.0, 1.0
    x_val = 0.0
    pdf_val = normal_pdf(x_val, mu, sigma)
    print(f"  N(μ={mu}, σ={sigma})")
    print(f"  p(x={x_val})               : {pdf_val:.4f}")   # 0.3989
    # 68-95-99.7 rule verification
    within_1sigma = normal_cdf(mu + sigma, mu, sigma) - normal_cdf(mu - sigma, mu, sigma)
    within_2sigma = normal_cdf(mu + 2*sigma, mu, sigma) - normal_cdf(mu - 2*sigma, mu, sigma)
    within_3sigma = normal_cdf(mu + 3*sigma, mu, sigma) - normal_cdf(mu - 3*sigma, mu, sigma)
    print(f"  P(μ±1σ) = P(-1 ≤ X ≤ 1)  : {within_1sigma:.4f}  (expect ≈ 0.6827)")
    print(f"  P(μ±2σ) = P(-2 ≤ X ≤ 2)  : {within_2sigma:.4f}  (expect ≈ 0.9545)")
    print(f"  P(μ±3σ) = P(-3 ≤ X ≤ 3)  : {within_3sigma:.4f}  (expect ≈ 0.9973)")

    print()
    print("=" * 55)
    print("2. Standardisation (feature scaling)")
    print("=" * 55)
    # Simulate user ages in an ML dataset
    np.random.seed(42)
    ages = np.random.normal(loc=35.0, scale=8.0, size=10).round(1)
    z_scores, sample_mean, sample_std = standardise(ages)
    print(f"  Raw ages    : {ages}")
    print(f"  Sample mean : {sample_mean:.2f}   Sample std : {sample_std:.2f}")
    print(f"  Z-scores    : {np.round(z_scores, 3)}")
    print(f"  Z mean      : {np.mean(z_scores):.6f}  (expect ≈ 0.0)")
    print(f"  Z std       : {np.std(z_scores, ddof=1):.6f}  (expect = 1.0)")

    print()
    print("=" * 55)
    print("3. Bayes' Theorem — Spam Filter")
    print("=" * 55)
    prior_spam          = 0.30   # P(spam)
    p_buy_given_spam    = 0.80   # P("buy" | spam)
    p_buy_given_legit   = 0.10   # P("buy" | ¬spam)

    posterior, marginal = bayes_update(prior_spam, p_buy_given_spam, p_buy_given_legit)
    print(f"  Prior  P(spam)                : {prior_spam:.2f}")
    print(f"  P('buy' | spam)               : {p_buy_given_spam:.2f}")
    print(f"  P('buy' | ¬spam)              : {p_buy_given_legit:.2f}")
    print(f"  Marginal P('buy')             : {marginal:.4f}")
    print(f"  Posterior P(spam | 'buy')     : {posterior:.4f}  ← 30% → 77%")

    print()
    print("=" * 55)
    print("4. Sequential Bayes — Multiple Word Evidence")
    print("=" * 55)
    # Three words observed in sequence: "buy", "now", "free"
    words                = ["buy",  "now",  "free"]
    likelihoods_spam     = [0.80,   0.70,   0.90]
    likelihoods_legit    = [0.10,   0.20,   0.05]
    posteriors = sequential_bayes(prior_spam, likelihoods_spam, likelihoods_legit)
    prior_chain = prior_spam
    for word, post in zip(words, posteriors):
        print(f"  After '{word}': P(spam) {prior_chain:.4f} → {post:.4f}")
        prior_chain = post
    print(f"\n  Final spam probability after all three words: {posteriors[-1]:.4f}")

    print()
    plot_normal_curves()


if __name__ == "__main__":
    main()
```

Run with `python lesson.py`. Notice five things in the output:

1. **PDF at 0** prints `0.3989`, matching the hand-computed $1/\sqrt{2\pi}$ from the Theory section exactly.
2. **68-95-99.7 rule** is confirmed numerically — the CDF values match the textbook thresholds to four decimal places.
3. **Standardisation** drives the Z-score mean to essentially zero and std to 1.0, even on 10 samples — this is what `sklearn.preprocessing.StandardScaler` does internally before every gradient-based model.
4. **Single Bayes step** lifts spam probability from 0.30 to 0.7742, matching the worked arithmetic.
5. **Sequential updates** compound evidence: three words push the posterior past 0.99 — exactly how a Naive Bayes classifier accumulates evidence across all tokens in an email.

## Java Implementation

```java
// Dependencies: org.apache.commons:commons-math3:3.6.1
// Maven:
//   <dependency>
//     <groupId>org.apache.commons</groupId>
//     <artifactId>commons-math3</artifactId>
//     <version>3.6.1</version>
//   </dependency>

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

public class DistributionsBayesian {

    /** Evaluate the Gaussian PDF at x for N(mean, std^2). */
    public static double normalPdf(double x, double mean, double std) {
        NormalDistribution dist = new NormalDistribution(mean, std);
        return dist.density(x);
    }

    /** Return P(X <= x) for X ~ N(mean, std^2). */
    public static double normalCdf(double x, double mean, double std) {
        NormalDistribution dist = new NormalDistribution(mean, std);
        return dist.cumulativeProbability(x);
    }

    /**
     * Apply Bayes' theorem once and return the posterior.
     *
     * @param prior               P(H)
     * @param likelihoodGivenH    P(E | H)
     * @param likelihoodGivenNotH P(E | ¬H)
     * @return P(H | E)
     */
    public static double bayesUpdate(
            double prior,
            double likelihoodGivenH,
            double likelihoodGivenNotH) {
        double marginal = likelihoodGivenH * prior
                        + likelihoodGivenNotH * (1.0 - prior);
        return (likelihoodGivenH * prior) / marginal;
    }

    /**
     * Apply Bayes' theorem sequentially across multiple evidence items.
     * Each posterior becomes the prior for the next step.
     */
    public static double[] sequentialBayes(
            double initialPrior,
            double[] likelihoodsH,
            double[] likelihoodsNotH) {
        double[] posteriors = new double[likelihoodsH.length];
        double prior = initialPrior;
        for (int i = 0; i < likelihoodsH.length; i++) {
            prior = bayesUpdate(prior, likelihoodsH[i], likelihoodsNotH[i]);
            posteriors[i] = prior;
        }
        return posteriors;
    }

    /** Return z-scores using sample mean and sample std (ddof=1). */
    public static double[] standardise(double[] data) {
        DescriptiveStatistics stats = new DescriptiveStatistics(data);
        double mean = stats.getMean();
        double std  = stats.getStandardDeviation();  // uses N-1
        double[] z = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            z[i] = (data[i] - mean) / std;
        }
        return z;
    }

    public static void main(String[] args) {
        // 1. Normal distribution PDF and CDF
        System.out.println("=".repeat(55));
        System.out.println("1. Normal Distribution — PDF and CDF");
        System.out.println("=".repeat(55));
        double mu = 0.0, sigma = 1.0;
        System.out.printf("  p(x=0 | N(0,1))           : %.4f%n", normalPdf(0.0, mu, sigma));  // 0.3989
        double within1Sigma = normalCdf(mu + sigma, mu, sigma) - normalCdf(mu - sigma, mu, sigma);
        double within2Sigma = normalCdf(mu + 2*sigma, mu, sigma) - normalCdf(mu - 2*sigma, mu, sigma);
        System.out.printf("  P(μ±1σ)                    : %.4f  (expect ≈ 0.6827)%n", within1Sigma);
        System.out.printf("  P(μ±2σ)                    : %.4f  (expect ≈ 0.9545)%n", within2Sigma);

        // 2. Standardisation
        System.out.println("\n" + "=".repeat(55));
        System.out.println("2. Standardisation (feature scaling)");
        System.out.println("=".repeat(55));
        double[] ages   = {29.3, 41.1, 35.0, 27.8, 50.2, 33.4, 38.9, 44.5, 31.6, 26.0};
        double[] zScores = standardise(ages);
        System.out.print("  Z-scores: ");
        for (double z : zScores) System.out.printf("%.3f  ", z);
        DescriptiveStatistics zStats = new DescriptiveStatistics(zScores);
        System.out.printf("%n  Z mean : %.6f  (expect ≈ 0.0)%n", zStats.getMean());
        System.out.printf("  Z std  : %.6f  (expect = 1.0)%n", zStats.getStandardDeviation());

        // 3. Bayes' theorem — spam filter
        System.out.println("\n" + "=".repeat(55));
        System.out.println("3. Bayes' Theorem — Spam Filter");
        System.out.println("=".repeat(55));
        double priorSpam       = 0.30;
        double pBuyGivenSpam   = 0.80;
        double pBuyGivenLegit  = 0.10;
        double posterior = bayesUpdate(priorSpam, pBuyGivenSpam, pBuyGivenLegit);
        System.out.printf("  Prior  P(spam)                : %.2f%n", priorSpam);
        System.out.printf("  Posterior P(spam | 'buy')     : %.4f  ← 30%% → 77%%%n", posterior);

        // 4. Sequential Bayes — multiple words
        System.out.println("\n" + "=".repeat(55));
        System.out.println("4. Sequential Bayes — Multiple Word Evidence");
        System.out.println("=".repeat(55));
        String[] words           = {"buy",  "now",  "free"};
        double[] likelihoodsSpam = {0.80,   0.70,   0.90};
        double[] likelihoodsLeg  = {0.10,   0.20,   0.05};
        double[] posteriors      = sequentialBayes(priorSpam, likelihoodsSpam, likelihoodsLeg);
        double prev = priorSpam;
        for (int i = 0; i < words.length; i++) {
            System.out.printf("  After '%s': P(spam) %.4f → %.4f%n",
                              words[i], prev, posteriors[i]);
            prev = posteriors[i];
        }
        System.out.printf("%n  Final spam probability after all words: %.4f%n",
                          posteriors[posteriors.length - 1]);
    }
}
```

Compile and run with:

```bash
javac -cp commons-math3-3.6.1.jar DistributionsBayesian.java
java  -cp .:commons-math3-3.6.1.jar DistributionsBayesian
```

## Stack Comparison

| Dimension | Python (SciPy) | Java (Commons Math 3) |
|---|---|---|
| **Library** | `scipy>=1.11` + `numpy>=1.24` | `commons-math3:3.6.1` |
| **Normal distribution** | `scipy.stats.norm` (frozen object) | `NormalDistribution(μ, σ)` (instance per call) |
| **PDF evaluation** | `norm.pdf(x, loc=μ, scale=σ)` | `dist.density(x)` |
| **CDF evaluation** | `norm.cdf(x, loc=μ, scale=σ)` | `dist.cumulativeProbability(x)` |
| **Descriptive stats** | `np.mean`, `np.std` | `DescriptiveStatistics` (streaming, low memory) |
| **Visualisation** | `matplotlib` — direct plotting | No built-in; export data and plot externally |
| **Ecosystem fit** | Primary for ML/stats pipelines | JVM services doing runtime probability scoring |
| **Distribution zoo** | 100+ distributions in `scipy.stats` | ~20 distributions in Commons Math |

## Key Takeaways

- **The normal distribution is the default noise model** — when you standardise features to $\mathcal{N}(0,1)$, you align all input scales so gradient descent takes equally-sized steps in every weight direction; skip this step and one feature can dominate and stall training.
- **Bayes' theorem converts priors into posteriors** — $P(H \mid E) = P(E \mid H) \cdot P(H) / P(E)$ means every new piece of evidence multiplicatively updates your belief; run it sequentially and you have a Naive Bayes classifier, which remains competitive on text tasks despite its simplicity.
- **Posterior uncertainty is as important as the point estimate** — a model predicting 0.77 spam is not the same as predicting 1.0; keeping track of that remaining 23% uncertainty (via calibration, confidence intervals, or full Bayesian posteriors) is what separates production-safe ML from overconfident prototypes.
