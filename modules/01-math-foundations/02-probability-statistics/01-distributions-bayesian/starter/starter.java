// Dependencies (Maven):
//   <dependency>
//     <groupId>org.apache.commons</groupId>
//     <artifactId>commons-math3</artifactId>
//     <version>3.6.1</version>
//   </dependency>
//
// Node: 1.2.1 — Distributions & Bayesian Statistics
// Compile: javac -cp commons-math3-3.6.1.jar StarterAssignment.java
// Run:     java  -cp .:commons-math3-3.6.1.jar StarterAssignment

import org.apache.commons.math3.distribution.NormalDistribution;

import java.util.Arrays;

public class StarterAssignment {

    // -----------------------------------------------------------------------
    // Precomputed Iris training-set statistics (random_state=42, test_size=0.2)
    // Layout: CLASS_STATS[class_id][feature_id] = {mean, std}
    //
    // Features: 0=sepal_length, 1=sepal_width, 2=petal_length, 3=petal_width
    // Classes:  0=Setosa, 1=Versicolor, 2=Virginica
    // -----------------------------------------------------------------------
    static final double[][][] CLASS_STATS = {
        // Class 0 — Setosa (40 training samples, random_state=42)
        { {4.985, 0.308}, {3.415, 0.399}, {1.478, 0.161}, {0.255, 0.115} },
        // Class 1 — Versicolor (40 training samples)
        { {5.930, 0.477}, {2.750, 0.309}, {4.253, 0.443}, {1.320, 0.187} },
        // Class 2 — Virginica (40 training samples)
        { {6.610, 0.685}, {2.980, 0.354}, {5.580, 0.576}, {2.040, 0.269} }
    };

    static final String[] FEATURE_NAMES = {
        "sepal length", "sepal width", "petal length", "petal width"
    };

    static final String[] CLASS_NAMES = { "Setosa", "Versicolor", "Virginica" };

    // -----------------------------------------------------------------------
    // Provided helpers — do not modify
    // -----------------------------------------------------------------------

    /**
     * Six representative Iris test samples (4 features each).
     * Drawn from the sklearn train_test_split with random_state=42.
     */
    static double[][] testSamples() {
        return new double[][] {
            { 5.1, 3.5, 1.4, 0.2 },  // true label 0 — Setosa
            { 6.4, 3.2, 4.5, 1.5 },  // true label 1 — Versicolor
            { 6.3, 3.3, 6.0, 2.5 },  // true label 2 — Virginica
            { 5.5, 2.4, 3.8, 1.1 },  // true label 1 — Versicolor
            { 4.9, 3.1, 1.5, 0.2 },  // true label 0 — Setosa
            { 7.7, 2.6, 6.9, 2.3 },  // true label 2 — Virginica
        };
    }

    static int[] testLabels() {
        return new int[] { 0, 1, 2, 1, 0, 2 };
    }

    // -----------------------------------------------------------------------
    // Task 1 — implement gaussianLogLikelihood
    // -----------------------------------------------------------------------

    /**
     * Return the log probability density of x under N(mean, std^2).
     *
     * Use org.apache.commons.math3.distribution.NormalDistribution:
     *   new NormalDistribution(mean, std).logDensity(x)
     *
     * Expected:
     *   gaussianLogLikelihood(0.0, 0.0, 1.0) → -0.9189
     *   gaussianLogLikelihood(1.0, 0.0, 1.0) → -1.4189
     *
     * @param x    observed feature value
     * @param mean class-conditional mean  (μ_c,f)
     * @param std  class-conditional std   (σ_c,f)
     * @return log p(x | N(mean, std^2))
     */
    public static double gaussianLogLikelihood(double x, double mean, double std) {
        // TODO: implement this using new NormalDistribution(mean, std).logDensity(x)
        throw new UnsupportedOperationException("TODO: implement gaussianLogLikelihood");
    }

    // -----------------------------------------------------------------------
    // Task 2 — implement naiveBayesPredict
    // -----------------------------------------------------------------------

    /**
     * Classify each sample using log-space Gaussian Naive Bayes.
     *
     * Algorithm (for each sample row):
     *   1. For each class c (0, 1, 2):
     *        logScore[c] = logPriors[c]
     *                    + sum of gaussianLogLikelihood(x[f], mean_c_f, std_c_f)
     *                      over all features f
     *   2. Return argmax(logScore) as the predicted label.
     *
     * Working in log-space avoids floating-point underflow when multiplying
     * many small probabilities.
     *
     * @param samples    shape [nSamples][nFeatures] — test feature matrix
     * @param classStats [class_id][feature_id] = {mean, std}
     * @param logPriors  log P(class) per class; index matches class_id
     * @return integer predicted class label per sample, length nSamples
     */
    public static int[] naiveBayesPredict(
            double[][] samples,
            double[][][] classStats,
            double[] logPriors) {
        // TODO: implement this
        //   Hint: outer loop over samples, inner loop over classes,
        //         innermost loop over features accumulating log-likelihoods.
        //         Track the class with the highest log score per sample.
        throw new UnsupportedOperationException("TODO: implement naiveBayesPredict");
    }

    // -----------------------------------------------------------------------
    // Task 3 — implement evaluateAccuracy
    // -----------------------------------------------------------------------

    /**
     * Return the fraction of predictions that match the true labels.
     *
     * Expected range on the provided 6 test samples: 0.83–1.00.
     *
     * @param yTrue ground-truth labels, length nSamples
     * @param yPred predicted labels,    length nSamples
     * @return accuracy in [0.0, 1.0]
     */
    public static double evaluateAccuracy(int[] yTrue, int[] yPred) {
        // TODO: implement this
        //   Count matches, divide by total length.
        throw new UnsupportedOperationException("TODO: implement evaluateAccuracy");
    }

    // -----------------------------------------------------------------------
    // Task 4 — implement sequentialBayesUpdate
    // -----------------------------------------------------------------------

    /**
     * Apply Bayes' theorem iteratively across multiple independent observations.
     * Each posterior becomes the prior for the next step.
     *
     * Per-step formula:
     *   marginal  = P(E|H)*P(H) + P(E|¬H)*(1 - P(H))
     *   posterior = P(E|H)*P(H) / marginal
     *
     * Scenario: updating P(Setosa) after observing whether petal_length > 2.45.
     *   P(E=1 | Setosa)  ≈ 0.02  — almost no Setosa has petal_length > 2.45
     *   P(E=1 | ¬Setosa) ≈ 0.98  — almost all non-Setosa have petal_length > 2.45
     *
     * @param prior              initial P(H) before any evidence
     * @param likelihoodsGivenH  [P(E_1|H), P(E_2|H), ...] — one per observation
     * @param likelihoodsGivenNH [P(E_1|¬H), P(E_2|¬H), ...] — one per observation
     * @return posteriors array: [P(H|E_1), P(H|E_1,E_2), ...]
     */
    public static double[] sequentialBayesUpdate(
            double prior,
            double[] likelihoodsGivenH,
            double[] likelihoodsGivenNH) {
        // TODO: implement this
        //   Hint: loop over observations, compute marginal and posterior,
        //         then set prior = posterior before next iteration.
        throw new UnsupportedOperationException("TODO: implement sequentialBayesUpdate");
    }

    // -----------------------------------------------------------------------
    // main — runs all tasks in order, prints labeled results
    // -----------------------------------------------------------------------

    public static void main(String[] args) {
        double[][] samples    = testSamples();
        int[]      trueLabels = testLabels();

        // Equal class priors: log(1/3) per class (balanced Iris training set)
        double logPrior   = Math.log(1.0 / 3.0);
        double[] logPriors = { logPrior, logPrior, logPrior };

        // ── Task 1: gaussianLogLikelihood sanity check ───────────────────
        System.out.println("=".repeat(60));
        System.out.println("Task 1: gaussianLogLikelihood — sanity check");
        System.out.println("=".repeat(60));
        try {
            double ll0 = gaussianLogLikelihood(0.0, 0.0, 1.0);
            double ll1 = gaussianLogLikelihood(1.0, 0.0, 1.0);
            System.out.printf("  log p(x=0.0 | N(0,1)) : %.4f   (expect -0.9189)%n", ll0);
            System.out.printf("  log p(x=1.0 | N(0,1)) : %.4f   (expect -1.4189)%n", ll1);
        } catch (UnsupportedOperationException e) {
            System.out.println("  [NOT IMPLEMENTED] gaussianLogLikelihood");
        }

        // Class stats printout — always runs (data is hardcoded above)
        System.out.println();
        System.out.println("=".repeat(60));
        System.out.println("Class-conditional statistics (precomputed from training set):");
        System.out.println("=".repeat(60));
        for (int c = 0; c < CLASS_STATS.length; c++) {
            System.out.printf(
                "  Class %d (%s) — %s: mean=%.3f  std=%.3f%n",
                c, CLASS_NAMES[c], FEATURE_NAMES[0],
                CLASS_STATS[c][0][0], CLASS_STATS[c][0][1]
            );
        }

        // ── Task 2 & 3: Scratch Gaussian Naive Bayes ─────────────────────
        System.out.println();
        System.out.println("=".repeat(60));
        System.out.println("Task 2 & 3: Scratch Gaussian Naive Bayes — predict and evaluate");
        System.out.println("=".repeat(60));
        try {
            int[] predictions = naiveBayesPredict(samples, CLASS_STATS, logPriors);
            System.out.println("  Predictions : " + Arrays.toString(predictions));
            System.out.println("  True labels : " + Arrays.toString(trueLabels));
            try {
                double acc = evaluateAccuracy(trueLabels, predictions);
                System.out.printf("  Accuracy    : %.4f   (expect 0.83–1.00)%n", acc);
            } catch (UnsupportedOperationException ex) {
                System.out.println("  [NOT IMPLEMENTED] evaluateAccuracy");
            }
        } catch (UnsupportedOperationException e) {
            System.out.println("  [NOT IMPLEMENTED] naiveBayesPredict");
        }

        // ── Task 4: Sequential Bayes update ──────────────────────────────
        System.out.println();
        System.out.println("=".repeat(60));
        System.out.println("Task 4: Sequential Bayes — P(Setosa | petal_length > 2.45)");
        System.out.println("=".repeat(60));
        // All 5 observed petal lengths are > 2.45 → strong evidence against Setosa.
        // Each observation independently drives P(Setosa) toward 0.
        double   priorSetosa      = 1.0 / 3.0;
        double[] likelihoodsH     = { 0.02, 0.02, 0.02, 0.02, 0.02 }; // P(E|Setosa)
        double[] likelihoodsNotH  = { 0.98, 0.98, 0.98, 0.98, 0.98 }; // P(E|¬Setosa)
        System.out.printf("  Initial prior P(Setosa)  : %.4f%n", priorSetosa);
        try {
            double[] posteriors = sequentialBayesUpdate(priorSetosa, likelihoodsH, likelihoodsNotH);
            double prev = priorSetosa;
            for (int i = 0; i < posteriors.length; i++) {
                System.out.printf(
                    "  After obs %d:  P(Setosa) %.4f → %.4f%n",
                    i + 1, prev, posteriors[i]
                );
                prev = posteriors[i];
            }
            double delta = Math.abs(posteriors[posteriors.length - 1] - priorSetosa);
            System.out.printf("%n  Total shift in P(Setosa) : %.4f   (expect ≥ 0.30)%n", delta);
        } catch (UnsupportedOperationException e) {
            System.out.println("  [NOT IMPLEMENTED] sequentialBayesUpdate");
        }
    }
}
