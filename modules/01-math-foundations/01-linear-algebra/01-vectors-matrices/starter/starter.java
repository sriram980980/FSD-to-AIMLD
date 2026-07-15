// Dependencies (Maven):
//   <dependency>
//     <groupId>org.apache.commons</groupId>
//     <artifactId>commons-math3</artifactId>
//     <version>3.6.1</version>
//   </dependency>
//
// Node: 1.1.1 — Vectors & Matrices: Core Operations
// Compile: javac -cp commons-math3-3.6.1.jar StarterAssignment.java
// Run:     java  -cp .:commons-math3-3.6.1.jar StarterAssignment

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class StarterAssignment {

    // -----------------------------------------------------------------------
    // Implemented helpers — provided for you, no changes needed
    // -----------------------------------------------------------------------

    /** Print a labelled section header. */
    static void printSeparator(String label) {
        System.out.println("==================================================");
        System.out.println(label);
        System.out.println("==================================================");
    }

    /** Print a pass/fail line comparing two double values within a tolerance. */
    static boolean verifyClose(String label, double actual, double expected, double atol) {
        boolean pass = Math.abs(actual - expected) <= atol;
        System.out.printf("  [%s] %s%n", pass ? "PASS" : "FAIL", label);
        return pass;
    }

    /** Print a pass/fail line comparing two matrices within a tolerance. */
    static boolean verifyMatrixClose(String label, double[][] actual,
                                     double[][] expected, double atol) {
        for (int i = 0; i < actual.length; i++) {
            for (int j = 0; j < actual[i].length; j++) {
                if (Math.abs(actual[i][j] - expected[i][j]) > atol) {
                    System.out.printf("  [FAIL] %s%n", label);
                    return false;
                }
            }
        }
        System.out.printf("  [PASS] %s%n", label);
        return true;
    }

    // -----------------------------------------------------------------------
    // Task 1 — Cosine Similarity
    // -----------------------------------------------------------------------

    /**
     * Return the cosine similarity between two double arrays of equal length.
     *
     * Steps:
     *   1. Compute the numerator as the sum of element-wise products (do NOT
     *      use ArrayRealVector.dotProduct — multiply manually in a loop).
     *   2. Compute the L2 norm of each vector (loop-based sqrt of sum of squares).
     *   3. Return numerator / (normA * normB).
     *
     * TODO: implement this method.
     */
    static double cosineSimilarity(double[] a, double[] b) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // -----------------------------------------------------------------------
    // Task 2 — Row-Wise L2 Normalisation
    // -----------------------------------------------------------------------

    /**
     * Return a new matrix where every row of X has unit L2 norm.
     * Rows whose norm is zero are left unchanged (copied as-is).
     *
     * Steps:
     *   1. Allocate a new double[][] with the same dimensions as X.
     *   2. For each row, compute its L2 norm.
     *   3. If norm > 0, divide each element by the norm; otherwise copy unchanged.
     *
     * TODO: implement this method.
     */
    static double[][] rowNormalize(double[][] X) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // -----------------------------------------------------------------------
    // Task 3 — Batch Feature Projection
    // -----------------------------------------------------------------------

    /**
     * Project each row of feature matrix X (n×d) onto weight vector (d,).
     * Return a double[] of length n where element i is the dot product of
     * row i of X with weights.
     *
     * Use RealMatrix.multiply(RealVector) or equivalent — no explicit loops.
     *
     * TODO: implement this method.
     */
    static double[] batchProject(double[][] X, double[] weights) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // -----------------------------------------------------------------------
    // Task 4 — Power Iteration for Dominant Eigenpair
    // -----------------------------------------------------------------------

    /** Simple container for an eigenpair result. */
    record EigenPair(double eigenvalue, double[] eigenvector) {}

    /**
     * Approximate the dominant eigenpair of symmetric square matrix M.
     *
     * Algorithm:
     *   1. Initialise vector v = {1.0, 1.0, ..., 1.0} (length = M rows).
     *   2. Normalise v to unit L2 norm.
     *   3. For nIter steps:
     *        v = M × v   (matrix-vector multiply via RealMatrix.operate)
     *        normalise v to unit L2 norm
     *   4. Compute eigenvalue as the Rayleigh quotient: λ = vᵀ M v
     *        (i.e. dotProduct(v, M.operate(v))).
     *   5. Return new EigenPair(eigenvalue, v.toArray()).
     *
     * TODO: implement this method.
     */
    static EigenPair powerIteration(double[][] mData, int nIter) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // -----------------------------------------------------------------------
    // Task 5 — Covariance Matrix
    // -----------------------------------------------------------------------

    /**
     * Compute the (d×d) sample covariance matrix from data matrix X (n×d).
     *
     * Steps:
     *   1. Compute the column mean vector (length d).
     *   2. Subtract the mean from each row to produce X_centred (n×d).
     *   3. Return (X_centredᵀ × X_centred) / (n - 1) as a double[][].
     *
     * Do NOT use Apache Commons CovarianceMatrix or similar utility.
     * Use only RealMatrix operations (multiply, transpose, scalarMultiply).
     *
     * TODO: implement this method.
     */
    static double[][] covarianceMatrix(double[][] X) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // -----------------------------------------------------------------------
    // main — runs all tasks in order with labelled output
    // -----------------------------------------------------------------------

    public static void main(String[] args) {

        // ------------------------------------------------------------------
        // Task 1
        // ------------------------------------------------------------------
        printSeparator("Task 1 — Cosine Similarity");
        double[] a = {1.0, 2.0, 3.0};
        double[] b = {4.0, 5.0, 6.0};
        double sim = cosineSimilarity(a, b);
        System.out.printf("  a                      : [1.0, 2.0, 3.0]%n");
        System.out.printf("  b                      : [4.0, 5.0, 6.0]%n");
        System.out.printf("  cosineSimilarity(a, b) : %.4f%n", sim);
        verifyClose("Expected ≈ 0.9746", sim, 0.9746, 1e-4);
        System.out.println();

        // ------------------------------------------------------------------
        // Task 2
        // ------------------------------------------------------------------
        printSeparator("Task 2 — Row-Wise L2 Normalisation");
        double[][] Xnorm = {{3.0, 4.0}, {1.0, 0.0}, {0.0, 5.0}};
        double[][] normalised = rowNormalize(Xnorm);
        System.out.println("  Normalised rows:");
        for (int i = 0; i < normalised.length; i++) {
            double norm = 0;
            for (double v : normalised[i]) norm += v * v;
            norm = Math.sqrt(norm);
            System.out.printf("    row %d: [%.4f, %.4f]  norm=%.4f%n",
                    i, normalised[i][0], normalised[i][1], norm);
        }
        System.out.println();

        // ------------------------------------------------------------------
        // Task 3
        // ------------------------------------------------------------------
        printSeparator("Task 3 — Batch Feature Projection");
        double[][] Xproj  = {{0.25, 0.80}, {0.60, 0.45}, {0.90, 0.10}};
        double[]   weights = {0.7, 0.3};
        double[]   projections = batchProject(Xproj, weights);
        System.out.println("  Projections (expected: 0.4150, 0.5550, 0.6600):");
        for (int i = 0; i < projections.length; i++) {
            System.out.printf("    User %d: %.4f%n", i, projections[i]);
        }
        System.out.println();

        // ------------------------------------------------------------------
        // Task 4
        // ------------------------------------------------------------------
        printSeparator("Task 4 — Power Iteration (dominant eigenpair)");
        double[][] M = {{4.0, 2.0}, {2.0, 3.0}};
        EigenPair ep = powerIteration(M, 100);
        System.out.printf("  Dominant eigenvalue  : %.4f%n", ep.eigenvalue());
        System.out.printf("  Dominant eigenvector : [%.4f, %.4f]%n",
                ep.eigenvector()[0], ep.eigenvector()[1]);
        verifyClose("Eigenvalue ≈ 5.5616 (±0.001)", ep.eigenvalue(), 5.5616, 1e-3);
        // Verify M·v ≈ λ·v
        RealMatrix Mmat = new Array2DRowRealMatrix(M, false);
        RealVector v    = new ArrayRealVector(ep.eigenvector(), false);
        RealVector Mv   = Mmat.operate(v);
        RealVector lv   = v.mapMultiply(ep.eigenvalue());
        boolean match   = Mv.subtract(lv).getNorm() < 1e-6;
        System.out.printf("  Verification M·v ≈ λ·v : %b%n", match);
        System.out.println();

        // ------------------------------------------------------------------
        // Task 5
        // ------------------------------------------------------------------
        printSeparator("Task 5 — Covariance Matrix");
        double[][] Xcov = {
            {2.0, 0.0, 4.0},
            {3.0, 1.0, 5.0},
            {4.0, 2.0, 6.0},
            {5.0, 3.0, 7.0}
        };
        double[][] cov = covarianceMatrix(Xcov);
        System.out.println("  Covariance matrix:");
        for (double[] row : cov) {
            System.out.printf("    [%.4f, %.4f, %.4f]%n", row[0], row[1], row[2]);
        }
        // Expected: all cells ≈ 1.6667
        double[][] expectedCov = {
            {5.0/3.0, 5.0/3.0, 5.0/3.0},
            {5.0/3.0, 5.0/3.0, 5.0/3.0},
            {5.0/3.0, 5.0/3.0, 5.0/3.0}
        };
        verifyMatrixClose("Covariance matches expected (±1e-10)", cov, expectedCov, 1e-10);
        System.out.println();
    }
}
