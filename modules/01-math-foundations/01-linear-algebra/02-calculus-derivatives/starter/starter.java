// Dependencies (Maven):
//   <dependency>
//     <groupId>org.apache.commons</groupId>
//     <artifactId>commons-math3</artifactId>
//     <version>3.6.1</version>
//   </dependency>
// Node: 1.1.2 — Calculus: Derivatives & Gradients
// Java 17+ | Run: javac -cp commons-math3-3.6.1.jar StarterAssignment.java
//                  java  -cp .:commons-math3-3.6.1.jar StarterAssignment

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.analysis.differentiation.UnivariateDifferentiableFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class StarterAssignment {

    // -----------------------------------------------------------------------
    // Dataset — fixed seed to match starter.py (X in [0,10], y = 3x + 0.5 + noise)
    // -----------------------------------------------------------------------
    static final int N = 50;
    static final double[] X_DATA;
    static final double[] Y_DATA;

    static {
        Random rng = new Random(42);
        X_DATA = new double[N];
        Y_DATA = new double[N];
        for (int i = 0; i < N; i++) {
            X_DATA[i] = rng.nextDouble() * 10.0;
            Y_DATA[i] = 3.0 * X_DATA[i] + 0.5 + rng.nextGaussian();
        }
    }

    // -----------------------------------------------------------------------
    // Provided helper — pretty section header
    // -----------------------------------------------------------------------
    static void printSection(String title) {
        System.out.println("\n" + "=".repeat(50));
        System.out.printf("  %s%n", title);
        System.out.println("=".repeat(50));
    }

    // -----------------------------------------------------------------------
    // Record for gradient descent result
    // -----------------------------------------------------------------------
    record GDResult(double weight, double bias, List<Double> lossHistory) {}

    // -----------------------------------------------------------------------
    // Task 1 — Numerical Derivative via Finite Difference
    //
    // Implement: (f(x + h) - f(x - h)) / (2 * h)
    // -----------------------------------------------------------------------

    /**
     * Estimate f'(x) using the central-difference formula.
     *
     * @param f the scalar function to differentiate (use a lambda: x -> x*x*x - 4*x)
     * @param x the point at which to evaluate the derivative
     * @param h the finite-difference step size
     * @return approximate f'(x)
     */
    static double numericalDerivative(UnivariateFunction f, double x, double h) {
        // TODO: implement central-difference formula: (f(x+h) - f(x-h)) / (2*h)
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // -----------------------------------------------------------------------
    // Task 2 — MSE Loss and Analytical Gradient
    // -----------------------------------------------------------------------

    /**
     * Compute mean squared error: (1/n) * sum((yPred[i] - yTrue[i])^2).
     *
     * @param yPred predicted values, length n
     * @param yTrue ground-truth values, length n
     * @return scalar MSE loss
     */
    static double mseLoss(double[] yPred, double[] yTrue) {
        // TODO: implement MSE: (1/n) * sum((yPred[i] - yTrue[i])^2)
        throw new UnsupportedOperationException("TODO: implement this");
    }

    /**
     * Compute analytical partial derivatives of MSE w.r.t. weight and bias.
     *
     * Model:   yHat[i] = weight * X[i] + bias
     * Loss:    L = (1/n) * sum((yHat[i] - y[i])^2)
     * dL/dw  = (2/n) * sum((yHat[i] - y[i]) * X[i])
     * dL/db  = (2/n) * sum((yHat[i] - y[i]))
     *
     * @return double[] { dL/dw, dL/db }
     */
    static double[] mseGradient(double weight, double bias, double[] X, double[] y) {
        // TODO: compute yHat, then compute dL/dw and dL/db using the formulas above
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // -----------------------------------------------------------------------
    // Task 3 — Gradient Descent Loop
    // -----------------------------------------------------------------------

    /**
     * Run gradient descent on a single-weight linear model with MSE loss.
     *
     * Start: weight = 0.0, bias = 0.0.
     * Each epoch:
     *   1. Compute yHat = weight * X + bias
     *   2. Compute loss  = mseLoss(yHat, y)
     *   3. Compute grads = mseGradient(weight, bias, X, y)
     *   4. weight -= learningRate * grads[0]
     *      bias   -= learningRate * grads[1]
     *
     * @param X            input features, length n
     * @param y            true targets, length n
     * @param learningRate step size alpha
     * @param epochs       number of full-dataset passes
     * @return GDResult containing final weight, bias, and per-epoch loss list
     */
    static GDResult gradientDescent(double[] X, double[] y, double learningRate, int epochs) {
        // TODO: implement the gradient descent loop described above
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // -----------------------------------------------------------------------
    // Task 4 — Gradient Check
    // -----------------------------------------------------------------------

    /**
     * Verify analytical gradient against numerical gradient for weight and bias.
     *
     * For weight: define lossOfWeight(w) = mseLoss(w * X + bias, y), then
     *   g_numerical_w = numericalDerivative(lossOfWeight, weight, h)
     *
     * For bias: define lossOfBias(b) = mseLoss(weight * X + b, y), then
     *   g_numerical_b = numericalDerivative(lossOfBias, bias, h)
     *
     * Relative error = |g_analytical - g_numerical| / (|g_analytical| + |g_numerical| + 1e-8)
     * Print PASS if relative error < 1e-5, else FAIL.
     *
     * @param weight current weight value
     * @param bias   current bias value
     * @param X      input features
     * @param y      true targets
     * @param h      finite-difference step size
     */
    static void gradientCheck(double weight, double bias, double[] X, double[] y, double h) {
        // TODO: compute analytical and numerical gradients for weight and bias,
        //       calculate relative error, and print PASS or FAIL for each.
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // -----------------------------------------------------------------------
    // main — calls all tasks in order
    // -----------------------------------------------------------------------
    public static void main(String[] args) {

        // ------------------------------------------------------------------
        // Task 1: Numerical Derivative
        // ------------------------------------------------------------------
        printSection("Task 1: Numerical Derivative");

        UnivariateFunction f = x -> x * x * x - 4 * x;   // f(x) = x^3 - 4x
        double x0 = 2.0;
        double analyticalDeriv = 3 * x0 * x0 - 4;         // f'(x) = 3x^2 - 4 = 8
        double numericalDeriv  = numericalDerivative(f, x0, 1e-5);
        double absError = Math.abs(analyticalDeriv - numericalDeriv);

        System.out.printf("Analytical f'(%.1f) = %.6f%n", x0, analyticalDeriv);
        System.out.printf("Numerical  f'(%.1f) = %.6f (h=1e-05)%n", x0, numericalDeriv);
        System.out.printf("Absolute error     = %.2e (< 1e-06: %s)%n",
                absError, absError < 1e-6 ? "PASS" : "FAIL");

        // ------------------------------------------------------------------
        // Task 2: Analytical MSE Gradient
        // ------------------------------------------------------------------
        printSection("Task 2: Analytical MSE Gradient");

        double w0 = 0.0, b0 = 0.0;
        double[] yHat0 = new double[N];
        for (int i = 0; i < N; i++) yHat0[i] = w0 * X_DATA[i] + b0;
        double initialLoss = mseLoss(yHat0, Y_DATA);
        double[] grads0    = mseGradient(w0, b0, X_DATA, Y_DATA);

        System.out.printf("Initial weight=%.1f, bias=%.1f%n", w0, b0);
        System.out.printf("  Initial loss = %.4f%n", initialLoss);
        System.out.printf("  dL/dw        = %.4f%n", grads0[0]);
        System.out.printf("  dL/db        = %.4f%n", grads0[1]);

        // ------------------------------------------------------------------
        // Task 3: Gradient Descent
        // ------------------------------------------------------------------
        printSection("Task 3: Gradient Descent");

        GDResult result = gradientDescent(X_DATA, Y_DATA, 0.01, 500);
        int[] logEpochs = {0, 100, 200, 300, 400, 499};
        for (int ep : logEpochs) {
            System.out.printf("Epoch %3d | Loss: %8.4f%n", ep, result.lossHistory().get(ep));
        }
        System.out.printf("%nFinal weight = %.4f  (expected ~3.0)%n", result.weight());
        System.out.printf("Final bias   = %.4f  (expected ~0.5)%n",   result.bias());

        // ------------------------------------------------------------------
        // Task 4: Gradient Check
        // ------------------------------------------------------------------
        printSection("Task 4: Gradient Check");
        gradientCheck(w0, b0, X_DATA, Y_DATA, 1e-5);

        // ------------------------------------------------------------------
        // Task 5 & 6: Plotting
        // ------------------------------------------------------------------
        printSection("Task 5 & 6: Plotting");
        System.out.println("Note: Plotting is not available in the Java starter.");
        System.out.println("Implement Tasks 5 & 6 in starter.py, or export loss history");
        System.out.println("to a CSV and plot with a separate tool.");

        // Demonstrate loss history access:
        System.out.printf("Loss at epoch 0:   %.4f%n", result.lossHistory().get(0));
        System.out.printf("Loss at epoch 499: %.4f%n", result.lossHistory().get(499));

        // LR sensitivity — Task 6
        printSection("Task 6: LR Sensitivity (loss values only)");
        double[] lrs = {0.001, 0.01, 0.1};
        for (double lr : lrs) {
            GDResult r = gradientDescent(X_DATA, Y_DATA, lr, 500);
            System.out.printf("lr=%.3f | final loss: %.4f%n", lr, r.lossHistory().get(499));
        }
    }
}
