// Dependencies (Maven):
//   <dependency>
//     <groupId>ai.djl</groupId>
//     <artifactId>api</artifactId>
//     <version>0.26.0</version>
//   </dependency>
//   <dependency>
//     <groupId>ai.djl.pytorch</groupId>
//     <artifactId>pytorch-engine</artifactId>
//     <version>0.26.0</version>
//     <scope>runtime</scope>
//   </dependency>
//
// Node: 3.1.2 — Optimization — Gradient Descent Variants
// Run: mvn compile exec:java -Dexec.mainClass="StarterAssignment"
//
// This file mirrors the Python assignment:
//   Task 1 — buildModel()         : construct a 2-layer MLP via DJL SequentialBlock
//   Task 2 — makeOptimizer()      : configure SGD / SGD+Momentum / Adam via DJL Optimizer
//   Task 3 — trainOneRun()        : standard DJL training loop, return per-epoch losses
//   Task 4 — runGridSearch()      : iterate all optimizer × lr combinations
//   Task 5 — printResultsTable()  : tabular console output (no image output in Java)

import ai.djl.Model;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.Sgd;
import ai.djl.training.tracker.Tracker;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class StarterAssignment {

    // ── Constants — match the Python experiment grid exactly ─────────────────

    static final String[] OPTIMIZER_NAMES = {"SGD", "SGD+Momentum", "Adam"};
    static final double[] LEARNING_RATES  = {0.1, 0.01, 0.001, 0.0001, 0.00001};
    static final int      EPOCHS          = 200;

    // ── Data container for one grid-search result ─────────────────────────────

    record RunResult(String optimizerName, double learningRate, double finalLoss) {}

    // ── Fully implemented helper — do not modify ──────────────────────────────

    static String formatLr(double lr) {
        return String.format("%.0e", lr);
    }

    static RunResult findBest(List<RunResult> results) {
        return results.stream()
                .min(Comparator.comparingDouble(r -> r.finalLoss()))
                .orElseThrow(() -> new IllegalStateException("Results list is empty"));
    }

    // ── Task 1 — Implement buildModel ─────────────────────────────────────────
    //
    // Return a DJL SequentialBlock representing:
    //   Linear(64) → ReLU → Linear(32) → ReLU → Linear(1) → Sigmoid
    //
    // Each call must produce an independent block (fresh weights after training config reset).
    // Use:
    //   SequentialBlock block = new SequentialBlock();
    //   block.add(Linear.builder().setUnits(64).build());
    //   block.add(Activation::relu);
    //   ... and so on.

    static Block buildModel() {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // ── Task 2 — Implement makeOptimizer ──────────────────────────────────────
    //
    // Build and return a DJL Optimizer for the given name and learning rate.
    //
    // "SGD"          → Sgd.builder().setLearningRateTracker(Tracker.fixed((float) lr)).build()
    // "SGD+Momentum" → Sgd.builder().optMomentum(0.9f)
    //                              .setLearningRateTracker(Tracker.fixed((float) lr)).build()
    // "Adam"         → Adam.builder().optLearningRateTracker(Tracker.fixed((float) lr)).build()
    //
    // Throw IllegalArgumentException for any unrecognised name.

    static Optimizer makeOptimizer(String optimizerName, double learningRate) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // ── Task 3 — Implement trainOneRun ────────────────────────────────────────
    //
    // Train the model for `epochs` steps using SoftmaxCrossEntropyLoss (or BinaryCrossEntropy
    // approximation — use Loss.sigmoidBinaryCrossEntropyLoss()).
    //
    // Because DJL's full Dataset/DataManager pipeline is verbose for 4-sample XOR,
    // implement a simplified manual loop:
    //   for each epoch:
    //     GradientCollector: forward → loss → backward → optimizer step
    //     record loss.getFloat() in the returned list
    //
    // Return a List<Double> of length `epochs`.
    //
    // Hint: use try (GradientCollector gc = Engine.getInstance().newGradientCollector())
    //       inside the loop to scope gradients.

    static List<Double> trainOneRun(
            Block model,
            Optimizer optimizer,
            int epochs
    ) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // ── Task 4 — Implement runGridSearch ──────────────────────────────────────
    //
    // Iterate all (optimizerName, learningRate) pairs:
    //   1. Call buildModel() for a fresh block.
    //   2. Call makeOptimizer() for the current combination.
    //   3. Call trainOneRun() and take the last element as finalLoss.
    //   4. Print progress:
    //      System.out.printf("Running %-14s lr=%-8s ...  final loss: %.4f%n",
    //                        name, formatLr(lr), finalLoss);
    //   5. Add a RunResult to the returned list.
    //
    // The returned list must have OPTIMIZER_NAMES.length × LEARNING_RATES.length entries.

    static List<RunResult> runGridSearch() {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // ── Task 5 — Implement printResultsTable ──────────────────────────────────
    //
    // Print a formatted ASCII table to stdout:
    //   - Header row: "Optimizer" then each learning rate via formatLr()
    //   - One data row per optimizer: optimizer name then final loss per lr (4 decimal places)
    //
    // Example (partial):
    //   Optimizer       | 1e-01  | 1e-02  | 1e-03  | 1e-04  | 1e-05
    //   SGD             | 0.6931 | 0.5843 | 0.6893 | 0.6931 | 0.6931
    //   SGD+Momentum    | 0.0541 | 0.2174 | 0.6730 | 0.6930 | 0.6931
    //   Adam            | 0.0138 | 0.0201 | 0.0689 | 0.4112 | 0.6901

    static void printResultsTable(List<RunResult> results) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // ── main — orchestrates all tasks ────────────────────────────────────────

    public static void main(String[] args) {
        System.out.printf(
            "=== Hyperparameter Grid Search — %d optimizers × %d learning rates × %d epochs ===%n%n",
            OPTIMIZER_NAMES.length, LEARNING_RATES.length, EPOCHS
        );

        // Task 4: run grid
        List<RunResult> results = runGridSearch();

        // Task 5: print table
        System.out.println();
        printResultsTable(results);

        // Task 6: report best
        RunResult best = findBest(results);
        System.out.printf(
            "%nBest combination: %-14s lr=%-8s  final loss: %.4f%n",
            best.optimizerName(), formatLr(best.learningRate()), best.finalLoss()
        );
    }
}
