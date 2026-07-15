// Dependencies (Maven):
// <dependency>
//   <groupId>ai.djl</groupId>
//   <artifactId>api</artifactId>
//   <version>0.26.0</version>
// </dependency>
// <dependency>
//   <groupId>ai.djl.pytorch</groupId>
//   <artifactId>pytorch-engine</artifactId>
//   <version>0.26.0</version>
// </dependency>
// <dependency>
//   <groupId>ai.djl.pytorch</groupId>
//   <artifactId>pytorch-native-auto</artifactId>
//   <version>2.1.1</version>
// </dependency>
//
// Node: 3.1.1 — Multi-Layer Perceptrons — Forward & Backward Pass
// Run (after Maven build):
//   mvn compile exec:java -Dexec.mainClass="StarterAssignment"

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.evaluator.BinaryAccuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Starter code for Assignment 3.1.1 — Build Basic PyTorch MLP (DJL / Java).
 *
 * <p>Complete the three TODO methods in order. Each method has explicit hints showing
 * which DJL APIs to use. Run after every task to see incremental output.
 *
 * <p>Java equivalent of the Python PyTorch MLP using Deep Java Library (DJL).
 */
public class StarterAssignment {

    // ── Implemented Helper ────────────────────────────────────────────────────

    /**
     * Generate a synthetic two-moons dataset as DJL NDArrays.
     *
     * <p>Two interleaved half-circles: class 0 = upper arc, class 1 = lower arc shifted by (1,
     * 0.5). Both classes have Gaussian noise added.
     *
     * @param manager   NDManager that owns the returned arrays
     * @param nSamples  total samples (split evenly between classes)
     * @param noise     standard deviation of Gaussian noise
     * @param seed      random seed for reproducibility
     * @return float[] array {X_flat, y_flat} where X_flat has shape (nSamples, 2) flattened
     *         row-major and y_flat has shape (nSamples, 1) flattened
     */
    public static NDArray[] generateMoonsData(
            NDManager manager, int nSamples, float noise, long seed) {
        Random rng = new Random(seed);
        int half = nSamples / 2;
        float[] xData = new float[nSamples * 2];
        float[] yData = new float[nSamples];

        // Class 0 — upper half-circle (angles 0 → π)
        for (int i = 0; i < half; i++) {
            double angle = Math.PI * i / half;
            xData[i * 2]     = (float) Math.cos(angle) + (float) (rng.nextGaussian() * noise);
            xData[i * 2 + 1] = (float) Math.sin(angle) + (float) (rng.nextGaussian() * noise);
            yData[i] = 0.0f;
        }
        // Class 1 — lower half-circle (angles 0 → π), shifted right and down
        for (int i = 0; i < half; i++) {
            double angle = Math.PI * i / half;
            int idx = half + i;
            xData[idx * 2]     = (float) (1.0 - Math.cos(angle)) + (float) (rng.nextGaussian() * noise);
            xData[idx * 2 + 1] = (float) (0.5 - Math.sin(angle)) + (float) (rng.nextGaussian() * noise);
            yData[idx] = 1.0f;
        }

        NDArray X = manager.create(xData, new Shape(nSamples, 2));
        NDArray y = manager.create(yData, new Shape(nSamples, 1));
        return new NDArray[]{X, y};
    }

    /** Print a labeled metric value with consistent formatting. */
    public static void printMetric(String label, float value) {
        System.out.printf("%-30s %.4f%n", label + ":", value);
    }

    // ── Task 1 ────────────────────────────────────────────────────────────────

    /**
     * Task 1: Build an MLP using DJL's SequentialBlock.
     *
     * <p>Return a SequentialBlock with EXACTLY this architecture:
     * <pre>
     *   Linear(hiddenDim)  → ReLU
     *   Linear(hiddenDim)  → ReLU
     *   Linear(outputDim)  → Sigmoid
     * </pre>
     *
     * <p>DJL hint — SequentialBlock construction:
     * <pre>{@code
     *   SequentialBlock block = new SequentialBlock();
     *   block.add(Linear.builder().setUnits(hiddenDim).build());
     *   block.add(Activation::relu);
     *   block.add(Linear.builder().setUnits(hiddenDim).build());
     *   block.add(Activation::relu);
     *   block.add(Linear.builder().setUnits(outputDim).build());
     *   block.add(Activation::sigmoid);
     *   return block;
     * }</pre>
     *
     * <p>Note: DJL Linear layers infer inputDim automatically from the first batch shape.
     * The Trainer.initialize() call in main() triggers this inference.
     *
     * @param hiddenDim number of neurons in each hidden layer
     * @param outputDim number of output neurons (1 for binary classification)
     * @return configured SequentialBlock (weights not yet allocated)
     */
    public static SequentialBlock buildMlp(int hiddenDim, int outputDim) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this — see Javadoc above");
    }

    // ── Task 2 ────────────────────────────────────────────────────────────────

    /**
     * Task 2: Run the full training loop using DJL's EasyTrain API.
     *
     * <p>DJL wraps the mini-batch loop inside {@link EasyTrain#fit}. Your job is to
     * build the {@link ArrayDataset}, call {@code EasyTrain.fit()}, and collect the
     * per-epoch training loss from the {@link Trainer} metrics.
     *
     * <p>Steps:
     * <ol>
     *   <li>Build an ArrayDataset from X (data) and y (labels) with the given batchSize.</li>
     *   <li>Call {@code dataset.prepare()} to load it into memory.</li>
     *   <li>For each epoch 1..epochs, call {@code EasyTrain.trainDataset(trainer, dataset)}.
     *       After each epoch call {@code trainer.notifyListeners(l -> l.onEpoch(trainer))}
     *       and read the train loss via
     *       {@code trainer.getTrainingResult().getTrainLoss()}.</li>
     *   <li>Print each epoch: {@code "Epoch e/epochs | Train Loss: X.XXXX"}</li>
     * </ol>
     *
     * <p>DJL hint — ArrayDataset construction:
     * <pre>{@code
     *   ArrayDataset dataset = new ArrayDataset.Builder()
     *       .setData(X)
     *       .optLabels(y)
     *       .setSampling(batchSize, true)   // true = shuffle each epoch
     *       .build();
     * }</pre>
     *
     * @param trainer   initialized DJL Trainer bound to the model block
     * @param X         feature tensor, shape (nSamples, 2)
     * @param y         label tensor,   shape (nSamples, 1)
     * @param epochs    number of full dataset passes
     * @param batchSize mini-batch size
     * @return list of per-epoch training losses (length == epochs)
     */
    public static List<Float> trainModel(
            Trainer trainer, NDArray X, NDArray y, int epochs, int batchSize)
            throws IOException, TranslateException {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this — see Javadoc above");
    }

    // ── Task 3 ────────────────────────────────────────────────────────────────

    /**
     * Task 3: Compute binary classification accuracy without gradient tracking.
     *
     * <p>Steps:
     * <ol>
     *   <li>Call {@code model.getBlock().forward(params, new NDList(X), false)} where
     *       {@code params} is {@code model.getBlock().getParameters()} and {@code false}
     *       disables training mode (no gradient computation).</li>
     *   <li>Extract the output NDArray via {@code .singletonOrThrow()}.</li>
     *   <li>Threshold at 0.5: predicted class = {@code (output >= 0.5) ? 1 : 0}.</li>
     *   <li>Compare predictions to y element-wise and compute the fraction that match.</li>
     * </ol>
     *
     * <p>DJL hint — element-wise threshold and comparison:
     * <pre>{@code
     *   NDArray predicted = output.gte(0.5f).toType(DataType.FLOAT32, false);
     *   NDArray correct   = predicted.eq(y);
     *   return correct.mean().getFloat();
     * }</pre>
     *
     * <p>Acceptance threshold: returned accuracy must be >= 0.90 after training.
     *
     * @param model   trained DJL Model
     * @param manager NDManager for temporary sub-arrays
     * @param X       feature tensor, shape (nSamples, 2)
     * @param y       label tensor,   shape (nSamples, 1), values in {0.0, 1.0}
     * @return accuracy in [0, 1]
     */
    public static float evaluateAccuracy(Model model, NDManager manager, NDArray X, NDArray y) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this — see Javadoc above");
    }

    // ── Main ─────────────────────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {
        System.out.println("=== 3.1.1 — Build Basic PyTorch MLP (DJL) ===\n");

        try (NDManager manager = NDManager.newBaseManager()) {

            // Implemented helper — generate synthetic moons data
            NDArray[] data = generateMoonsData(manager, 1000, 0.2f, 42L);
            NDArray X = data[0];  // shape (1000, 2)
            NDArray y = data[1];  // shape (1000, 1)
            System.out.printf("Dataset shape: X=%s  y=%s%n%n", X.getShape(), y.getShape());

            // Task 1 — build model block
            SequentialBlock block = buildMlp(/* hiddenDim= */ 64, /* outputDim= */ 1);
            System.out.println("Model block built successfully.");

            // Configure DJL trainer: Adam optimizer + sigmoid BCE loss + accuracy evaluator
            TrainingConfig config = new DefaultTrainingConfig(
                        Loss.sigmoidBinaryCrossEntropyLoss())
                    .optOptimizer(
                        Optimizer.adam()
                            .optLearningRateTracker(Tracker.fixed(0.001f))
                            .build())
                    .addEvaluator(new BinaryAccuracy())
                    .addTrainingListeners(TrainingListener.Defaults.logging());

            try (Model model = Model.newInstance("mlp-3.1.1")) {
                model.setBlock(block);

                try (Trainer trainer = model.newTrainer(config)) {
                    // Initialize weights — DJL defers allocation until first shape is known
                    trainer.initialize(new Shape(1, 2));
                    System.out.println("Trainer initialized.\n");

                    // Task 2 — full training loop
                    List<Float> trainLosses = trainModel(
                            trainer, X, y, /* epochs= */ 50, /* batchSize= */ 32);

                    System.out.printf("%nFinal Train Loss: %.4f%n", trainLosses.get(trainLosses.size() - 1));

                    // Task 3 — evaluate final accuracy
                    float accuracy = evaluateAccuracy(model, manager, X, y);
                    printMetric("Final Accuracy (full dataset)", accuracy);

                    if (accuracy >= 0.90f) {
                        System.out.println("\n✓ PASS — accuracy >= 0.90");
                    } else {
                        System.out.printf("%n✗ FAIL — accuracy %.4f < 0.90%n", accuracy);
                    }
                }
            }
        }
    }
}
