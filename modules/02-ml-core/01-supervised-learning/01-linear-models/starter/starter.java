// Dependencies (Maven):
// <dependency>
//   <groupId>org.tribuo</groupId>
//   <artifactId>tribuo-classification-liblinear</artifactId>
//   <version>4.3.1</version>
// </dependency>
//
// Node: 2.1.1 — Linear Models — Regression & Classification
// Run: javac StarterAssignment.java && java StarterAssignment
//      (with tribuo-classification-liblinear-4.3.1-jar-with-dependencies.jar on classpath)

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

public class StarterAssignment {

    // ── implemented helpers (do not modify) ────────────────────────────────────

    /**
     * Build a synthetic binary classification training dataset (400 samples).
     * Positive samples are drawn from N(+1.5, 1) and negative from N(-1.5, 1)
     * in both feature dimensions.
     */
    static MutableDataset<Label> buildTrainData(LabelFactory labelFactory, Random rng) {
        List<org.tribuo.Example<Label>> examples = new ArrayList<>();
        for (int i = 0; i < 400; i++) {
            boolean positive = (i % 2 == 0);
            double offset = positive ? 1.5 : -1.5;
            double x1 = rng.nextGaussian() + offset;
            double x2 = rng.nextGaussian() + offset;
            String labelStr = positive ? "positive" : "negative";
            ListExample<Label> example = new ListExample<>(new Label(labelStr));
            example.add(new Feature("x1", x1));
            example.add(new Feature("x2", x2));
            examples.add(example);
        }
        var source = new ListDataSource<>(
            examples, labelFactory,
            new SimpleDataSourceProvenance("synthetic-train", labelFactory)
        );
        return new MutableDataset<>(source);
    }

    /**
     * Build a synthetic binary classification test dataset (100 samples).
     * Uses the same generative process as buildTrainData with a different RNG offset.
     */
    static MutableDataset<Label> buildTestData(LabelFactory labelFactory, Random rng) {
        List<org.tribuo.Example<Label>> examples = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            boolean positive = (i % 2 == 0);
            double offset = positive ? 1.5 : -1.5;
            double x1 = rng.nextGaussian() + offset;
            double x2 = rng.nextGaussian() + offset;
            String labelStr = positive ? "positive" : "negative";
            ListExample<Label> example = new ListExample<>(new Label(labelStr));
            example.add(new Feature("x1", x1));
            example.add(new Feature("x2", x2));
            examples.add(example);
        }
        var source = new ListDataSource<>(
            examples, labelFactory,
            new SimpleDataSourceProvenance("synthetic-test", labelFactory)
        );
        return new MutableDataset<>(source);
    }

    // ── student implementations ────────────────────────────────────────────────

    /**
     * Task 1: Apply the sigmoid function σ(z) = 1 / (1 + e^{−z}).
     *
     * @param z any real-valued logit
     * @return probability in the open interval (0, 1)
     *
     * Expected: sigmoid(0.0) → 0.5, sigmoid(0.8) → ≈0.6900
     */
    static double sigmoid(double z) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this");
    }

    /**
     * Task 2: Compute Mean Squared Error — (1/n) * Σ (yTrue_i − yPred_i)².
     *
     * @param yTrue ground-truth continuous targets, length n
     * @param yPred model predictions, length n
     * @return scalar MSE ≥ 0
     *
     * Expected: mseLoss([3.0, 5.0], [2.5, 4.5]) → 0.25
     */
    static double mseLoss(double[] yTrue, double[] yPred) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this");
    }

    /**
     * Task 3: Compute Binary Cross-Entropy — −(1/n) Σ [y_i log(p_i) + (1−y_i) log(1−p_i)].
     * Clip probabilities into [eps, 1−eps] with eps=1e-12 before taking log.
     *
     * @param yTrue binary labels in {0.0, 1.0}, length n
     * @param yProb predicted probabilities in (0, 1), length n
     * @return scalar cross-entropy ≥ 0
     *
     * Expected: crossEntropyLoss([1.0], [0.6900]) → ≈0.3711
     */
    static double crossEntropyLoss(double[] yTrue, double[] yProb) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this");
    }

    /**
     * Task 4: Train a Tribuo LibLinear logistic regression model and print evaluation metrics.
     *
     * Steps:
     *   1. Construct a LibLinearClassificationTrainer with default settings.
     *   2. Call trainer.train(trainData) to fit the model.
     *   3. Use LabelEvaluator to evaluate on testData.
     *   4. Print accuracy and macro-averaged F1 using the format shown below.
     *   5. Predict the label for a single held-out sample (x1=1.5, x2=1.5) and print it.
     *
     * Expected printed format:
     *   Train samples: 400 | Test samples: 100
     *   Accuracy : 0.XXXX
     *   Macro-F1 : 0.XXXX
     *   Sample (x1=1.5, x2=1.5) → predicted: <label>
     *
     * @param trainData Tribuo training dataset
     * @param testData  Tribuo test dataset
     */
    static void trainAndEvaluate(
        MutableDataset<Label> trainData,
        MutableDataset<Label> testData
    ) throws Exception {
        System.out.printf("Train samples: %d | Test samples: %d%n",
            trainData.size(), testData.size());
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // ── main ──────────────────────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {
        System.out.println("=".repeat(60));
        System.out.println("SANITY CHECKS — pure math (no dependencies)");
        System.out.println("=".repeat(60));

        // Task 1: sigmoid
        System.out.printf("sigmoid(0.0) = %.4f  (expected 0.5000)%n", sigmoid(0.0));
        System.out.printf("sigmoid(0.8) = %.4f  (expected ≈0.6900)%n", sigmoid(0.8));

        // Task 2: MSE
        double mse = mseLoss(new double[]{3.0, 5.0}, new double[]{2.5, 4.5});
        System.out.printf("MSE worked example = %.4f  (expected 0.2500)%n", mse);

        // Task 3: cross-entropy
        double ce = crossEntropyLoss(new double[]{1.0}, new double[]{0.6900});
        System.out.printf("Cross-Entropy worked ex = %.4f  (expected ≈0.3711)%n%n", ce);

        System.out.println("=".repeat(60));
        System.out.println("TRIBUO — LibLinear Logistic Regression");
        System.out.println("=".repeat(60));

        // Task 4: Tribuo training and evaluation
        LabelFactory labelFactory = new LabelFactory();
        Random rngTrain = new Random(42);
        Random rngTest  = new Random(99);

        MutableDataset<Label> trainData = buildTrainData(labelFactory, rngTrain);
        MutableDataset<Label> testData  = buildTestData(labelFactory, rngTest);

        trainAndEvaluate(trainData, testData);
    }
}
