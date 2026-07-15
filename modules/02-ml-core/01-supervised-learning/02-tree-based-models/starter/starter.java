// Dependencies (Maven — add to pom.xml):
//
// <dependency>
//   <groupId>org.tribuo</groupId>
//   <artifactId>tribuo-classification-xgboost</artifactId>
//   <version>4.3.1</version>
// </dependency>
//
// Node: 2.1.2 — Tree-Based Models & Ensembles
// Run: mvn compile exec:java -Dexec.mainClass=StarterAssignment
//
// PREREQUISITE: Run `python starter.py` first to generate data/wine.csv.
// The CSV must have numeric feature columns and a "target" column with class names
// (class_0, class_1, class_2) produced by the Python export_wine_csv() helper.

import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.xgboost.XGBoostClassificationTrainer;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.evaluation.TrainTestSplitter;

import java.nio.file.Paths;

/**
 * Node 2.1.2 — Tree-Based Models & Ensembles (Java)
 *
 * Mirrors the Python starter tasks using Tribuo 4.3 + XGBoost JNI bindings.
 * Implement each TODO method, then run main() to verify your output.
 */
public class StarterAssignment {

    // ── Implemented helper ───────────────────────────────────────────────────

    /**
     * Load data/wine.csv and produce an 80/20 stratified train/test split.
     * Expects a "target" column with string class labels (class_0, class_1, class_2).
     */
    public static TrainTestSplitter<Label> loadAndSplit(String csvPath) throws Exception {
        var labelFactory = new LabelFactory();
        var csvLoader    = new CSVLoader<>(labelFactory);
        var dataSource   = csvLoader.loadDataSource(Paths.get(csvPath), "target");
        // 0.8 = train fraction, 42L = random seed
        return new TrainTestSplitter<>(dataSource, 0.8, 42L);
    }

    /**
     * Evaluate a trained model against test data and print labeled metrics.
     * Returns accuracy as a double.
     */
    public static double evaluateAndPrint(String name, Model<Label> model, Dataset<Label> testData) {
        var evaluator  = new LabelEvaluator();
        LabelEvaluation eval = evaluator.evaluate(model, testData);
        System.out.printf("%n%s%n", "─".repeat(50));
        System.out.printf("Model       : %s%n", name);
        System.out.printf("Accuracy    : %.4f%n", eval.accuracy());
        System.out.printf("Macro F1    : %.4f%n", eval.macroAveragedF1());
        System.out.println(eval);
        return eval.accuracy();
    }

    // ── Student stubs ────────────────────────────────────────────────────────

    /**
     * Train an XGBoost classifier (300 rounds) and return the trained model.
     *
     * TODO: Implement this method.
     * Requirements:
     * - Create an XGBoostClassificationTrainer with numRounds = 300.
     * - Call trainer.train(trainData) and capture the result.
     * - Call evaluateAndPrint("XGBoost (300 rounds)", model, testData).
     * - Return the trained model.
     */
    public static Model<Label> trainXGBoost(
            Dataset<Label> trainData,
            Dataset<Label> testData) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement trainXGBoost");
    }

    /**
     * Train a 1-round single-tree baseline and a 200-round ensemble; compare accuracies.
     *
     * TODO: Implement this method.
     * Requirements:
     * - Create XGBoostClassificationTrainer(1) for the baseline.
     * - Train on trainData, evaluate on testData via evaluateAndPrint().
     * - Create XGBoostClassificationTrainer(200) for the ensemble.
     * - Train on trainData, evaluate on testData via evaluateAndPrint().
     * - Print whether the ensemble accuracy exceeds the baseline:
     *     System.out.printf("Ensemble beats baseline : %b%n", ensembleAcc > baselineAcc);
     */
    public static void compareBaselineVsEnsemble(
            Dataset<Label> trainData,
            Dataset<Label> testData) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement compareBaselineVsEnsemble");
    }

    /**
     * Manually search over numRounds values and print the best one.
     *
     * TODO: Implement this method.
     * Requirements:
     * - Iterate over numRounds in {50, 100, 200, 300, 500}.
     * - For each value, create an XGBoostClassificationTrainer(numRounds),
     *   train on trainData, and evaluate accuracy on testData.
     * - Print each result:
     *     System.out.printf("  numRounds=%-4d  accuracy=%.4f%n", numRounds, acc);
     * - After the loop, print the best numRounds and its accuracy:
     *     System.out.printf("Best numRounds : %d  (accuracy=%.4f)%n", bestRounds, bestAcc);
     */
    public static void tuneNumRounds(
            Dataset<Label> trainData,
            Dataset<Label> testData) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement tuneNumRounds");
    }

    // ── Entrypoint ───────────────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {
        System.out.println("=".repeat(50));
        System.out.println("Node 2.1.2 — Tree-Based Models & Ensembles (Java)");
        System.out.println("=".repeat(50));

        // Task 1 — Load CSV produced by starter.py
        System.out.println("\n[Task 1] Loading Wine CSV dataset");
        var splitter  = loadAndSplit("data/wine.csv");
        MutableDataset<Label> trainData = new MutableDataset<>(splitter.getTrain());
        MutableDataset<Label> testData  = new MutableDataset<>(splitter.getTest());
        System.out.printf("Train size  : %d | Test size : %d%n",
                trainData.size(), testData.size());

        // Task 2 — Train XGBoost classifier
        System.out.println("\n[Task 2] XGBoost Classifier (300 rounds)");
        Model<Label> xgbModel = trainXGBoost(trainData, testData);

        // Task 3 — Compare single-tree baseline vs 200-round ensemble
        System.out.println("\n[Task 3] Baseline vs Ensemble Comparison");
        compareBaselineVsEnsemble(trainData, testData);

        // Task 4 — Tune numRounds hyperparameter
        System.out.println("\n[Task 4] Hyperparameter Tuning (numRounds)");
        tuneNumRounds(trainData, testData);

        System.out.println("\n" + "=".repeat(50));
        System.out.println("All tasks complete.");
    }
}
