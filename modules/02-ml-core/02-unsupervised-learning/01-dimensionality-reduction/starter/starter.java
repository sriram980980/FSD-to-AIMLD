// Dependencies (Maven):
// <dependency>
//   <groupId>org.tribuo</groupId>
//   <artifactId>tribuo-clustering-kmeans</artifactId>
//   <version>4.3.1</version>
// </dependency>
// <dependency>
//   <groupId>org.tribuo</groupId>
//   <artifactId>tribuo-math</artifactId>
//   <version>4.3.1</version>
// </dependency>
//
// Node: 2.2.1 — Dimensionality Reduction — PCA & Clustering
// Run: javac StarterAssignment.java && java StarterAssignment
//      (or mvn compile exec:java -Dexec.mainClass=StarterAssignment)

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import org.tribuo.*;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.evaluation.ClusteringEvaluation;
import org.tribuo.clustering.evaluation.ClusteringEvaluator;
import org.tribuo.clustering.kmeans.KMeansModel;
import org.tribuo.clustering.kmeans.KMeansTrainer;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.provenance.SimpleDataSourceProvenance;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Assignment 2.2.1 — Customer Segmentation Pipeline (Java / Tribuo)
 *
 * Implement the four stub methods below. Each method maps to one pipeline stage.
 * Run main() to execute the full pipeline end-to-end.
 */
public class StarterAssignment {

    // ------------------------------------------------------------------
    // IMPLEMENTED HELPER — generate synthetic 2-D customer feature matrix
    // ------------------------------------------------------------------

    /**
     * Returns a (nCustomers × 2) matrix of synthetic customer data.
     * Two features: normalised spend score and normalised visit frequency.
     * Four natural clusters are embedded in the data.
     */
    public static double[][] generateCustomerData(int nCustomers, long seed) {
        Random rng = new Random(seed);
        // Four cluster centres in 2-D
        double[][] centres = {{2.0, 2.0}, {-2.0, 2.0}, {2.0, -2.0}, {-2.0, -2.0}};
        double[][] data = new double[nCustomers][2];
        for (int i = 0; i < nCustomers; i++) {
            int clusterIdx = i % centres.length;
            data[i][0] = centres[clusterIdx][0] + rng.nextGaussian() * 0.8;
            data[i][1] = centres[clusterIdx][1] + rng.nextGaussian() * 0.8;
        }
        return data;
    }

    // ------------------------------------------------------------------
    // STUDENT TASK 1 — Standardize features
    // ------------------------------------------------------------------

    /**
     * Standardize each column of X to mean=0 and std=1.
     *
     * Steps:
     * 1. Compute the column mean and column standard deviation.
     * 2. Subtract the mean and divide by the std for every element.
     * 3. Return a new double[][] of the same shape.
     *
     * Print: "Standardized: cols=<ncols>, row0[0]=<val>"
     *
     * TODO: implement this method.
     */
    public static double[][] standardizeFeatures(double[][] X) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // ------------------------------------------------------------------
    // STUDENT TASK 2 — Build Tribuo Dataset from feature matrix
    // ------------------------------------------------------------------

    /**
     * Convert a raw double[][] feature matrix into a Tribuo MutableDataset<ClusterID>.
     *
     * Steps:
     * 1. Create a MutableDataset<ClusterID> with a suitable DataProvenance.
     * 2. For each row in X create an ArrayExample<ClusterID> with feature names
     *    "f0", "f1", ... and a ClusterID label of 0 (placeholder — K-Means ignores it).
     * 3. Add each example to the dataset and return it.
     *
     * Print: "Dataset built: <n> examples, <d> features"
     *
     * TODO: implement this method.
     */
    public static MutableDataset<ClusterID> buildDataset(double[][] X) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // ------------------------------------------------------------------
    // STUDENT TASK 3 — Train K-Means
    // ------------------------------------------------------------------

    /**
     * Train a KMeansTrainer on the dataset and return the fitted KMeansModel.
     *
     * Steps:
     * 1. Instantiate KMeansTrainer with k clusters, 100 iterations, seed=42,
     *    and KMeansTrainer.Distance.EUCLIDEAN.
     * 2. Call trainer.train(dataset) to get the model.
     * 3. Print the number of centroids (should equal k).
     *
     * Print: "K-Means trained: k=<k> centroids"
     *
     * TODO: implement this method.
     */
    public static KMeansModel trainKMeans(MutableDataset<ClusterID> dataset, int k) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // ------------------------------------------------------------------
    // STUDENT TASK 4 — Evaluate clustering
    // ------------------------------------------------------------------

    /**
     * Evaluate the trained KMeansModel and print cluster assignment counts.
     *
     * Steps:
     * 1. Use model.predict(dataset) to get a List<Prediction<ClusterID>>.
     * 2. Use a ClusteringEvaluator to compute a ClusteringEvaluation.
     * 3. Count how many examples were assigned to each cluster ID.
     * 4. Print a summary of cluster sizes.
     *
     * Print:
     *   "Cluster sizes:"
     *   "  Cluster <id>: <count> customers"
     *
     * Return the ClusteringEvaluation object.
     *
     * TODO: implement this method.
     */
    public static ClusteringEvaluation evaluateClustering(
            KMeansModel model, MutableDataset<ClusterID> dataset) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // ------------------------------------------------------------------
    // MAIN — orchestrates the full pipeline
    // ------------------------------------------------------------------

    public static void main(String[] args) {
        System.out.println("=== Customer Segmentation Pipeline (Java / Tribuo) ===\n");

        // Generate synthetic data (500 customers, 2 features)
        double[][] rawData = generateCustomerData(500, 42L);
        System.out.printf("Raw data: %d customers, %d features%n%n", rawData.length, rawData[0].length);

        // Task 1 — Standardize
        double[][] scaledData = standardizeFeatures(rawData);
        System.out.println();

        // Task 2 — Build Tribuo dataset
        MutableDataset<ClusterID> dataset = buildDataset(scaledData);
        System.out.println();

        // Task 3 — Train K-Means (k=4 matches the 4 embedded clusters)
        int k = 4;
        KMeansModel model = trainKMeans(dataset, k);
        System.out.println();

        // Task 4 — Evaluate
        ClusteringEvaluation evaluation = evaluateClustering(model, dataset);
        System.out.println();

        System.out.println("=== Pipeline complete ===");
    }
}
