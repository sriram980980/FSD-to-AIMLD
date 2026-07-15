// Dependencies (Maven):
//   <dependency>
//     <groupId>ai.djl</groupId><artifactId>api</artifactId><version>0.26.0</version>
//   </dependency>
//   <dependency>
//     <groupId>ai.djl.pytorch</groupId><artifactId>pytorch-engine</artifactId><version>0.26.0</version>
//   </dependency>
//   <dependency>
//     <groupId>ai.djl.pytorch</groupId><artifactId>pytorch-native-auto</artifactId><version>2.1.1</version>
//   </dependency>
//
// Node: 3.2.1 — Attention Mechanisms: Self-Attention
// Build: mvn compile
// Run:   mvn exec:java -Dexec.mainClass="StarterAssignment"

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/**
 * Node 3.2.1 — Attention Mechanisms: Self-Attention (Java / DJL)
 *
 * Complete every method marked with // TODO: to pass all validation checks in main().
 * DJL NDArray mirrors PyTorch Tensor semantics — matMul, softmax, and div behave
 * identically to torch.matmul, torch.softmax, and tensor / scalar.
 */
public class StarterAssignment {

    // -----------------------------------------------------------------------
    // Task 1 — Implement scaled dot-product attention
    // -----------------------------------------------------------------------

    /**
     * Compute Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V.
     *
     * @param query  shape (batch, seq_q, d_k)
     * @param key    shape (batch, seq_k, d_k)
     * @param value  shape (batch, seq_k, d_v)
     * @return NDList of [output (batch, seq_q, d_v), weights (batch, seq_q, seq_k)]
     *
     * Implementation steps:
     *   1. long dK = query.getShape().get(2);
     *   2. scores  = query.matMul(key.transpose(0, 2, 1)).div(Math.sqrt(dK))
     *   3. weights = scores.softmax(2)      // softmax over last axis (key positions)
     *   4. output  = weights.matMul(value)
     *   5. return new NDList(output, weights)
     */
    public static NDList scaledDotProductAttention(
            NDArray query, NDArray key, NDArray value) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement scaledDotProductAttention");
    }

    // -----------------------------------------------------------------------
    // Task 2 — Validate the lesson's 3-token numeric example
    // -----------------------------------------------------------------------

    /**
     * Re-create the 3-token, d_k=2 worked example from the lesson and verify
     * that attention weights for token 0 are within ±0.005 of [0.401, 0.198, 0.401].
     *
     * Q = [[1,0],[0,1],[1,1]]
     * K = [[1,0],[0,1],[1,1]]
     * V = [[0.5,0],[0,0.5],[1,1]]
     *
     * Add a batch dimension before calling scaledDotProductAttention:
     *   Q.reshape(1, 3, 2), K.reshape(1, 3, 2), V.reshape(1, 3, 2)
     *
     * Print:
     *   "Token 0 weights: [w0, w1, w2]"
     *   "PASS" or "FAIL — expected [0.401, 0.198, 0.401]"
     */
    public static void validateNumericExample(NDManager manager) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement validateNumericExample");
    }

    // -----------------------------------------------------------------------
    // Task 3 — Build a causal (lower-triangular) mask
    // -----------------------------------------------------------------------

    /**
     * Create a lower-triangular mask of shape (seqLen, seqLen).
     *
     * mask[i][j] = 1.0f  if j <= i   (token i may attend to token j)
     * mask[i][j] = 0.0f  if j >  i   (token i must NOT see future token j)
     *
     * Implementation hint:
     *   Build a float[] of length seqLen*seqLen, set element [i*seqLen+j]=1 when j<=i,
     *   then manager.create(data, new Shape(seqLen, seqLen)).
     *
     * @param manager NDManager for array creation
     * @param seqLen  sequence length
     * @return NDArray shape (seqLen, seqLen), dtype float32
     */
    public static NDArray makeCausalMask(NDManager manager, int seqLen) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement makeCausalMask");
    }

    // -----------------------------------------------------------------------
    // Task 4 — Apply causal mask inside attention (masked attention)
    // -----------------------------------------------------------------------

    /**
     * Compute attention with a causal mask: future positions receive weight 0.
     *
     * The mask blocks attention by replacing scores at mask==0 positions with
     * Float.NEGATIVE_INFINITY before softmax — exp(-inf)=0, so those weights vanish.
     *
     * @param query  shape (batch, seq_q, d_k)
     * @param key    shape (batch, seq_k, d_k)
     * @param value  shape (batch, seq_k, d_v)
     * @param mask   shape (seq_q, seq_k) — 1.0 = attend, 0.0 = block
     * @return NDList of [output (batch, seq_q, d_v), weights (batch, seq_q, seq_k)]
     *
     * Implementation steps:
     *   1. Compute raw scores = query.matMul(key.transpose(0,2,1)).div(Math.sqrt(dK))
     *   2. Broadcast mask to scores shape, then:
     *        NDArray blocked = scores.where(mask.reshape(1, seqLen, seqLen).eq(1f),
     *                                       manager.full(scores.getShape(), Float.NEGATIVE_INFINITY))
     *      Alternatively use: scores = scores.add(mask.log()) — mask=0 → -inf, mask=1 → 0
     *   3. weights = blocked.softmax(2)
     *   4. output  = weights.matMul(value)
     *   5. return new NDList(output, weights)
     */
    public static NDList maskedAttention(
            NDManager manager,
            NDArray query, NDArray key, NDArray value, NDArray mask) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement maskedAttention");
    }

    // -----------------------------------------------------------------------
    // Task 5 — Shape invariance check
    // -----------------------------------------------------------------------

    /**
     * Verify that self-attention (Q=K=V=X) preserves the input shape exactly.
     *
     * For the test case X of shape (1, 6, 64):
     *   NDArray X = manager.randomNormal(new Shape(1, 6, 64));
     *   NDList result = scaledDotProductAttention(X, X, X);
     *   NDArray output = result.get(0);
     *   assert output.getShape().equals(X.getShape())
     *
     * Print:
     *   "Input (1, 6, 64) → Output <shape>  PASS"  or  "FAIL"
     *
     * Also test shapes (2, 10, 128) and (4, 1, 64). Print one line per case.
     */
    public static void checkShapeInvariance(NDManager manager) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement checkShapeInvariance");
    }

    // -----------------------------------------------------------------------
    // Task 6 — Causal constraint check
    // -----------------------------------------------------------------------

    /**
     * Confirm maskedAttention zeroes all future attention weights.
     *
     * Setup:
     *   int seqLen = 6;
     *   NDArray X    = manager.randomNormal(new Shape(1, seqLen, 64));
     *   NDArray mask = makeCausalMask(manager, seqLen);
     *   NDList result = maskedAttention(manager, X, X, X, mask);
     *   NDArray weights = result.get(1);   // shape (1, seqLen, seqLen)
     *
     * Verification:
     *   For every query position p in [0, seqLen):
     *     for every key position j in (p, seqLen):
     *       weights[0][p][j] must be < 1e-6
     *
     * Count violations. Print:
     *   "Sequence length: 6"
     *   "Future-position violations: <count>"
     *   "PASS — all future attention weights are 0.0"   (count == 0)
     *   "FAIL — <count> future weights are non-zero"    (count > 0)
     */
    public static void checkCausalConstraint(NDManager manager) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement checkCausalConstraint");
    }

    // -----------------------------------------------------------------------
    // Main — orchestrates all tasks in order
    // -----------------------------------------------------------------------

    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("Node 3.2.1 — Self-Attention Mechanisms Assignment (Java)");
        System.out.println("=".repeat(60));

        try (NDManager manager = NDManager.newBaseManager()) {

            System.out.println("\n--- Task 2: Numeric Worked Example Validation ---");
            validateNumericExample(manager);

            System.out.println("\n--- Task 5: Shape Invariance Check ---");
            checkShapeInvariance(manager);

            System.out.println("\n--- Task 6: Causal Constraint Check ---");
            checkCausalConstraint(manager);
        }

        System.out.println("\nAll tasks complete.");
    }
}
