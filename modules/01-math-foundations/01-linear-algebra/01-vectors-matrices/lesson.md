# 1.1.1 — Vectors & Matrices: Core Operations

## Hook

A matrix is a typed lookup table — rows are records, columns are fields, and matrix multiplication is a relational join with computed projections: every output cell is a weighted sum of a full row crossed with a full column, exactly as a SQL `SELECT` aggregation folds many input rows into one output value.

## The Problem

Every layer of a neural network is a matrix multiplication. When your model transforms a 768-dimensional embedding into a 10-class output, it is doing `y = Wx + b` — a matrix-vector product. Without fluency in dot products, matrix shapes, and eigenvalues, you cannot reason about why networks are sized the way they are, why gradients vanish, or what PCA actually compresses. These three operations are the `Array.map`, `Array.reduce`, and `Object.keys` of the ML world — everything else is built on top of them.

## Theory

### Dot Product

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i \, b_i$$

- $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$ — two vectors of the same length $n$
- $a_i$ — the $i$-th element of vector $\mathbf{a}$
- The result is a single scalar measuring how much the two vectors "agree" in direction

**Worked example:**

$$\mathbf{a} = [1, 2, 3], \quad \mathbf{b} = [4, 5, 6]$$

$$\mathbf{a} \cdot \mathbf{b} = (1)(4) + (2)(5) + (3)(6) = 4 + 10 + 18 = 32$$

FSD translation: this is the SQL `SUM(a.val * b.val)` aggregation — element-wise multiply then reduce.

---

### Matrix Multiplication

$$(AB)_{ij} = \sum_{k=1}^{m} A_{ik} \, B_{kj}$$

- $A \in \mathbb{R}^{n \times m}$ — left matrix with $n$ rows and $m$ columns
- $B \in \mathbb{R}^{m \times p}$ — right matrix; its row count must match $A$'s column count
- $(AB)_{ij}$ — element at row $i$, column $j$ of the result; it is the dot product of row $i$ of $A$ with column $j$ of $B$
- Result shape: $\mathbb{R}^{n \times p}$

**Worked example** ($2 \times 2$ case):

$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$$

$$AB = \begin{bmatrix} (1)(5)+(2)(7) & (1)(6)+(2)(8) \\ (3)(5)+(4)(7) & (3)(6)+(4)(8) \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

FSD translation: each output cell is one aggregated projection — like computing a `JOIN` between every row of table A and every column of table B and reducing with `SUM(product)`.

---

### Eigenvalue Decomposition

$$A\mathbf{v} = \lambda\mathbf{v}$$

- $A \in \mathbb{R}^{n \times n}$ — a square matrix (e.g., a covariance or weight matrix)
- $\mathbf{v} \in \mathbb{R}^n$ — an **eigenvector**: a special direction that $A$ does *not* rotate, only scales
- $\lambda \in \mathbb{R}$ — the **eigenvalue**: the scaling factor for that direction
- Together $(\lambda, \mathbf{v})$ pairs reveal the principal axes of a transformation

**Worked example** (diagonal matrix — eigenvalues read directly off the diagonal):

$$M = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix}$$

Characteristic equation: $\det(M - \lambda I) = 0$

$$(4 - \lambda)(3 - \lambda) - (2)(2) = 0$$
$$\lambda^2 - 7\lambda + 8 = 0$$
$$\lambda = \frac{7 \pm \sqrt{49 - 32}}{2} = \frac{7 \pm \sqrt{17}}{2}$$
$$\lambda_1 \approx 5.5616, \quad \lambda_2 \approx 1.4384$$

$\lambda_1 \approx 5.56$ tells you that one direction is stretched by a factor of 5.56× — the dominant direction of variance. In PCA this is your first principal component.

## Python Implementation

```python
# Dependencies: numpy>=1.24
import numpy as np
from numpy.linalg import eig
from typing import Tuple


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Return the scalar dot product of two 1-D vectors."""
    return float(np.dot(a, b))


def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiply two 2-D matrices; @ calls np.matmul under the hood."""
    return A @ B


def eigen_decompose(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (eigenvalues, eigenvectors) sorted by descending eigenvalue."""
    eigenvalues, eigenvectors = eig(M)
    # eig() may return complex numbers for non-symmetric matrices;
    # .real is safe here because M is symmetric.
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    idx = np.argsort(eigenvalues)[::-1]          # descending order
    return eigenvalues[idx], eigenvectors[:, idx]


def main() -> None:
    print("=" * 50)
    print("1. Dot Product")
    print("=" * 50)
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    dp = dot_product(a, b)
    print(f"  a           : {a}")
    print(f"  b           : {b}")
    print(f"  a · b       : {dp}")                # 32.0

    print()
    print("=" * 50)
    print("2. Matrix Multiplication")
    print("=" * 50)
    A = np.array([[1, 2],
                  [3, 4]], dtype=float)
    B = np.array([[5, 6],
                  [7, 8]], dtype=float)
    C = matrix_multiply(A, B)
    print(f"  A:\n{A}")
    print(f"  B:\n{B}")
    print(f"  A @ B:\n{C}")
    # [[19. 22.]
    #  [43. 50.]]

    print()
    print("=" * 50)
    print("3. Eigenvalue Decomposition")
    print("=" * 50)
    M = np.array([[4.0, 2.0],
                  [2.0, 3.0]])
    eigenvalues, eigenvectors = eigen_decompose(M)
    print(f"  M:\n{M}")
    print(f"  Eigenvalues (descending) : {np.round(eigenvalues, 4)}")
    print(f"  Eigenvectors (columns)  :\n{np.round(eigenvectors, 4)}")

    # Verify A·v = λ·v for the dominant eigenpair
    v0 = eigenvectors[:, 0]
    lam0 = eigenvalues[0]
    Av   = M @ v0
    lam_v = lam0 * v0
    print(f"\n  Verification M·v₀ = {np.round(Av,   6)}")
    print(f"               λ·v₀ = {np.round(lam_v, 6)}")
    print(f"               Match: {np.allclose(Av, lam_v)}")

    print()
    print("=" * 50)
    print("4. FSD Analogy — Matrix as Lookup Projection")
    print("=" * 50)
    # Each row is a user record [age_score, spend_score] (pre-normalised 0-1)
    users = np.array([[0.25, 0.80],
                      [0.60, 0.45],
                      [0.90, 0.10]])
    # Projection weights: 2 features → 1 risk score
    weights = np.array([[0.7],
                        [0.3]])
    scores = users @ weights           # shape (3, 1)
    print(f"  User records (age_score, spend_score):\n{users}")
    print(f"  Projection weights:\n{weights.T}")
    for i, score in enumerate(scores.flatten()):
        print(f"  User {i} risk score: {score:.4f}")


if __name__ == "__main__":
    main()
```

Run the script with `python lesson.py` (rename as needed). Notice four things in the output:

1. **Dot product** returns 32.0 — confirming `(1×4) + (2×5) + (3×6)` matches the worked example exactly.
2. **Matrix product** yields `[[19, 22], [43, 50]]` — each cell is the dot product of one row from A and one column from B.
3. **Eigenvalues** are ≈ 5.5616 and ≈ 1.4384, confirming the hand-computed quadratic solution above.
4. **Verification** `M·v₀ == λ·v₀` prints `True` — the eigenvector equation holds numerically.
5. **Projection** shows how a `(3×2) @ (2×1)` multiplication collapses two features into one score per user — identical to a dense layer in a neural network.

## Java Implementation

```java
// Dependencies: org.apache.commons:commons-math3:3.6.1
// Maven:
//   <dependency>
//     <groupId>org.apache.commons</groupId>
//     <artifactId>commons-math3</artifactId>
//     <version>3.6.1</version>
//   </dependency>

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class VectorsMatrices {

    /** Return the scalar dot product of two vectors. */
    public static double dotProduct(double[] a, double[] b) {
        RealVector va = new ArrayRealVector(a, false);
        RealVector vb = new ArrayRealVector(b, false);
        return va.dotProduct(vb);
    }

    /** Return the matrix product A × B. */
    public static RealMatrix matrixMultiply(double[][] aData, double[][] bData) {
        RealMatrix A = new Array2DRowRealMatrix(aData, false);
        RealMatrix B = new Array2DRowRealMatrix(bData, false);
        return A.multiply(B);
    }

    /**
     * Return real eigenvalues of a symmetric matrix in descending order.
     * EigenDecomposition requires a symmetric (self-adjoint) matrix.
     */
    public static double[] eigenvalues(double[][] mData) {
        RealMatrix M = new Array2DRowRealMatrix(mData, false);
        EigenDecomposition decomp = new EigenDecomposition(M);
        double[] vals = decomp.getRealEigenvalues();
        // Sort descending in-place
        for (int i = 0; i < vals.length - 1; i++) {
            for (int j = i + 1; j < vals.length; j++) {
                if (vals[j] > vals[i]) {
                    double tmp = vals[i]; vals[i] = vals[j]; vals[j] = tmp;
                }
            }
        }
        return vals;
    }

    public static void main(String[] args) {
        // 1. Dot product
        double[] a = {1.0, 2.0, 3.0};
        double[] b = {4.0, 5.0, 6.0};
        System.out.printf("Dot product a·b: %.1f%n", dotProduct(a, b));  // 32.0

        // 2. Matrix multiplication
        double[][] A = {{1, 2}, {3, 4}};
        double[][] B = {{5, 6}, {7, 8}};
        RealMatrix C = matrixMultiply(A, B);
        System.out.println("Matrix product A × B:");
        for (double[] row : C.getData()) {
            System.out.printf("  [%.0f, %.0f]%n", row[0], row[1]);
        }
        // [19, 22]
        // [43, 50]

        // 3. Eigenvalues of symmetric matrix
        double[][] M = {{4.0, 2.0}, {2.0, 3.0}};
        double[] eigenvalues = eigenvalues(M);
        System.out.print("Eigenvalues (descending): ");
        for (double lambda : eigenvalues) {
            System.out.printf("%.4f  ", lambda);
        }
        System.out.println();
        // 5.5616  1.4384

        // 4. FSD analogy — matrix as projection
        double[][] users  = {{0.25, 0.80}, {0.60, 0.45}, {0.90, 0.10}};
        double[][] weights = {{0.7}, {0.3}};
        RealMatrix scores = matrixMultiply(users, weights);
        System.out.println("User risk scores (age_score·0.7 + spend_score·0.3):");
        double[][] scoreData = scores.getData();
        for (int i = 0; i < scoreData.length; i++) {
            System.out.printf("  User %d: %.4f%n", i, scoreData[i][0]);
        }
    }
}
```

Compile and run with:

```bash
javac -cp commons-math3-3.6.1.jar VectorsMatrices.java
java  -cp .:commons-math3-3.6.1.jar VectorsMatrices
```

## Stack Comparison

| Dimension | Python (NumPy) | Java (Commons Math 3) |
|---|---|---|
| **Library** | `numpy>=1.24` | `commons-math3:3.6.1` |
| **Matrix type** | `np.ndarray` (n-dimensional, any dtype) | `Array2DRowRealMatrix` (2-D, `double` only) |
| **Multiplication** | `A @ B` operator | `A.multiply(B)` method |
| **Dot product** | `np.dot(a, b)` | `va.dotProduct(vb)` |
| **Eigendecomposition** | `numpy.linalg.eig` (any square matrix) | `EigenDecomposition` (symmetric matrices only) |
| **Ecosystem fit** | Primary for ML pipelines, GPU via CuPy | Suited for JVM microservices doing feature scoring |
| **Performance** | BLAS/LAPACK under the hood, vectorised | BLAS not linked by default; slower for large matrices |
| **Shape errors** | `ValueError` at runtime | `DimensionMismatchException` at runtime |

## Key Takeaways

- **Matrix multiplication is a batch dot product** — each output cell $(i, j)$ is the dot product of row $i$ from the left matrix and column $j$ from the right matrix; this is how every dense neural-network layer transforms its input.
- **Eigenvalues measure directional stretch** — $\lambda_1 \approx 5.56$ for matrix $M$ above means the dominant eigenvector direction is amplified 5.56×, which is why PCA ranks principal components by descending eigenvalue.
- **Shape discipline prevents runtime bugs** — for $A \in \mathbb{R}^{n \times m}$ and $B \in \mathbb{R}^{m \times p}$, the inner dimensions ($m$) must match and the output is $\mathbb{R}^{n \times p}$; check shapes before multiplying the same way you check API contract types before calling an endpoint.
