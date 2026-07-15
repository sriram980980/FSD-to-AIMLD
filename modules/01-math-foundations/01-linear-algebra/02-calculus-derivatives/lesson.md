# 1.1.2 — Calculus: Derivatives & Gradients

## Hook

A gradient is like a profiler flame graph — it tells you which weights (hotspots) contribute most to your loss (slow render), so you know where to optimize first; just as a flame graph ranks functions by cumulative CPU time, the gradient ranks each weight by how steeply it drives up the loss, pointing you straight to the biggest gains.

## The Problem

Every training loop in ML boils down to one question: "In which direction should I nudge each weight to make the loss smaller?" Without derivatives and gradients, you have no mathematical answer to that question — you would be forced to try random weight perturbations and hope for improvement. The chain rule makes it possible to propagate that directional signal backward through arbitrarily deep compositions of functions (layers), turning what looks like an intractable global search into a local, computable update at every weight in the network.

## Theory

### The Derivative

The derivative of a function $f$ with respect to a variable $x$ measures the instantaneous rate of change of $f$ as $x$ changes:

$$f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

- $f(x)$ — the function being differentiated (e.g., a loss function)
- $x$ — the variable of differentiation (e.g., a single weight)
- $h$ — an infinitesimally small step; the limit formalises "instantaneous"
- $f'(x)$ — the slope of $f$ at point $x$; positive means $f$ is rising, negative means falling

**Worked example:**

Let $f(x) = x^3 - 4x$. The analytical derivative is:

$$f'(x) = 3x^2 - 4$$

At $x = 2$:

$$f'(2) = 3(2)^2 - 4 = 3(4) - 4 = 12 - 4 = 8$$

So $f$ is rising at a rate of 8 units per unit of $x$ when $x = 2$. To minimise $f$, step in the **negative** direction: $x \leftarrow x - \alpha \cdot 8$, where $\alpha$ (the learning rate) controls how large the step is.

---

### Partial Derivatives

When a function has multiple inputs — like a loss that depends on many weights — you compute the **partial derivative** with respect to each input separately, holding all others constant:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}$$

- $x_i$ — the one variable you are differentiating with respect to
- $\frac{\partial}{\partial x_i}$ — Leibniz partial-derivative notation; the curved $\partial$ signals "all other variables frozen"

**Worked example:**

Let $f(x, y) = 3x^2 + 2xy + y^3$.

$$\frac{\partial f}{\partial x} = 6x + 2y \qquad \frac{\partial f}{\partial y} = 2x + 3y^2$$

At $(x, y) = (1, 2)$:

$$\frac{\partial f}{\partial x}\bigg|_{(1,2)} = 6(1) + 2(2) = 10$$

$$\frac{\partial f}{\partial y}\bigg|_{(1,2)} = 2(1) + 3(2)^2 = 2 + 12 = 14$$

$f$ rises 10× faster when nudging $x$ than 1 unit, and 14× faster when nudging $y$ — so $y$ is the bigger hotspot at this point.

---

### The Chain Rule

Neural networks are **compositions** of functions: output = layer3(layer2(layer1(input))). The **chain rule** lets you differentiate such compositions by multiplying local derivatives along the path:

$$\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$$

For a chain of three functions $z = f(y),\ y = g(x)$:

$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$

- $\frac{dz}{dy}$ — how $z$ responds to a change in $y$ (the outer derivative)
- $\frac{dy}{dx}$ — how $y$ responds to a change in $x$ (the inner derivative)
- The product "passes" the gradient signal from output back to input, layer by layer — this is backpropagation

**Worked example (single-neuron MSE):**

Let the prediction be $\hat{y} = wx$ and the loss be $L = (\hat{y} - y)^2 = (wx - y)^2$.

Decompose: $L = u^2$ where $u = wx - y$.

$$\frac{dL}{dw} = \frac{dL}{du} \cdot \frac{du}{dw} = 2u \cdot x = 2(wx - y) \cdot x$$

At $w = 0.5,\ x = 3,\ y = 2$:

$$u = 0.5 \times 3 - 2 = -0.5$$

$$\frac{dL}{dw} = 2(-0.5)(3) = -3$$

The gradient is $-3$: the loss is decreasing as $w$ grows at this point, so the weight update $w \leftarrow w - \alpha(-3) = w + 3\alpha$ will push $w$ upward toward the correct value.

---

### The Gradient

The **gradient** $\nabla L$ is the vector of all partial derivatives of a scalar loss with respect to every weight. It generalises the single-variable derivative to high-dimensional weight spaces:

$$\nabla L(\mathbf{w}) = \begin{bmatrix} \frac{\partial L}{\partial w_1} \\ \frac{\partial L}{\partial w_2} \\ \vdots \\ \frac{\partial L}{\partial w_n} \end{bmatrix}$$

- $\mathbf{w} \in \mathbb{R}^n$ — the weight vector (all learnable parameters)
- $L(\mathbf{w}) \in \mathbb{R}$ — the scalar loss (a single number you want to minimise)
- $\nabla L$ — a vector in $\mathbb{R}^n$ pointing in the direction of **steepest ascent** of $L$; subtract it to descend
- The gradient update rule: $\mathbf{w} \leftarrow \mathbf{w} - \alpha \nabla L(\mathbf{w})$, where $\alpha$ is the learning rate

**Worked example (two weights):**

Let $L(w_1, w_2) = (w_1 x_1 + w_2 x_2 - y)^2$ with $x_1=1,\ x_2=2,\ y=5$.

$$\frac{\partial L}{\partial w_1} = 2(w_1 x_1 + w_2 x_2 - y) \cdot x_1$$

$$\frac{\partial L}{\partial w_2} = 2(w_1 x_1 + w_2 x_2 - y) \cdot x_2$$

At $w_1 = 1,\ w_2 = 1$: prediction $= 1(1) + 1(2) = 3$, residual $= 3 - 5 = -2$.

$$\nabla L = \begin{bmatrix} 2(-2)(1) \\ 2(-2)(2) \end{bmatrix} = \begin{bmatrix} -4 \\ -8 \end{bmatrix}$$

With $\alpha = 0.1$:

$$\mathbf{w} \leftarrow \begin{bmatrix} 1 \\ 1 \end{bmatrix} - 0.1 \begin{bmatrix} -4 \\ -8 \end{bmatrix} = \begin{bmatrix} 1.4 \\ 1.8 \end{bmatrix}$$

New prediction: $1.4(1) + 1.8(2) = 5.0$ — perfect in one step for this linear case.

## Python Implementation

```python
# Dependencies: numpy>=1.24, matplotlib>=3.7
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend; remove for Jupyter/IDE
import matplotlib.pyplot as plt
from typing import Tuple


# ---------------------------------------------------------------------------
# 1. Numerical gradient via finite differences (validates analytical formulas)
# ---------------------------------------------------------------------------

def numerical_gradient(f, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Approximate the gradient of f at x using central differences."""
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad


# ---------------------------------------------------------------------------
# 2. Analytical gradient for MSE loss: L(w) = (w·x - y)²
# ---------------------------------------------------------------------------

def mse_loss(weights: np.ndarray, x: np.ndarray, y: float) -> float:
    """Mean-squared error for a single sample."""
    prediction = float(np.dot(weights, x))
    return (prediction - y) ** 2


def mse_gradient(weights: np.ndarray, x: np.ndarray, y: float) -> np.ndarray:
    """Analytical gradient ∂L/∂w = 2(w·x - y)·x."""
    prediction = float(np.dot(weights, x))
    residual = prediction - y
    return 2 * residual * x


# ---------------------------------------------------------------------------
# 3. Chain rule: L = (wx - y)², single weight, step-by-step
# ---------------------------------------------------------------------------

def chain_rule_demo(w: float, x: float, y: float) -> Tuple[float, float]:
    """Return (loss, dL/dw) using the chain rule decomposition."""
    u = w * x - y              # inner function  u = wx - y
    loss = u ** 2              # outer function  L = u²
    du_dw = x                  # ∂u/∂w = x
    dl_du = 2 * u              # ∂L/∂u = 2u
    dl_dw = dl_du * du_dw      # chain rule: ∂L/∂w = ∂L/∂u · ∂u/∂w
    return loss, dl_dw


# ---------------------------------------------------------------------------
# 4. Gradient descent loop
# ---------------------------------------------------------------------------

def gradient_descent(
    x: np.ndarray,
    y: float,
    learning_rate: float = 0.1,
    epochs: int = 30,
) -> Tuple[np.ndarray, list]:
    """Run gradient descent to minimise MSE; return final weights and loss history."""
    weights = np.zeros(len(x))
    loss_history = []
    for epoch in range(epochs):
        loss = mse_loss(weights, x, y)
        loss_history.append(loss)
        grad = mse_gradient(weights, x, y)
        weights = weights - learning_rate * grad
    return weights, loss_history


def main() -> None:
    print("=" * 60)
    print("1. Derivative — Worked Example: f(x) = x³ - 4x")
    print("=" * 60)
    # Analytical: f'(x) = 3x² - 4  →  f'(2) = 8
    f_scalar = lambda arr: arr[0] ** 3 - 4 * arr[0]
    x_val = np.array([2.0])
    analytical = 3 * (2.0 ** 2) - 4
    numerical = numerical_gradient(f_scalar, x_val)[0]
    print(f"  f(x)  = x³ - 4x  at  x = 2")
    print(f"  Analytical f'(2) : {analytical:.6f}")
    print(f"  Numerical  f'(2) : {numerical:.6f}")
    print(f"  Absolute error   : {abs(analytical - numerical):.2e}")

    print()
    print("=" * 60)
    print("2. Partial Derivatives — f(x, y) = 3x² + 2xy + y³")
    print("=" * 60)
    f_multi = lambda v: 3 * v[0] ** 2 + 2 * v[0] * v[1] + v[1] ** 3
    point = np.array([1.0, 2.0])
    analytical_dx = 6 * point[0] + 2 * point[1]   # = 10
    analytical_dy = 2 * point[0] + 3 * point[1] ** 2  # = 14
    num_grad = numerical_gradient(f_multi, point)
    print(f"  At (x, y) = {tuple(point)}")
    print(f"  ∂f/∂x  analytical: {analytical_dx:.4f}  numerical: {num_grad[0]:.4f}")
    print(f"  ∂f/∂y  analytical: {analytical_dy:.4f}  numerical: {num_grad[1]:.4f}")

    print()
    print("=" * 60)
    print("3. Chain Rule — Single-Neuron MSE: L = (wx - y)²")
    print("=" * 60)
    w, x_single, y_target = 0.5, 3.0, 2.0
    loss, dl_dw = chain_rule_demo(w, x_single, y_target)
    print(f"  w={w}, x={x_single}, y={y_target}")
    print(f"  Prediction (wx)   : {w * x_single:.4f}")
    print(f"  Loss L            : {loss:.4f}")
    print(f"  dL/dw (chain)     : {dl_dw:.4f}")
    # Numerical check
    f_chain = lambda arr: (arr[0] * x_single - y_target) ** 2
    num_dl_dw = numerical_gradient(f_chain, np.array([w]))[0]
    print(f"  dL/dw (numerical) : {num_dl_dw:.4f}")

    print()
    print("=" * 60)
    print("4. Gradient — Two Weights, One Step")
    print("=" * 60)
    weights_init = np.array([1.0, 1.0])
    x_vec = np.array([1.0, 2.0])
    y_val = 5.0
    grad = mse_gradient(weights_init, x_vec, y_val)
    loss_before = mse_loss(weights_init, x_vec, y_val)
    weights_after = weights_init - 0.1 * grad
    loss_after = mse_loss(weights_after, x_vec, y_val)
    print(f"  x={x_vec}, y={y_val}")
    print(f"  Weights before    : {weights_init}")
    print(f"  Gradient ∇L       : {grad}")
    print(f"  Weights after     : {weights_after}")
    print(f"  Loss before       : {loss_before:.4f}")
    print(f"  Loss after        : {loss_after:.4f}")

    print()
    print("=" * 60)
    print("5. Gradient Descent — 30 Epochs")
    print("=" * 60)
    final_weights, loss_history = gradient_descent(x_vec, y_val, learning_rate=0.1, epochs=30)
    print(f"  Final weights     : {np.round(final_weights, 6)}")
    print(f"  Final loss        : {loss_history[-1]:.8f}")
    print(f"  Final prediction  : {np.dot(final_weights, x_vec):.6f}  (target: {y_val})")

    # Plot and save loss curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_history, color="steelblue", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Gradient Descent Convergence")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("loss_curve.png", dpi=120)
    print("\n  Loss curve saved to loss_curve.png")


if __name__ == "__main__":
    main()
```

Run with `python lesson.py`. Notice five things:

1. **Derivative check** — the numerical finite-difference result matches the analytical $f'(2) = 8$ to within $10^{-10}$, confirming the formula.
2. **Partial derivatives** — at $(1, 2)$, $\partial f/\partial y = 14$ is larger than $\partial f/\partial x = 10$: $y$ is the bigger hotspot, exactly as the flame graph analogy implies.
3. **Chain rule dL/dw = −3** — the negative sign means loss decreases as $w$ increases from 0.5, so the weight update moves $w$ upward.
4. **One gradient step** — with two weights the gradient $[-4, -8]$ immediately cuts the loss from 4.0 to 0.0 because the system is linear and $\alpha = 0.1$ happens to be perfectly calibrated here.
5. **Convergence curve** — `loss_curve.png` shows exponential decay over 30 epochs; the rate of decay is controlled by the learning rate.

## Java Implementation

```java
// Dependencies: org.apache.commons:commons-math3:3.6.1
// Maven:
//   <dependency>
//     <groupId>org.apache.commons</groupId>
//     <artifactId>commons-math3</artifactId>
//     <version>3.6.1</version>
//   </dependency>

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.analysis.differentiation.UnivariateDifferentiableFunction;
import org.apache.commons.math3.analysis.differentiation.GradientFunction;
import org.apache.commons.math3.analysis.MultivariateFunction;

import java.util.Arrays;

public class CalculusDerivatives {

    // -----------------------------------------------------------------------
    // 1. Univariate derivative using DerivativeStructure: f(x) = x³ - 4x
    // -----------------------------------------------------------------------

    /**
     * Compute f(x) = x³ - 4x and its first derivative at a given x.
     * DerivativeStructure tracks the value and up to 'order' derivatives.
     *
     * @param x     the point at which to evaluate
     * @param order differentiation order (1 = first derivative)
     * @return DerivativeStructure carrying both f(x) and f'(x)
     */
    public static DerivativeStructure polynomialWithDerivative(double x, int order) {
        // params: number of free variables=1, order, variable index=0, value=x
        DerivativeStructure xDS = new DerivativeStructure(1, order, 0, x);
        // f(x) = x³ - 4x
        return xDS.pow(3).subtract(xDS.multiply(4));
    }

    // -----------------------------------------------------------------------
    // 2. Numerical gradient of a multivariate function (central differences)
    // -----------------------------------------------------------------------

    /**
     * Approximate ∇f at point p using central finite differences.
     *
     * @param f  the scalar multivariate function
     * @param p  the point at which to compute the gradient
     * @param h  step size for finite differences
     * @return gradient vector
     */
    public static double[] numericalGradient(MultivariateFunction f, double[] p, double h) {
        double[] gradient = new double[p.length];
        for (int i = 0; i < p.length; i++) {
            double[] pPlus  = Arrays.copyOf(p, p.length);
            double[] pMinus = Arrays.copyOf(p, p.length);
            pPlus[i]  += h;
            pMinus[i] -= h;
            gradient[i] = (f.value(pPlus) - f.value(pMinus)) / (2.0 * h);
        }
        return gradient;
    }

    // -----------------------------------------------------------------------
    // 3. Analytical gradient for MSE loss: L(w) = (w·x - y)²
    // -----------------------------------------------------------------------

    /**
     * Compute the MSE loss for a single sample.
     *
     * @param weights weight vector
     * @param x       feature vector
     * @param y       target value
     * @return scalar loss
     */
    public static double mseLoss(double[] weights, double[] x, double y) {
        double prediction = 0.0;
        for (int i = 0; i < weights.length; i++) {
            prediction += weights[i] * x[i];
        }
        double residual = prediction - y;
        return residual * residual;
    }

    /**
     * Compute the analytical gradient ∂L/∂w = 2(w·x - y)·x.
     *
     * @param weights weight vector
     * @param x       feature vector
     * @param y       target value
     * @return gradient vector
     */
    public static double[] mseGradient(double[] weights, double[] x, double y) {
        double prediction = 0.0;
        for (int i = 0; i < weights.length; i++) {
            prediction += weights[i] * x[i];
        }
        double residual = prediction - y;
        double[] gradient = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            gradient[i] = 2.0 * residual * x[i];
        }
        return gradient;
    }

    // -----------------------------------------------------------------------
    // 4. Gradient descent loop
    // -----------------------------------------------------------------------

    /**
     * Run gradient descent and return the final weights.
     *
     * @param x            feature vector (single sample)
     * @param y            target value
     * @param learningRate step size α
     * @param epochs       number of update steps
     * @return optimised weight vector
     */
    public static double[] gradientDescent(
            double[] x, double y, double learningRate, int epochs) {
        double[] weights = new double[x.length];   // initialise to zeros
        for (int epoch = 0; epoch < epochs; epoch++) {
            double[] grad = mseGradient(weights, x, y);
            for (int i = 0; i < weights.length; i++) {
                weights[i] -= learningRate * grad[i];
            }
        }
        return weights;
    }

    public static void main(String[] args) {
        // 1. Univariate derivative: f(x) = x³ - 4x at x = 2
        System.out.println("=".repeat(60));
        System.out.println("1. Derivative — f(x) = x³ - 4x  at  x = 2");
        System.out.println("=".repeat(60));
        DerivativeStructure result = polynomialWithDerivative(2.0, 1);
        // getPartialDerivative(1) retrieves the first-order derivative value
        System.out.printf("   f(2)   = %.6f%n", result.getValue());         // 0.0
        System.out.printf("   f'(2)  = %.6f  (analytical: 8.0)%n",
                result.getPartialDerivative(1));                            // 8.0

        // 2. Partial derivatives of f(x, y) = 3x² + 2xy + y³ at (1, 2)
        System.out.println();
        System.out.println("=".repeat(60));
        System.out.println("2. Partial Derivatives — f(x,y) = 3x² + 2xy + y³ at (1,2)");
        System.out.println("=".repeat(60));
        MultivariateFunction fMulti = v -> 3 * v[0] * v[0] + 2 * v[0] * v[1] + v[1] * v[1] * v[1];
        double[] gradMulti = numericalGradient(fMulti, new double[]{1.0, 2.0}, 1e-5);
        System.out.printf("   ∂f/∂x  ≈ %.4f  (analytical: 10.0)%n", gradMulti[0]);
        System.out.printf("   ∂f/∂y  ≈ %.4f  (analytical: 14.0)%n", gradMulti[1]);

        // 3. Chain rule: L = (wx - y)²  →  dL/dw at w=0.5, x=3, y=2
        System.out.println();
        System.out.println("=".repeat(60));
        System.out.println("3. Chain Rule — L = (wx - y)²  at  w=0.5, x=3, y=2");
        System.out.println("=".repeat(60));
        double w = 0.5, xVal = 3.0, yVal = 2.0;
        double u = w * xVal - yVal;
        double loss = u * u;
        double dlDw = 2 * u * xVal;                  // chain rule
        System.out.printf("   Prediction (wx)  : %.4f%n", w * xVal);
        System.out.printf("   Loss L           : %.4f%n", loss);
        System.out.printf("   dL/dw (chain)    : %.4f  (analytical: -3.0)%n", dlDw);

        // 4. Gradient — two weights, one step
        System.out.println();
        System.out.println("=".repeat(60));
        System.out.println("4. Gradient — Two Weights, One Step");
        System.out.println("=".repeat(60));
        double[] weights = {1.0, 1.0};
        double[] x = {1.0, 2.0};
        double y = 5.0;
        double[] grad = mseGradient(weights, x, y);
        double lossBefore = mseLoss(weights, x, y);
        double[] weightsAfter = {weights[0] - 0.1 * grad[0], weights[1] - 0.1 * grad[1]};
        double lossAfter = mseLoss(weightsAfter, x, y);
        System.out.printf("   Gradient ∇L      : [%.2f, %.2f]%n", grad[0], grad[1]);
        System.out.printf("   Weights after    : [%.4f, %.4f]%n",
                weightsAfter[0], weightsAfter[1]);
        System.out.printf("   Loss before      : %.4f%n", lossBefore);
        System.out.printf("   Loss after       : %.4f%n", lossAfter);

        // 5. Gradient descent — 30 epochs
        System.out.println();
        System.out.println("=".repeat(60));
        System.out.println("5. Gradient Descent — 30 Epochs");
        System.out.println("=".repeat(60));
        double[] finalWeights = gradientDescent(x, y, 0.1, 30);
        double finalLoss = mseLoss(finalWeights, x, y);
        double finalPrediction = 0.0;
        for (int i = 0; i < finalWeights.length; i++) finalPrediction += finalWeights[i] * x[i];
        System.out.printf("   Final weights    : [%.6f, %.6f]%n",
                finalWeights[0], finalWeights[1]);
        System.out.printf("   Final loss       : %.8f%n", finalLoss);
        System.out.printf("   Final prediction : %.6f  (target: %.1f)%n", finalPrediction, y);
    }
}
```

Compile and run with:

```bash
javac -cp commons-math3-3.6.1.jar CalculusDerivatives.java
java  -cp .:commons-math3-3.6.1.jar CalculusDerivatives
```

## Stack Comparison

| Dimension | Python (NumPy) | Java (Commons Math 3) |
|---|---|---|
| **Library** | `numpy>=1.24` | `commons-math3:3.6.1` |
| **Derivative tool** | Manual analytical or `jax.grad` / `torch.autograd` | `DerivativeStructure` (forward-mode AD) |
| **Differentiation mode** | Reverse-mode (backprop) in PyTorch; forward in JAX | Forward-mode only via `DerivativeStructure` |
| **Gradient of multivariate f** | `numpy` finite-differences or autograd | Manual finite-differences or `GradientFunction` wrapper |
| **Visualization** | `matplotlib` for loss curves, decision boundaries | No standard equivalent; export data and use a tool |
| **Ecosystem fit** | Primary choice for all ML gradient computation | Useful for JVM services doing numerical optimisation (e.g., hyperparameter search in a Spring Boot service) |
| **Autograd support** | Full autograd via PyTorch/JAX | None — Commons Math is non-differentiable-graph-based |
| **Error on bad input** | `ValueError` / `LinAlgError` | `MathIllegalArgumentException` / `NullPointerException` |

## Key Takeaways

- **The gradient is a ranked hotspot list** — $\nabla L(\mathbf{w})$ tells you the exact magnitude and direction each weight must move to reduce loss fastest; the weight with the largest absolute partial derivative is the performance bottleneck, just as the widest bar in a flame graph is the CPU hotspot.
- **The chain rule is backpropagation** — by decomposing a deep network into a chain of simpler functions and multiplying local derivatives, you propagate the gradient signal from the loss all the way back to the first layer without re-computing anything from scratch; this is why training a 100-layer network is computationally feasible.
- **Learning rate controls stability vs. speed** — a gradient update $\mathbf{w} \leftarrow \mathbf{w} - \alpha \nabla L$ converges when $\alpha$ is small enough that the loss decreases monotonically; too large and the weights overshoot and diverge, exactly like a feedback loop with too-high gain causing oscillation in a control system.
