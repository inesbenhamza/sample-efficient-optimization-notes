# Kernel Methods – Lecture Notes

## 1. What is a Kernel?
A **kernel function** is a way to measure similarity between two data points:

\[
K(x_i, x_j) = \phi(x_i)^T \phi(x_j)
\]

- \(x_i, x_j \in \mathbb{R}^d\): feature vectors.  
- \(\phi(x)\): feature mapping (possibly to a higher-dimensional space).  
- Kernels let us compute dot products in that space **without explicitly constructing \(\phi(x)\)** → this is the *kernel trick*.

---

## 2. Kernel Matrix (Gram Matrix)
Given training data \(\{x_1, \dots, x_n\}\), the kernel matrix is:

\[
K \in \mathbb{R}^{n \times n}, \quad K_{ij} = K(x_i, x_j)
\]

- Each entry = similarity between example \(i\) and example \(j\).  
- \(K\) is symmetric and positive semi-definite.

---

## 3. Common Kernels

### Linear Kernel
\[
K(x_i, x_j) = x_i^T x_j
\]

- Just the dot product in the original space.  
- Equivalent to ordinary linear models.

---

### Polynomial Kernel
\[
K(x_i, x_j) = (x_i^T x_j + c)^d
\]

- Expands features into polynomials of degree \(d\).  
- Example: quadratic kernel adds squared and cross terms.

---

### Gaussian / RBF Kernel
\[
K(x_i, x_j) = \exp\!\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)
\]

- Measures closeness in Euclidean distance.  
- Produces smooth, nonlinear decision boundaries.  
- Implicitly corresponds to an **infinite-dimensional** feature space.

---

## 4. Prediction with Kernels

### Kernel Ridge Regression
We express the predictor as:

\[
f(x) = \sum_{i=1}^n \alpha_i K(x, x_i)
\]

- Coefficients \(\alpha\) are learned from training data.  
- Solve:
\[
\alpha = (K + \lambda I)^{-1} y
\]

- Prediction for a new point \(x_*\):
\[
f(x_*) = k_*^T \alpha, \quad k_* = [K(x_*, x_1), \dots, K(x_*, x_n)]^T
\]

---

## 5. Intuition
- **Linear view**: dot products measure similarity in feature space.  
- **Kernel view**: replace dot product with any kernel function → gain nonlinear power.  
- Kernels let models depend only on pairwise similarities, not explicit coordinates.

---

## 6. Connection to Gaussian Processes
Gaussian Processes (GPs) use kernels as **covariance functions**:

\[
\text{Cov}[f(x_i), f(x_j)] = K(x_i, x_j)
\]

- The kernel defines the shape/smoothness of functions sampled from the GP.  
- Same mathematics as kernel regression, but fully Bayesian: gives **uncertainty estimates** as well as predictions.

---

## 7. Tiny Example (Linear Kernel, 1D Data)
Training data:  
- \(x = [1, 2, 3]\)  
- \(y = [1, 2, 2]\)  

Kernel matrix:
\[
K =
\begin{bmatrix}
1 & 2 & 3 \\
2 & 4 & 6 \\
3 & 6 & 9
\end{bmatrix}
\]

For new point \(x_* = 4\):  
\[
k_* = [4, 8, 12]^T
\]

Prediction:
\[
f(x_*) = k_*^T (K + \lambda I)^{-1} y
\]




### Gaussian Process Intuition

A **Gaussian Process (GP)** is a *distribution over functions*.

Formally, we write:

\[
f(x) \sim \mathcal{GP}\big(m(x), k(x, x')\big)
\]

This means that for any finite set of inputs

\[
X = [x_1, x_2, \ldots, x_N],
\]

the corresponding function values

\[
\mathbf{f} = [f(x_1), f(x_2), \ldots, f(x_N)]
\]

follow a **multivariate Gaussian distribution**:

\[
\mathbf{f} \sim \mathcal{N}\big(m(X), K(X, X)\big)
\]

where:
- \( m(X) = [m(x_1), \ldots, m(x_N)] \) is the **mean vector**  
- \( K(X, X) \) is the **covariance matrix** with entries \( K_{ij} = k(x_i, x_j) \)

---

### Intuition

| Symbol | Meaning |
|:-------|:---------|
| \( f(x) \) | Random variable — function value at input \( x \) |
| \( \mathcal{GP}(m, k) \) | Distribution over functions defined by mean \( m(x) \) and covariance \( k(x, x') \) |
| \( m(x) \) | Mean function (expected value of \( f(x) \)) |
| \( k(x, x') \) | Covariance function (how correlated \( f(x) \) and \( f(x') \) are) |
| A sample \( f(x) \) | One possible function drawn from the GP distribution |
| \( \mathbf{f} = [f(x_1), \ldots, f(x_N)] \) | A random vector following a multivariate normal distribution |

---

### Summary

\[
\underbrace{f(x)}_{\text{random variable at input } x}
\quad \sim \quad
\underbrace{\mathcal{GP}\big(m(x), k(x, x')\big)}_{\text{distribution over functions}}
\]

and for any finite input set:

\[
[f(x_1), f(x_2), \ldots, f(x_N)] \sim \mathcal{N}\big(m(X), K(X, X)\big)
\]

---
