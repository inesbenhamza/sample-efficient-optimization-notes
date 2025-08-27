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

---
