# Neural Networks and ML – Concepts and Mathematics

Neural networks in machine learning are built from a small set of core mathematical ideas: linear algebra for representing neurons and layers, calculus for backpropagation, probability/statistics for loss functions and generalization, and optimization for gradient-based training.[web:28][web:31] This document is a self-contained study file and roadmap to master the theory.

---

## 1. Mathematical Prerequisites

Neural networks are parametric functions \(f_\theta\) that map inputs to outputs using matrices (weights) and vectors (biases). Linear algebra, calculus, probability, and optimization provide the tools to define and train these functions.[web:28][web:31]

- **Linear algebra basics**[web:28]  
  - Vectors in \(\mathbb{R}^n\), inner product \(\langle x, y \rangle = \sum_i x_i y_i\).  
  - Matrices \(W \in \mathbb{R}^{m \times n}\), matrix–vector product \(Wx\).  
  - Norms: \(\|x\2 = \sqrt{\sum_i x_i^2}\), \(\|x\|_1 = \sum_i |x_i|\).  
  - Eigenvalues/eigenvectors: \(Av = \lambda v\); SVD, \(A = U\Sigma V^\top\).[web:28]

- **Multivariable calculus**[web:28][web:31]  
  - Partial derivative: \(\frac{\partial f}{\partial x_i}\).  
  - Gradient: \(\nabla_x f(x) = \left[\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n}\right]^\top\).[web:28]  
  - Chain rule: for \(y = g(u), u = h(x)\), \(\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}\).[web:31]

- **Probability and statistics**[web:28]  
  - Random variable \(X\), expectation \(\mathbb{E}[X]\), variance \(\mathrm{Var}(X)\).  
  - Distributions (Bernoulli, Gaussian), likelihood \(p_\theta(y|x)\).  
  - Law of large numbers: empirical averages \(\to\) expectations as \(n \to \infty\).

- **Optimization**[web:28][web:27]  
  - Objective \(J(\theta)\), unconstrained minimization \(\min_\theta J(\theta)\).  
  - Gradient descent: \(\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t)\).[web:27]  
  - Stochastic gradient descent: gradient on minibatches to approximate full gradient.

---

## 2. Core Neural Network Model

A feedforward neural network is a composition of affine maps and nonlinear activations applied layer by layer.[web:24][web:30]

- **Single neuron**[web:24][web:26]  
  - Inputs \(x \in \mathbb{R}^n\), weights \(w \in \mathbb{R}^n\), bias \(b \in \mathbb{R}\).  
  - Pre-activation: \(z = w^\top x + b\).[web:24]  
  - Activation (output): \(a = \sigma(z)\), where \(\sigma\) is an activation function.[web:24]

- **Layer and network equations**[web:24][web:26][web:30]  
  - Layer \(l\) with weights \(W^l \in \mathbb{R}^{m_l \times m_{l-1}}\), bias \(b^l \in \mathbb{R}^{m_l}\).  
  - Pre-activation: \(z^l = W^l a^{l-1} + b^l\).  
  - Activation: \(a^l = \sigma^l(z^l)\).  
  - Network of depth \(L\): \(f_\theta(x) = a^L\), with \(a^0 = x\).[web:24][web:30]

- **Parameter set**[web:30]  
  - \(\theta = \{W^1, b^1, \dots, W^L, b^L\}\).  
  - Number of parameters \(= \sum_{l=1}^L (m_l m_{l-1} + m_l)\).

---

## 3. Activation Functions and Their Derivatives

Activation functions introduce **nonlinearity**, allowing networks to approximate complex functions beyond linear models.[web:24][web:26]

- **Sigmoid**[web:24][web:31]  
  - \(\sigma(z) = \frac{1}{1 + e^{-z}}\).  
  - Derivative: \(\sigma'(z) = \sigma(z)(1 - \sigma(z))\).[web:31]  
  - Used historically in binary classification; prone to vanishing gradients for large \(|z|\).

- **Tanh**[web:24][web:26]  
  - \(\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}\).  
  - Derivative: \(\tanh'(z) = 1 - \tanh^2(z)\).[web:31]  
  - Zero-centered but still can saturate.

- **ReLU (Rectified Linear Unit)**[web:24][web:26]  
  - \(\mathrm{ReLU}(z) = \max(0, z)\).  
  - Derivative: \(\mathrm{ReLU}'(z) = 1\) if \(z > 0\), \(0\) if \(z < 0\).[web:31]  
  - Sparse activations; helps with vanishing gradient issues; not differentiable at 0 but subgradient used.

- **Leaky ReLU and variants**[web:26]  
  - \(\mathrm{LeakyReLU}(z) = \max(\alpha z, z)\) with small \(\alpha > 0\).  
  - Derivative: \(\alpha\) for \(z < 0\), \(1\) for \(z > 0\).

- **Softmax (multi-class output)**[web:24][web:30]  
  - For logits \(z \in \mathbb{R}^K\),  
    \[
    \mathrm{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}.
    \]  
  - Jacobian:  
    \[
    \frac{\partial \mathrm{softmax}(z)_k}{\partial z_j} =
    \begin{cases}
      \mathrm{softmax}(z)_k (1 - \mathrm{softmax}(z)_k), & j = k,\\
      -\mathrm{softmax}(z)_k \mathrm{softmax}(z)_j, & j \neq k.
    \end{cases}
    \][web:31]

---

## 4. Loss Functions and Empirical Risk

Training minimizes an **empirical risk**: average loss over training data.[web:28][web:30]

- **Dataset and model**[web:30]  
  - Data: \(\{(x^{(i)}, y^{(i)})\}_{i=1}^N\).  
  - Network output: \(a^{L(i)} = f_\theta(x^{(i)})\).  
  - Empirical risk:  
    \[
    J(\theta) = \frac{1}{N}\sum_{i=1}^N L(y^{(i)}, a^{L(i)}).
    \]

- **Mean Squared Error (MSE)**[web:28]  
  - For regression:  
    \[
    L(y, a) = \frac{1}{2}\|a - y\|_2^2 = \frac{1}{2}\sum_j (a_j - y_j)^2.
    \]  
  - Gradient w.r.t. activations: \(\nabla_a L = a - y\).[web:31]

- **Binary cross-entropy**[web:28][web:30]  
  - For \(y \in \{0,1\}\), prediction \(p = \sigma(z)\):  
    \[
    L(y,p) = -\big(y \log p + (1-y)\log(1-p)\big).
    \]  
  - \(\frac{\partial L}{\partial z} = p - y\).  

- **Categorical cross-entropy (with softmax)**[web:28][web:30]  
  - One-hot \(y \in \{0,1\}^K\), \(p = \mathrm{softmax}(z)\):  
    \[
    L(y,p) = -\sum_{k=1}^K y_k \log p_k.
    \]  
  - Gradient of loss w.r.t. logits: \(\nabla_z L = p - y\).[web:31]

- **Regularized cost**[web:28]  
  - L2 (weight decay):  
    \[
    J_\lambda(\theta) = J(\theta) + \frac{\lambda}{2}\sum_l \|W^l\|_F^2.
    \]  
  - Adds \(\lambda W^l\) term to gradient.

---

## 5. Backpropagation Mathematics

Backpropagation uses the chain rule to compute gradients of the loss with respect to all parameters efficiently.[web:27][web:31][web:29]

### 5.1 Forward Pass

Compute all intermediate quantities from input to output.[web:24][web:27]

- Given input \(x = a^0\).  
- For each layer \(l = 1, \dots, L\):  
  - \(z^l = W^l a^{l-1} + b^l\).  
  - \(a^l = \sigma^l(z^l)\).  
- Compute loss \(L(y, a^L)\).

### 5.2 Backward Pass – Error Terms

Define error at layer \(l\) as \(\delta^l = \frac{\partial L}{\partial z^l}\).[web:31][web:29]

- **Output layer**[web:31]  
  - For general activation \(\sigma^L\):  
    \[
    \delta^L = \nabla_{a^L} L \odot (\sigma^L)'(z^L),
    \]  
    where \(\odot\) is elementwise product.  
  - For MSE + linear output: \(\delta^L = a^L - y\).[web:31]  
  - For softmax + cross-entropy: \(\delta^L = a^L - y\).[web:31]

- **Hidden layers**[web:31][web:29]  
  - For layer \(l = L-1, \dots, 1\):  
    \[
    \delta^l = (W^{l+1})^\top \delta^{l+1} \odot (\sigma^l)'(z^l).
    \]

### 5.3 Gradients w.r.t. Parameters

Use chain rule to relate errors to weights and biases.[web:31][web:29]

- For weights:  
  \[
  \frac{\partial L}{\partial W^l} = \delta^l (a^{l-1})^\top.
  \]  
- For biases:  
  \[
  \frac{\partial L}{\partial b^l} = \delta^l.
  \]  
- For minibatch of size \(m\), average gradients over samples.

### 5.4 Full Training Objective

Training seeks \(\theta^*\) minimizing expected loss.[web:30][web:27]

- Population risk (intractable):  
  \[
  R(\theta) = \mathbb{E}_{(x,y)\sim \mathcal{D}}[L(y, f_\theta(x))].
  \]  
- Empirical risk:  
  \[
  J(\theta) = \frac{1}{N}\sum_{i=1}^N L(y^{(i)}, f_\theta(x^{(i)})).
  \]  
- Backprop computes \(\nabla_\theta J(\theta)\) from a sample or minibatch.[web:27][web:31]

---

## 6. Gradient Descent and Variants

Optimization algorithms update parameters using gradients from backpropagation to reduce loss.[web:27][web:28]

- **Batch gradient descent**[web:27]  
  - Update rule:  
    \[
    \theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t),
    \]  
    where gradient uses all \(N\) samples each step.

- **Stochastic Gradient Descent (SGD)**[web:27]  
  - Use one sample or small minibatch \(\mathcal{B}\):  
    \[
    \theta_{t+1} = \theta_t - \eta \frac{1}{|\mathcal{B}|}\sum_{(x,y)\in\mathcal{B}} \nabla_\theta L(y, f_\theta(x)).
    \]  
  - Introduces stochasticity, helps escape shallow minima.[web:27]

- **Momentum**[web:27]  
  - Velocity \(v_t = \mu v_{t-1} + \nabla_\theta J(\theta_t)\).  
  - Update: \(\theta_{t+1} = \theta_t - \eta v_t\).  
  - Smooths noisy gradients and accelerates in consistent directions.

- **Adam (Adaptive Moment Estimation)**[web:27]  
  - Exponential moving averages of gradient and squared gradient.  
  - \(m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t\); \(v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2\).  
  - Bias-corrected and scaled update: \(\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}\).

---

## 7. Regularization and Generalization

Regularization controls model complexity to improve generalization on unseen data.[web:28][web:30]

- **L2 regularization (weight decay)**[web:28]  
  - Add \(\frac{\lambda}{2}\sum_l \|W^l\|_F^2\) to loss.  
  - Gradient for \(W^l\) becomes \(\frac{\partial J}{\partial W^l} + \lambda W^l\).  
  - Encourages small weights and smoother functions.

- **L1 regularization**[web:28]  
  - Add \(\lambda \sum_l \|W^l\|_1\).  
  - Promotes sparsity in weights.

- **Dropout (concept level math)**[web:30]  
  - Random mask \(m^l \sim \mathrm{Bernoulli}(p)\) applied to activations: \(\tilde a^l = m^l \odot a^l\).  
  - At test time, use expectation by scaling weights/activations.

- **Bias–variance view**[web:28][web:30]  
  - Expected prediction error decomposes into bias, variance, and noise.  
  - Overparameterized networks rely on optimization and regularization to control effective capacity.

---

## 8. Key Supervised ML Models (Math View)

Neural networks sit inside the broader ML family; comparing their math to classical models clarifies relationships.[web:28][web:30]

### Title: Core supervised models and objectives

| Model type        | Hypothesis form                                  | Loss / objective                                                | Typical data fit                                       |
|-------------------|--------------------------------------------------|------------------------------------------------------------------|--------------------------------------------------------|
| Linear regression | \(f_w(x) = w^\top x + b\)                        | \(\min_w \frac{1}{N}\sum_i \frac{1}{2}(y^{(i)} - f_w(x^{(i)}))^2\)[web:28] | Numeric targets with linear-ish relationships[web:28] |
| Logistic reg.     | \(p(y=1|x) = \sigma(w^\top x)\)                  | \(\min_w -\frac{1}{N}\sum_i y^{(i)}\log p_i + (1-y^{(i)})\log(1-p_i)\)[web:28] | Binary classification with linear decision boundary[web:28] |
| Neural net (MLP)  | \(f_\theta(x) = \sigma^L(W^L \dots \sigma^1(W^1 x + b^1)\dots)\) | Same losses as above but on deep \(f_\theta\); trained via backprop[web:24][web:31] | Structured and unstructured data with complex patterns[web:30] |
| Softmax classifier| \(p(y=k|x) = \mathrm{softmax}_k(Wx)\)            | Multiclass cross-entropy \(-\frac{1}{N}\sum_i \log p(y^{(i)}|x^{(i)})\)[web:28][web:30] | Multiclass classification problems[web:30]            |

---

## 9. Optimization Landscape and Deep Learning Intuition

Deep networks create high-dimensional nonconvex optimization problems; gradient-based methods still work well in practice.[web:30][web:31]

- **Nonconvex objective**[web:30]  
  - Composition of nonlinear activations yields a nonconvex loss surface.  
  - Many local minima and saddle points; yet many minima give similar performance.

- **Overparameterization**[web:30]  
  - Networks often have more parameters than data points.  
  - Modern theory studies how gradient descent still finds solutions with good generalization.

- **Backprop as dynamic programming**[web:31]  
  - Forward pass stores intermediate \(z^l, a^l\).  
  - Backward pass reuses these to compute gradients layer by layer, cost \(O(\text{#params})\).[web:31]

---

## 10. Directions for Further Study

To master modern neural networks and ML, extend these foundations into more advanced architectures and theory.[web:23][web:30][web:31]

- **Convolutional neural networks (CNNs)**  
  - Convolutions, weight sharing, feature maps; mathematically linear operators with local receptive fields.[web:30]
- **Recurrent networks and sequence models**  
  - Recurrence equations, unrolling in time, vanishing/exploding gradients.[web:30]
- **Transformers and attention**  
  - Scaled dot-product attention, self-attention matrices, multi-head mechanisms.[web:30]  
- **Theoretical learning foundations**  
  - Generalization bounds, VC dimension, PAC learning, optimization theory for deep networks.[web:23][web:30]

---
