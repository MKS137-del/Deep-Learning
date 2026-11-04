# Chapter 8: How Networks Learn: Backpropagation

**Level 1: Essential Foundations | Tag: Essential**

## Chapter Roadmap

Chapter 7 showed how a network turns an input into a prediction and a loss. This chapter shows how it **improves**. It combines three earlier ideas: gradients (Chapter 4), layers as matrix operations (Chapters 3 and 7), and loss functions (Chapters 5 and 7). It is the central chapter of the book: everything later is a variation on this mechanism.

By the end of this chapter you will be able to:

- Describe the learning loop: forward pass, loss, backward pass, parameter update.
- Draw a computational graph and compute gradients on it by hand.
- Explain gradient descent and its batch, stochastic, and mini-batch variants, and use the terms **epoch**, **batch**, and **iteration** correctly.
- Explain the role of the learning rate and recognise the symptoms of a bad one.
- Derive the backpropagation equations for a fully connected network, step by step, with every shape.
- Carry out a complete backward pass by hand and verify it with PyTorch autograd.
- Implement and train a small neural network from scratch in NumPy, with a numerical gradient check.
- Map each manual step to its PyTorch equivalent.

> **Honesty note.** The code in this chapter has not been executed by me. The worked numerical example in Section 8.5.6 was computed by hand, and its "Expected output" should match to the printed decimals. For the training runs in Section 8.7, the outcomes depend on random initialisation, so I describe what you should expect in qualitative terms and say so explicitly. Run each block and compare.

---

## 8.1 The Learning Problem

### 8.1.1 The loop

Training a neural network repeats four steps:

```text
   +--------------------------------------------------------------+
   |                                                              |
   v                                                              |
 1. FORWARD PASS      compute predictions from inputs (Ch 7)      |
   |                                                              |
   v                                                              |
 2. LOSS              measure how wrong they are (Ch 5, 7)        |
   |                                                              |
   v                                                              |
 3. BACKWARD PASS     compute dLoss/dParameter for every weight    |
   |                  and bias (this chapter: backpropagation)    |
   v                                                              |
 4. UPDATE            nudge every parameter against its gradient   |
   |                  (gradient descent, Ch 4)                    |
   +--------------------------------------------------------------+
```

| Step | Question answered | Tool |
|---|---|---|
| Forward pass | What does the network predict? | Matrix multiplications and activations |
| Loss | How wrong is it? | MSE, cross-entropy |
| Backward pass | Which direction should each parameter move, and how strongly? | **Backpropagation** (chain rule) |
| Update | How do I actually change the parameters? | **Gradient descent** |

### 8.1.2 Two different things people confuse

| | Backpropagation | Gradient descent |
|---|---|---|
| What it is | An algorithm for **computing** gradients efficiently | An algorithm for **using** gradients to update parameters |
| Output | The gradient $\nabla_\theta L$ | New parameter values |
| Can be swapped for | Numerical differentiation (slow, Chapter 4) | Momentum, Adam, and others (Chapter 11) |

Backpropagation does not change any parameter. It only produces the gradient. A common misconception is that "backprop is the optimiser"; the optimiser is gradient descent or one of its refinements.

### 8.1.3 Why we need an efficient method

A network may have millions or billions of parameters. Computing every partial derivative numerically needs about two extra forward passes per parameter (Chapter 4, Section 4.8), which is impossible at scale. Backpropagation computes **all** gradients at a cost of roughly a small constant multiple of a single forward pass, by reusing intermediate results. This efficiency is what made training deep networks practical. Backpropagation was popularised for neural networks by Rumelhart, Hinton, and Williams in 1986 (Chapter 1).

---

## 8.2 Computational Graphs

### 8.2.1 Definition

A **computational graph** represents a calculation as a network of simple steps. Each **node** is a value produced by one elementary operation (add, multiply, square, ReLU, and so on). Each **edge** shows which values feed into which operation.

**Why it helps.** Any function, however large, is a chain and tree of simple operations whose derivatives we already know. The chain rule (Chapter 4, Section 4.5) then gives the derivative of the whole thing by combining the **local derivatives** of each node.

### 8.2.2 Example: the one-neuron model from Chapter 4

Model: $\hat{y} = wx + b$, loss $L = (\hat{y} - y)^2$, with $x = 2$, $y = 7$, $w = 1$, $b = 0$.

```text
   w ----\
          (*)---> m ---\
   x ----/              (+)---> y_hat ---\
                  b ---/                  (-)---> e ---> (square) ---> L
                                  y ----/
```

**Forward pass** (left to right):

| Node | Operation | Value |
|---|---|---|
| $m$ | $w \cdot x$ | $2$ |
| $\hat{y}$ | $m + b$ | $2$ |
| $e$ | $\hat{y} - y$ | $-5$ |
| $L$ | $e^2$ | $25$ |

**Backward pass** (right to left). Start with $\dfrac{\partial L}{\partial L} = 1$. At every node, the gradient flowing *out* toward the inputs equals the gradient flowing *in* from the right (the **upstream gradient**) multiplied by the node's **local derivative**:

| Node | Local derivative | Upstream gradient | Gradient passed back |
|---|---|---|---|
| $L = e^2$ | $\dfrac{\partial L}{\partial e} = 2e = -10$ | $1$ | $\dfrac{\partial L}{\partial e} = -10$ |
| $e = \hat{y} - y$ | $\dfrac{\partial e}{\partial \hat{y}} = 1$ | $-10$ | $\dfrac{\partial L}{\partial \hat{y}} = -10$ |
| $\hat{y} = m + b$ | $\dfrac{\partial \hat{y}}{\partial m} = 1$, $\dfrac{\partial \hat{y}}{\partial b} = 1$ | $-10$ | $\dfrac{\partial L}{\partial m} = -10$, $\ \dfrac{\partial L}{\partial b} = -10$ |
| $m = w x$ | $\dfrac{\partial m}{\partial w} = x = 2$ | $-10$ | $\dfrac{\partial L}{\partial w} = -20$ |

These are exactly the values from Chapter 4, Section 4.6: $\partial L/\partial w = -20$ and $\partial L/\partial b = -10$.

**The universal rule of the backward pass:**

$$\text{gradient passed to an input} = \text{upstream gradient} \times \text{local derivative with respect to that input}$$

### 8.2.3 When a value is used more than once

If a node feeds into several later nodes, its gradient is the **sum** of the contributions along each path (Chapter 4, Section 4.5.5).

**Example.** $L = u \cdot v$ where $u = x^2$ and $v = x + 1$, at $x = 2$:

```text
            /---> (square) ---> u ---\
   x -------                          (*)---> L
            \---> (+1) -------> v ---/
```

- Forward: $u = 4$, $v = 3$, $L = 12$.
- Local derivatives of the product: $\partial L/\partial u = v = 3$, $\ \partial L/\partial v = u = 4$.
- Path through $u$: $3 \times \dfrac{du}{dx} = 3 \times 2x = 3 \times 4 = 12$.
- Path through $v$: $4 \times \dfrac{dv}{dx} = 4 \times 1 = 4$.
- **Total:** $\dfrac{dL}{dx} = 12 + 4 = 16$.

This sum-over-paths rule is why backpropagation must *accumulate* gradients wherever a value is reused. It is also why PyTorch's `.grad` accumulates (Chapter 4, Section 4.7.4).

### 8.2.4 Practical Python

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Computational graph by hand (one neuron, squared error) ---
x, y, w, b = 2.0, 7.0, 1.0, 0.0

m = w * x
y_hat = m + b
e = y_hat - y
L = e ** 2

dL_de = 2 * e
dL_dyhat = dL_de * 1.0
dL_dm = dL_dyhat * 1.0
dL_db = dL_dyhat * 1.0
dL_dw = dL_dm * x

print(L, dL_dw, dL_db)

# --- Two paths into the same variable, checked with autograd ---
xt = torch.tensor(2.0, requires_grad=True)
u = xt ** 2
v = xt + 1
out = u * v
out.backward()
print(xt.grad)
```

| Code | Explanation |
|---|---|
| `m = w * x` ... `L = e ** 2` | The forward pass, one node per line |
| `dL_de = 2 * e` | Local derivative of $e^2$ |
| `dL_dyhat = dL_de * 1.0` | Upstream gradient times the local derivative of the subtraction |
| `dL_dw = dL_dm * x` | Upstream gradient times $\partial m / \partial w = x$ |
| `xt.grad` | PyTorch applies the sum-over-paths rule automatically |

**Expected output:**

```text
25.0 -20.0 -10.0
tensor(16.)
```

### 8.2.5 Autograd builds the graph for you

When you use tensors with `requires_grad=True`, PyTorch records each operation as it runs (a **dynamic** graph, rebuilt on every forward pass). Calling `.backward()` walks this graph in reverse and applies the rule above at every node. Everything this chapter derives by hand is what autograd does automatically.

---

## 8.3 Gradient Descent

### 8.3.1 The update rule

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\,\nabla_{\boldsymbol{\theta}} L$$

| Symbol | Meaning |
|---|---|
| $\boldsymbol{\theta}$ | All parameters (every weight and bias) |
| $L$ | The loss |
| $\nabla_{\boldsymbol{\theta}} L$ | The gradient of the loss with respect to every parameter |
| $\eta$ | The learning rate |

This is the rule from Chapter 4, applied to every parameter at once. The minus sign matters: the gradient points uphill, so we step the opposite way.

### 8.3.2 How much data per step?

The loss is an average over the training set. We can compute the gradient from different amounts of data:

| Variant | Data used per update | Gradient quality | Speed per update | Typical use |
|---|---|---|---|---|
| **Batch** gradient descent | The **entire** training set | Exact | Slow; one update per full pass | Small datasets |
| **Stochastic** gradient descent (SGD) | **One** example | Very noisy | Fast, but cannot use parallel hardware well | Rare in pure form |
| **Mini-batch** gradient descent | A small group of examples (the **batch**), such as 32 or 128 | Good estimate with some noise | Fast and GPU-friendly | **The standard in practice** |

In deep learning, "SGD" almost always means the **mini-batch** version.

**Why a mini-batch gradient works.** If a batch is chosen uniformly at random, the expected value of its gradient equals the gradient over the whole dataset (the expectation from Chapter 5). The batch gradient is therefore an **unbiased but noisy estimate**. The noise slows nothing down in practice; it can even help the optimiser escape poor regions.

### 8.3.3 Epoch, batch, iteration

| Term | Definition |
|---|---|
| **Epoch** | One complete pass through the entire training set |
| **Batch** (mini-batch) | The group of examples used for one parameter update |
| **Batch size** $B$ | Number of examples in a batch |
| **Iteration** (step) | One parameter update: forward pass, backward pass, and update on one batch |

For a training set of $N$ examples:

$$\text{iterations per epoch} = \left\lceil \frac{N}{B} \right\rceil, \qquad \text{total updates} = \text{epochs} \times \left\lceil \frac{N}{B} \right\rceil$$

where $\lceil\cdot\rceil$ rounds up (the last batch may be smaller).

**Worked example.** $N = 1000$, $B = 32$: $1000/32 = 31.25$, so $32$ iterations per epoch (31 full batches of 32 plus one batch of 8). Training for 10 epochs gives $10 \times 32 = 320$ parameter updates.

**Standard procedure:** shuffle the training data at the start of every epoch, split it into batches, and run one iteration per batch.

### 8.3.4 Comparison: batch vs epoch vs iteration

| | Batch | Epoch | Iteration |
|---|---|---|---|
| What it counts | Examples in one update | Full passes over the data | Parameter updates |
| Example (with $N=1000$, $B=32$) | 32 examples | 1 epoch = all 1000 examples | 1 iteration = one batch of 32 |
| How they relate | | 1 epoch $=$ 32 iterations | 32 iterations $=$ 1 epoch |

### 8.3.5 Choosing the batch size

| Smaller batches | Larger batches |
|---|---|
| Noisier gradients | Smoother gradients |
| More updates per epoch | Fewer updates per epoch |
| Less memory per step | More memory per step |
| Often generalise slightly better | Better hardware utilisation (until memory runs out) |

Common values are powers of two such as 16, 32, 64, and 128, chosen by memory limits and experiment (Chapter 11).

### 8.3.6 Limits of gradient descent

- It follows the **local** slope and can stall on plateaus or saddle points (Chapter 4, Section 4.10.1).
- The loss surface of a neural network is not a simple bowl, and there is no guarantee of finding the global minimum. In practice, training reaches parameter values that work well.
- Its behaviour depends strongly on the learning rate, discussed next.

---

## 8.4 The Learning Rate

### 8.4.1 What it controls

The learning rate $\eta$ scales every step. For the quadratic in Chapter 4 the distance to the minimum shrank by the factor $1 - 2\eta$ per step (Section 4.3.3), so the same function shows every behaviour:

| $\eta$ too... | What happens | How it looks in the loss curve |
|---|---|---|
| **Small** | Steady but very slow progress | A shallow, nearly flat decline |
| **Well chosen** | Fast, smooth progress | A decline that levels off near a low value |
| **Large** | Overshoots the minimum, bounces | Jagged, noisy loss that may not settle |
| **Far too large** | Steps grow each time: divergence | Loss increases, then becomes `inf` or `nan` |

### 8.4.2 Practical guidance

- There is no universal best value. Common starting points are around $10^{-1}$ to $10^{-3}$ for plain SGD and around $10^{-3}$ to $10^{-4}$ for Adam (Chapter 11), but they must be tuned.
- If the loss is `nan` or rises quickly, lower the learning rate first.
- If the loss barely moves, try a larger one.
- Schedules that change $\eta$ during training are covered in Chapter 11.

---

## 8.5 Backpropagation Step by Step

### 8.5.1 Setup and notation

Take a fully connected network with $L$ layers, using the notation of Chapter 7 with a batch of $N$ examples:

$$\mathbf{Z}^{(\ell)} = \mathbf{A}^{(\ell-1)}\mathbf{W}^{(\ell)} + \mathbf{b}^{(\ell)}, \qquad \mathbf{A}^{(\ell)} = \varphi^{(\ell)}\!\big(\mathbf{Z}^{(\ell)}\big), \qquad \mathbf{A}^{(0)} = \mathbf{X}$$

and a scalar loss $L$ computed from $\mathbf{A}^{(L)}$ and the targets (averaged over the batch).

Define the **error signal** (sometimes called the **delta**) of layer $\ell$:

$$\boldsymbol{\delta}^{(\ell)} = \frac{\partial L}{\partial \mathbf{Z}^{(\ell)}} \qquad \text{shape } (N, n_\ell)$$

It records how sensitive the loss is to each pre-activation in the layer. Everything else follows from it.

### 8.5.2 Step 1: the error signal at the output layer

This depends on the output activation and loss. Two very common cases give the same simple form.

**Sigmoid output with binary cross-entropy.** With $a = \sigma(z)$ and $L = -[y\ln a + (1-y)\ln(1-a)]$ for one example:

$$\frac{\partial L}{\partial a} = -\frac{y}{a} + \frac{1-y}{1-a}, \qquad \frac{\partial a}{\partial z} = a(1-a)$$

$$\frac{\partial L}{\partial z} = \left(-\frac{y}{a} + \frac{1-y}{1-a}\right)a(1-a) = -y(1-a) + (1-y)a = a - y$$

**Softmax output with cross-entropy.** Using the form $L = -z_c + \ln\sum_j e^{z_j}$ for true class $c$ (Chapter 7, Section 7.4.4):

$$\frac{\partial L}{\partial z_k} = -\mathbb{1}[k = c] + \frac{e^{z_k}}{\sum_j e^{z_j}} = p_k - y_k$$

where $p_k$ is the softmax probability and $y_k$ is 1 for the true class and 0 otherwise (the one-hot label).

**Identity output with mean squared error** (one output): $L = \frac{1}{N}\sum_i(\hat{y}_i - y_i)^2$ gives $\dfrac{\partial L}{\partial \hat{y}_i} = \dfrac{2(\hat{y}_i - y_i)}{N}$.

**The punchline.** For the standard pairings, the output error signal is simply **prediction minus target**, divided by $N$ because the loss is averaged over the batch:

$$\boldsymbol{\delta}^{(L)} = \frac{1}{N}\big(\mathbf{A}^{(L)} - \mathbf{Y}\big) \quad \text{(sigmoid + BCE, or softmax + CE)}$$

This tidy result is part of why these pairings are used (Chapter 7, Section 7.4.5).

### 8.5.3 Step 2: gradients of the parameters of layer $\ell$

Given $\boldsymbol{\delta}^{(\ell)}$, the parameter gradients are:

$$\frac{\partial L}{\partial \mathbf{W}^{(\ell)}} = \big(\mathbf{A}^{(\ell-1)}\big)^{\top}\boldsymbol{\delta}^{(\ell)}, \qquad \frac{\partial L}{\partial \mathbf{b}^{(\ell)}} = \sum_{\text{rows}}\boldsymbol{\delta}^{(\ell)}$$

**Derivation.** For one entry, $Z_{ij} = \sum_k A_{ik}W_{kj} + b_j$ (example $i$, neuron $j$, input $k$), so $\dfrac{\partial Z_{ij}}{\partial W_{kj}} = A_{ik}$ and $\dfrac{\partial Z_{ij}}{\partial b_j} = 1$. Weight $W_{kj}$ affects the loss through $Z_{ij}$ for **every** example $i$ in the batch, so by the sum-over-paths rule:

$$\frac{\partial L}{\partial W_{kj}} = \sum_{i=1}^{N}\delta_{ij}\,A_{ik} = \big(\mathbf{A}^{\top}\boldsymbol{\delta}\big)_{kj}, \qquad \frac{\partial L}{\partial b_j} = \sum_{i=1}^{N}\delta_{ij}$$

| Quantity | Shape | Check |
|---|---|---|
| $\mathbf{A}^{(\ell-1)}$ | $(N, n_{\ell-1})$ | |
| $\big(\mathbf{A}^{(\ell-1)}\big)^{\top}$ | $(n_{\ell-1}, N)$ | |
| $\boldsymbol{\delta}^{(\ell)}$ | $(N, n_\ell)$ | |
| $\dfrac{\partial L}{\partial \mathbf{W}^{(\ell)}}$ | $(n_{\ell-1}, n_\ell)$ | Same shape as $\mathbf{W}^{(\ell)}$ |
| $\dfrac{\partial L}{\partial \mathbf{b}^{(\ell)}}$ | $(n_\ell,)$ | Same shape as $\mathbf{b}^{(\ell)}$ |

The shapes confirm the rule from Chapter 4: a gradient has the same shape as its parameter. The sum over the batch dimension is what turns per-example information into one gradient per parameter.

### 8.5.4 Step 3: send the error signal back to the previous layer

The previous layer's activations $\mathbf{A}^{(\ell-1)}$ influence the loss through $\mathbf{Z}^{(\ell)}$. Since $\dfrac{\partial Z_{ij}}{\partial A_{ik}} = W_{kj}$:

$$\frac{\partial L}{\partial A^{(\ell-1)}_{ik}} = \sum_{j}\delta^{(\ell)}_{ij}\,W^{(\ell)}_{kj}, \qquad\text{i.e.}\qquad \frac{\partial L}{\partial \mathbf{A}^{(\ell-1)}} = \boldsymbol{\delta}^{(\ell)}\big(\mathbf{W}^{(\ell)}\big)^{\top}$$

Shape: $(N, n_\ell)\cdot(n_\ell, n_{\ell-1}) = (N, n_{\ell-1})$, matching $\mathbf{A}^{(\ell-1)}$.

Then pass back through the activation function of layer $\ell-1$ with an **element-wise** product (the local derivative of an element-wise function is applied element by element):

$$\boldsymbol{\delta}^{(\ell-1)} = \Big(\boldsymbol{\delta}^{(\ell)}\big(\mathbf{W}^{(\ell)}\big)^{\top}\Big) \odot \varphi'\!\big(\mathbf{Z}^{(\ell-1)}\big)$$

| Symbol | Meaning |
|---|---|
| $\odot$ | Element-wise product (Chapter 3) |
| $\varphi'(\mathbf{Z}^{(\ell-1)})$ | The activation's derivative evaluated at the **stored** pre-activations from the forward pass |

This is why the forward pass must cache $\mathbf{Z}$ and $\mathbf{A}$ for every layer.

| Activation | $\varphi'(z)$ to use (from Chapter 7) |
|---|---|
| Sigmoid | $a(1 - a)$ where $a = \sigma(z)$ |
| Tanh | $1 - a^2$ where $a = \tanh(z)$ |
| ReLU | $1$ if $z > 0$, else $0$ |
| Leaky ReLU | $1$ if $z > 0$, else $\alpha$ |

### 8.5.5 The complete algorithm

1. **Forward pass.** For $\ell = 1, \dots, L$ compute and **store** $\mathbf{Z}^{(\ell)}$ and $\mathbf{A}^{(\ell)}$. Compute the loss.
2. **Output error.** $\boldsymbol{\delta}^{(L)} = \partial L / \partial \mathbf{Z}^{(L)}$ (for the standard pairings, $(\mathbf{A}^{(L)} - \mathbf{Y})/N$).
3. **For $\ell = L, L-1, \dots, 1$:**
   - $\dfrac{\partial L}{\partial \mathbf{W}^{(\ell)}} = (\mathbf{A}^{(\ell-1)})^{\top}\boldsymbol{\delta}^{(\ell)}$
   - $\dfrac{\partial L}{\partial \mathbf{b}^{(\ell)}} = \sum_{\text{rows}}\boldsymbol{\delta}^{(\ell)}$
   - if $\ell > 1$: $\boldsymbol{\delta}^{(\ell-1)} = \big(\boldsymbol{\delta}^{(\ell)}(\mathbf{W}^{(\ell)})^{\top}\big) \odot \varphi'(\mathbf{Z}^{(\ell-1)})$
4. **Update.** $\mathbf{W}^{(\ell)} \leftarrow \mathbf{W}^{(\ell)} - \eta\,\partial L/\partial \mathbf{W}^{(\ell)}$ and likewise for $\mathbf{b}^{(\ell)}$.

**Why it is efficient.** Each $\boldsymbol{\delta}^{(\ell)}$ is computed once and reused for every parameter in layer $\ell$ and for the layer before it. The backward pass costs about the same as the forward pass (two matrix multiplications per layer instead of one).

**A remark on vanishing gradients.** Step 3 multiplies by $\varphi'$ at every layer. If those factors are small (for example sigmoid's maximum of $0.25$), the error signal shrinks layer by layer on the way back. This is the vanishing-gradient problem previewed in Chapter 4, and it is why ReLU became the default hidden activation (Chapter 7) and why Chapter 10 exists.

### 8.5.6 A complete numerical example

Use the $2 \to 3 \to 1$ network from Chapter 7, Section 7.2.2, with a sigmoid output and binary cross-entropy loss, and true label $y = 1$ for the input $\mathbf{x} = [1, 2]$ ($N = 1$).

**Forward pass (from Chapter 7):**

- $\mathbf{z}^{(1)} = [2.5,\ 0.5,\ -2.0]$, $\ \mathbf{a}^{(1)} = [2.5,\ 0.5,\ 0]$
- $z^{(2)} = 1.75$, $\ a^{(2)} = \sigma(1.75) \approx 0.85195$
- Loss: $L = -\ln(0.85195) \approx 0.1602$

**Backward pass:**

1. **Output error:** $\delta^{(2)} = a^{(2)} - y = 0.85195 - 1 = -0.14805$.

2. **Layer 2 parameter gradients:**
   - $\dfrac{\partial L}{\partial \mathbf{W}^{(2)}} = (\mathbf{a}^{(1)})^{\top}\delta^{(2)} = \begin{bmatrix} 2.5 \\ 0.5 \\ 0 \end{bmatrix}(-0.14805) = \begin{bmatrix} -0.3701 \\ -0.0740 \\ 0 \end{bmatrix}$
   - $\dfrac{\partial L}{\partial b^{(2)}} = -0.14805$

3. **Send back to the hidden layer.** $\mathbf{W}^{(2)} = [1.0,\ -2.0,\ 0.5]^{\top}$, so:
   $$\delta^{(2)}\big(\mathbf{W}^{(2)}\big)^{\top} = -0.14805\,[1.0,\ -2.0,\ 0.5] = [-0.14805,\ 0.29609,\ -0.07402]$$

4. **Through the ReLU.** $\text{ReLU}'(\mathbf{z}^{(1)}) = [1, 1, 0]$ (the third pre-activation was negative):
   $$\boldsymbol{\delta}^{(1)} = [-0.14805,\ 0.29609,\ -0.07402] \odot [1, 1, 0] = [-0.14805,\ 0.29609,\ 0]$$
   The third hidden neuron was inactive, so no error flows through it and its weights receive no update.

5. **Layer 1 parameter gradients.** With $\mathbf{x} = [1, 2]$:
   $$\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \mathbf{x}^{\top}\boldsymbol{\delta}^{(1)} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}[-0.14805,\ 0.29609,\ 0] = \begin{bmatrix} -0.1480 & 0.2961 & 0 \\ -0.2961 & 0.5922 & 0 \end{bmatrix}$$
   $$\frac{\partial L}{\partial \mathbf{b}^{(1)}} = \boldsymbol{\delta}^{(1)} = [-0.14805,\ 0.29609,\ 0]$$

**Update with $\eta = 0.1$** and recompute the forward pass: the new loss is approximately $0.0949$, down from $0.1602$. (It does not drop by exactly $\eta\lVert\nabla L\rVert^2$, because the loss surface curves; the first-order estimate from Chapter 4, Section 4.4.5 is only exact for infinitesimal steps.)

Now the same computation in code, with PyTorch autograd as an independent check:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x = np.array([[1.0, 2.0]])                         # (1, 2)
y = np.array([[1.0]])                              # (1, 1)

W1 = np.array([[0.5, -1.0,  0.0],
               [1.0,  0.5, -0.5]])                 # (2, 3)
b1 = np.array([0.0, 0.5, -1.0])                    # (3,)
W2 = np.array([[1.0], [-2.0], [0.5]])              # (3, 1)
b2 = np.array([0.25])                              # (1,)

def forward_loss(W1, b1, W2, b2):
    A1 = np.maximum(0, x @ W1 + b1)
    A2 = sigmoid(A1 @ W2 + b2)
    return float(-np.mean(y * np.log(A2) + (1 - y) * np.log(1 - A2)))

# ---- Forward pass (store everything the backward pass needs) ----
Z1 = x @ W1 + b1
A1 = np.maximum(0, Z1)
Z2 = A1 @ W2 + b2
A2 = sigmoid(Z2)
loss = forward_loss(W1, b1, W2, b2)

# ---- Backward pass ----
N = x.shape[0]
dZ2 = (A2 - y) / N                                 # (1, 1)
dW2 = A1.T @ dZ2                                   # (3, 1)
db2 = dZ2.sum(axis=0)                              # (1,)
dA1 = dZ2 @ W2.T                                   # (1, 3)
dZ1 = dA1 * (Z1 > 0)                               # ReLU derivative as a 0/1 mask
dW1 = x.T @ dZ1                                    # (2, 3)
db1 = dZ1.sum(axis=0)                              # (3,)

print(f"loss = {loss:.4f}")
print(np.round(dW2[:2, 0], 4).tolist())
print(np.round(db2, 4).tolist())
print(np.round(dW1[:, :2], 4).tolist())

# ---- Independent check with PyTorch autograd ----
tW1 = torch.tensor(W1, requires_grad=True)
tb1 = torch.tensor(b1, requires_grad=True)
tW2 = torch.tensor(W2, requires_grad=True)
tb2 = torch.tensor(b2, requires_grad=True)
tx, ty = torch.tensor(x), torch.tensor(y)

tloss = F.binary_cross_entropy_with_logits(torch.relu(tx @ tW1 + tb1) @ tW2 + tb2, ty)
tloss.backward()

ok = (np.allclose(dW1, tW1.grad.numpy()) and np.allclose(db1, tb1.grad.numpy())
      and np.allclose(dW2, tW2.grad.numpy()) and np.allclose(db2, tb2.grad.numpy()))
print("autograd agrees:", ok)

# ---- One gradient descent step ----
lr = 0.1
new_loss = forward_loss(W1 - lr * dW1, b1 - lr * db1, W2 - lr * dW2, b2 - lr * db2)
print(f"loss after one step = {new_loss:.4f}")
```

| Code | Explanation |
|---|---|
| `forward_loss(...)` | A helper that runs the forward pass for any parameter values, so we can test updated parameters |
| `dZ2 = (A2 - y) / N` | The output error for sigmoid + BCE (Section 8.5.2), shape `(1, 1)` |
| `A1.T @ dZ2` | $(\mathbf{A}^{(1)})^{\top}\boldsymbol{\delta}^{(2)}$, shape `(3, 1)`, the same as `W2` |
| `dZ2.sum(axis=0)` | Sum of the error over the batch for the bias gradient |
| `dZ2 @ W2.T` | Sends the error back through the weights: `(1,1) @ (1,3)` gives `(1,3)` |
| `dA1 * (Z1 > 0)` | Element-wise product with the ReLU derivative. `(Z1 > 0)` is a boolean array that acts as 1.0 or 0.0 |
| `x.T @ dZ1` | $\mathbf{x}^{\top}\boldsymbol{\delta}^{(1)}$, shape `(2, 3)`, the same as `W1` |
| `np.round(dW2[:2, 0], 4)` | Shows only the first two entries: the third gradient is exactly zero (it can print as `-0.0`) |
| `torch.tensor(W1, requires_grad=True)` | Float64 tensors (NumPy's default), so the comparison is precise |
| `F.binary_cross_entropy_with_logits(logits, y)` | Sigmoid and BCE in one stable call, taking the raw logit (Chapter 7) |
| `np.allclose` | Compares manual and autograd gradients within tolerance |

**Expected output (hand-computed):**

```text
loss = 0.1602
[-0.3701, -0.074]
[-0.148]
[[-0.148, 0.2961], [-0.2961, 0.5922]]
autograd agrees: True
loss after one step = 0.0949
```

---

## 8.6 Implementing a Network from Scratch in NumPy

### 8.6.1 Design

We build a small two-layer network exactly as derived above, with no deep-learning library:

| Choice | Value |
|---|---|
| Architecture | Input $\to$ hidden layer (tanh) $\to$ one output (sigmoid) |
| Loss | Binary cross-entropy |
| Hidden derivative | $1 - a^2$, where $a = \tanh(z)$ |
| Output error | $(\mathbf{A}^{(2)} - \mathbf{Y})/N$ (Section 8.5.2) |
| Conventions | Row-per-sample batches; weights stored as $(n_{\text{in}}, n_{\text{out}})$ (Chapter 3 convention, **not** PyTorch's) |

Initial weights are drawn from a normal distribution with standard deviation $0.5$ and biases start at zero. (Why this matters, and better schemes, are the subject of Chapter 10.)

```python
class MLP:
    def __init__(self, n_in, n_hidden, n_out=1, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 0.5, size=(n_in, n_hidden))     # (n_in, n_hidden)
        self.b1 = np.zeros(n_hidden)                              # (n_hidden,)
        self.W2 = rng.normal(0.0, 0.5, size=(n_hidden, n_out))    # (n_hidden, n_out)
        self.b2 = np.zeros(n_out)                                 # (n_out,)

    def forward(self, X):
        self.X = X
        self.Z1 = X @ self.W1 + self.b1          # (N, n_hidden)
        self.A1 = np.tanh(self.Z1)               # (N, n_hidden)
        self.Z2 = self.A1 @ self.W2 + self.b2    # (N, n_out)
        self.A2 = sigmoid(self.Z2)               # (N, n_out)
        return self.A2

    def loss(self, Y):
        p = np.clip(self.A2, 1e-12, 1 - 1e-12)   # avoid log(0)
        return float(-np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p)))

    def backward(self, Y):
        N = self.X.shape[0]
        dZ2 = (self.A2 - Y) / N                  # (N, n_out)
        self.dW2 = self.A1.T @ dZ2               # (n_hidden, n_out)
        self.db2 = dZ2.sum(axis=0)               # (n_out,)
        dA1 = dZ2 @ self.W2.T                    # (N, n_hidden)
        dZ1 = dA1 * (1 - self.A1 ** 2)           # tanh derivative
        self.dW1 = self.X.T @ dZ1                # (n_in, n_hidden)
        self.db1 = dZ1.sum(axis=0)               # (n_hidden,)

    def step(self, lr):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2
```

| Code | Explanation |
|---|---|
| `np.random.default_rng(seed)` | A seeded random generator (Chapter 2): the same seed gives the same initial weights |
| `rng.normal(0.0, 0.5, size=...)` | Weights from a normal distribution with mean 0 and standard deviation 0.5 (Chapter 5) |
| `forward` stores `self.X`, `Z1`, `A1`, `Z2`, `A2` | The **cache** that the backward pass needs (Section 8.5.4) |
| `np.clip(..., 1e-12, 1 - 1e-12)` | Keeps probabilities away from exactly 0 or 1 so `log` never returns `-inf` |
| `backward` | Implements the algorithm of Section 8.5.5 line by line. Each comment gives the shape |
| `dA1 * (1 - self.A1 ** 2)` | Element-wise product with $\tanh'(z) = 1 - a^2$ |
| `-=` in `step` | In-place update $\theta \leftarrow \theta - \eta\,\nabla L$ |

`Y` must have shape `(N, 1)`, matching the output (a very common source of silent broadcasting bugs, Chapter 2).

### 8.6.2 Gradient checking

Never trust hand-written backward code until it has been compared with numerical differentiation (Chapter 4, Section 4.8). The check nudges each parameter by $\pm h$ and compares the finite-difference slope with the backward pass.

```python
def numerical_gradient(net, X, Y, param, h=1e-5):
    grad = np.zeros_like(param)
    for idx in np.ndindex(param.shape):          # visit every element of the parameter
        old = param[idx]
        param[idx] = old + h
        net.forward(X)
        loss_plus = net.loss(Y)
        param[idx] = old - h
        net.forward(X)
        loss_minus = net.loss(Y)
        param[idx] = old                         # restore the original value
        grad[idx] = (loss_plus - loss_minus) / (2 * h)
    return grad

def relative_error(a, b):
    return np.max(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-12))

rng = np.random.default_rng(0)
Xc = rng.normal(size=(5, 2))                                 # 5 random samples
Yc = (rng.random((5, 1)) > 0.5).astype(float)                # random 0/1 labels, shape (5, 1)

net = MLP(n_in=2, n_hidden=4, seed=1)
net.forward(Xc)
net.backward(Yc)

for name in ["W1", "b1", "W2", "b2"]:
    analytic = getattr(net, "d" + name)
    numeric = numerical_gradient(net, Xc, Yc, getattr(net, name))
    print(name, f"{relative_error(analytic, numeric):.1e}")
```

| Code | Explanation |
|---|---|
| `np.ndindex(param.shape)` | Generates every index tuple of an array of that shape |
| `param[idx] = old + h` | Modifies the parameter **in place**, which also changes `net`'s own array because they are the same object |
| `net.forward(X)` then `net.loss(Y)` | Re-evaluates the loss at the nudged parameter |
| `param[idx] = old` | Restores the value, so later checks are unaffected |
| `relative_error` | A scale-free measure of disagreement; tiny means the two gradients match |
| `getattr(net, "d" + name)` | Fetches `net.dW1`, `net.db1`, and so on |

**Expected output:** four lines, one per parameter. Each relative error should be **very small** (typically somewhere between about $10^{-10}$ and $10^{-6}$), for example in the form `W1 3.2e-09`. The exact digits depend on the random numbers and are not predictable by hand. A value near $10^{-2}$ or larger would indicate a bug.

### 8.6.3 The training function

```python
def train(net, X, Y, lr, epochs, batch_size, seed=0, log_every=1000):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    for epoch in range(1, epochs + 1):
        order = rng.permutation(N)                       # new random order each epoch
        for start in range(0, N, batch_size):
            idx = order[start:start + batch_size]        # indices of this mini-batch
            net.forward(X[idx])
            net.backward(Y[idx])
            net.step(lr)
        if epoch == 1 or epoch % log_every == 0:
            net.forward(X)                               # loss on the full training set
            print(f"epoch {epoch:5d}  loss {net.loss(Y):.4f}")
```

| Code | Explanation |
|---|---|
| `rng.permutation(N)` | A random shuffle of the indices $0,\dots,N-1$: **shuffling each epoch** (Section 8.3.3) |
| `range(0, N, batch_size)` | Start positions of each batch. The last batch may be smaller |
| `order[start:start + batch_size]` | The indices of one mini-batch |
| `X[idx]`, `Y[idx]` | Fancy indexing selects those rows (Chapter 2) |
| `forward`, `backward`, `step` | One **iteration**: the three steps of the learning loop |
| Logging block | Evaluates the loss on all of `X` so the numbers are comparable between epochs |

---

## 8.7 Training the Network

### 8.7.1 Learning XOR (full-batch)

XOR is the problem a single neuron cannot solve (Chapter 6). Our network has a hidden layer, so it should.

```python
X_xor = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])   # (4, 2)
Y_xor = np.array([[0.0], [1.0], [1.0], [0.0]])                        # (4, 1)

net = MLP(n_in=2, n_hidden=8, seed=1)
train(net, X_xor, Y_xor, lr=0.5, epochs=5000, batch_size=4)

preds = (net.forward(X_xor) >= 0.5).astype(int).ravel()
print(preds.tolist())
```

With `batch_size=4` equal to the dataset size, each epoch is a single iteration, so this is **batch** gradient descent, and the 5000 epochs are 5000 parameter updates.

**Expected behaviour (qualitative; not executed, results depend on the random initialisation):**

- The logged loss starts near $\ln 2 \approx 0.69$ (the network begins with near-uniform predictions, Chapter 7, Section 7.4.4), then falls steadily across the logged epochs to a small value, typically well under $0.1$ by epoch 5000.
- The final predictions should be `[0, 1, 1, 0]`: XOR solved.
- If an unlucky initialisation stalls, change the `seed` or increase `n_hidden`. Stalling on a tiny problem like this is a small taste of the optimisation difficulties of Chapter 10.

The same data defeated the single neuron in Chapter 6, regardless of training time. The difference is the hidden layer and the nonlinearity.

### 8.7.2 Learning a circular boundary (mini-batch)

Chapter 1 described points inside and outside a circle, a problem no single line can solve. Here we generate 400 such points and train with mini-batches.

```python
rng = np.random.default_rng(0)
X_circ = rng.uniform(-2.0, 2.0, size=(400, 2))                              # (400, 2)
Y_circ = ((X_circ ** 2).sum(axis=1) > 2.0).astype(float).reshape(-1, 1)      # (400, 1)

print("fraction labelled 1:", round(float(Y_circ.mean()), 2))

net = MLP(n_in=2, n_hidden=16, seed=0)
train(net, X_circ, Y_circ, lr=0.5, epochs=200, batch_size=32, log_every=50)

accuracy = float(((net.forward(X_circ) >= 0.5) == (Y_circ >= 0.5)).mean())
print(f"training accuracy: {accuracy:.2f}")
```

| Code | Explanation |
|---|---|
| `rng.uniform(-2.0, 2.0, size=(400, 2))` | 400 points, each coordinate uniform in $[-2, 2]$ |
| `(X_circ ** 2).sum(axis=1) > 2.0` | $x_1^2 + x_2^2 > 2$: outside a circle of radius $\sqrt{2}$ (Chapter 1's feature $r^2$) |
| `.reshape(-1, 1)` | Makes the labels shape `(400, 1)`, as the network requires |
| `batch_size=32` | **Mini-batch** gradient descent |

**Counting updates.** $N = 400$ and $B = 32$ give $\lceil 400/32 \rceil = 13$ iterations per epoch (12 full batches of 32 plus one of 16). Over 200 epochs that is $200 \times 13 = 2600$ parameter updates.

**What to expect.**

- The area of the circle is $2\pi \approx 6.28$ out of the square's area $16$, so about $61\%$ of points lie outside: the printed fraction should be close to `0.61`. A model that always predicts "outside" would therefore score about 0.61, which is the baseline to beat.
- The loss should decline across the logged epochs, and the final training accuracy should be well above that baseline (typically above 0.9). I have not run it, so treat these as expectations rather than measured results. Exact numbers depend on the data and initialisation.
- A single sigmoid neuron cannot do this (Chapter 6): its boundary is a straight line. The hidden layer is what produces a curved boundary, which is the learned counterpart of the hand-designed feature $r^2$ from Chapter 1.

**Training accuracy is not the goal.** This accuracy is measured on the training data. How well the model does on unseen data is a separate question (Chapter 12).

---

## 8.8 The Same Thing in PyTorch

Autograd and the built-in optimiser replace the hand-written `backward` and `step`:

```python
torch.manual_seed(0)

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.Tanh(),
    nn.Linear(8, 1),          # no sigmoid: BCEWithLogitsLoss expects logits
)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

X_t = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])    # (4, 2)
Y_t = torch.tensor([[0.], [1.], [1.], [0.]])                     # (4, 1), floats

for epoch in range(1, 5001):
    optimizer.zero_grad()              # 1. clear old gradients
    logits = model(X_t)                # 2. forward pass
    loss = loss_fn(logits, Y_t)        # 3. loss
    loss.backward()                    # 4. backward pass (autograd)
    optimizer.step()                   # 5. parameter update
    if epoch == 1 or epoch % 1000 == 0:
        print(f"epoch {epoch:5d}  loss {loss.item():.4f}")

preds = (torch.sigmoid(model(X_t)) >= 0.5).long().flatten()
print(preds)
```

| Code | Explanation |
|---|---|
| `torch.optim.SGD(model.parameters(), lr=0.5)` | An optimiser object holding the parameters and learning rate. Plain gradient descent here. Chapter 11 introduces Adam and others |
| `optimizer.zero_grad()` | Resets every `.grad` to zero. **Forgetting this makes gradients accumulate** (Chapter 4, Section 4.7.4) |
| `loss_fn(logits, Y_t)` | Sigmoid and BCE together, taking raw logits |
| `loss.backward()` | Runs backpropagation through the recorded graph |
| `optimizer.step()` | Applies $\theta \leftarrow \theta - \eta\nabla L$ to every parameter |

The loss printed here is the value from *before* that epoch's update, whereas the NumPy `train` function printed it *after*. The numbers are therefore offset by one update, and, because initial values differ between NumPy and PyTorch, they will not match each other.

**Expected behaviour (not executed):** the same qualitative pattern as in Section 8.7.1: the loss starts near $0.69$ and falls, and the final predictions should be `tensor([0, 1, 1, 0])`.

### 8.8.1 Mapping manual steps to PyTorch

| Manual NumPy | PyTorch |
|---|---|
| `net.forward(X)` | `model(X)` |
| `net.loss(Y)` | `loss_fn(logits, Y)` |
| `net.backward(Y)` with hand-derived formulas | `loss.backward()` (autograd) |
| Gradient arrays `dW1`, `db1`, ... | `p.grad` for each parameter `p` |
| `net.step(lr)` | `optimizer.step()` |
| Overwriting gradients each call | `optimizer.zero_grad()` (needed because PyTorch accumulates) |
| Writing `dZ1 = dA1 * (1 - A1**2)` for each activation | Handled automatically for every built-in operation |
| Weight layout `(n_in, n_out)` | `nn.Linear` stores `(n_out, n_in)` (Chapter 3) |

### 8.8.2 Manual backpropagation vs autograd

| | Hand-written backpropagation | Autograd |
|---|---|---|
| Effort | Derive and code every layer's backward step | None: write only the forward pass |
| Error risk | High (shape and sign bugs) | Low |
| Flexibility | Limited to what you derive | Any differentiable computation |
| Speed | Same algorithm | Same algorithm, plus small bookkeeping overhead |
| Value | Builds understanding; needed for custom ops | Used for all real work |

Understanding the manual version is what lets you read gradient problems, write custom layers, and debug when training misbehaves. Autograd is not magic: it is Section 8.5 applied automatically.

---

## 8.9 Diagnosing Learning Problems

| Symptom | Likely causes | What to try |
|---|---|---|
| Loss becomes `nan` or `inf` | Learning rate too high; `log(0)`; division by zero; exploding gradients | Lower the learning rate; clip probabilities; check the data for `nan` |
| Loss does not decrease at all | Learning rate too small; forgot `optimizer.step()`; parameters not passed to the optimiser; `requires_grad` off; gradients zero | Print gradient norms; raise the learning rate; check the loop |
| Loss decreases very slowly | Learning rate too small; saturated activations; poor initialisation or scaling | Larger learning rate; standardise inputs (Chapter 5); better initialisation (Chapter 10) |
| Loss is jagged and does not settle | Learning rate too large; very small batches | Lower the learning rate or increase the batch size |
| Initial loss far from $\ln K$ | Wrong labels, wrong output layer, or an activation applied twice | Check shapes, targets, and the loss pairing (Chapter 7) |
| Training loss low but validation loss high | Overfitting | Regularisation (Chapter 10), more data (Chapter 12) |
| Gradients are exactly zero for a layer | Dead ReLUs; a detached tensor; a broken graph | Inspect activations; check for `.detach()` or `torch.no_grad()` in the forward path |

### 8.9.1 Three habits that find most bugs

1. **Overfit a tiny batch first.** Train on 8 to 32 examples for many iterations. A correct model with enough capacity should drive the loss close to zero. If it cannot, there is a bug in the model, loss, or loop, not a data problem.
2. **Check the initial loss** against $\ln K$ (or the expected value for your task).
3. **Inspect gradient sizes** per layer after one backward pass:

```python
loss.backward()
for name, p in model.named_parameters():
    print(name, round(p.grad.norm().item(), 4))
```

| Code | Explanation |
|---|---|
| `model.named_parameters()` | Yields `(name, parameter)` pairs, such as `0.weight` and `2.bias` |
| `p.grad.norm()` | The L2 norm of the gradient (Chapter 3): a single number summarising its size |

Layers whose gradient norms are tiny (vanishing) or huge (exploding) compared with others point to the problems studied in Chapter 10. If you write custom backward code, run the gradient check from Section 8.6.2.

---

## 8.10 Common Misconceptions and Mistakes

| Misconception or mistake | Correction |
|---|---|
| "Backpropagation is the learning algorithm." | It only computes gradients. Learning is gradient descent (or a refinement) using those gradients |
| "Backpropagation only works for neural networks." | It is the chain rule on a computational graph and applies to any differentiable computation |
| "Backpropagation goes backwards in time or through the data." | It goes backwards through the **computation**, from the loss toward the inputs |
| "More epochs always help." | Beyond a point training loss keeps falling while validation performance gets worse (Chapter 12) |
| "An epoch is one weight update." | An epoch is a full pass over the data; an update is an iteration. One epoch contains $\lceil N/B\rceil$ iterations |
| "The batch gradient is the true gradient." | A mini-batch gradient is a noisy estimate of it |
| "A lower learning rate is always safer." | Too low wastes compute and can get stuck on flat regions |
| "Gradients are computed with respect to the data." | By default, with respect to the **parameters** (gradients with respect to inputs are used for special purposes such as adversarial examples) |
| Forgetting to zero gradients | They accumulate and corrupt the update (Chapter 4, Section 4.7.4) |
| Updating parameters before all gradients are computed | In the from-scratch code, compute all of `dW1`, `db1`, `dW2`, `db2` first. The backward pass for layer 1 uses the **old** `W2` |
| Shapes `(N, 1)` vs `(N,)` for targets | Silent broadcasting gives a wrong loss; make the shapes match |

---

## 8.11 Comparison Summary

| Concept | Definition | Role |
|---|---|---|
| Computational graph | Calculation as nodes (operations) and edges (data flow) | The structure backpropagation walks |
| Local derivative | Derivative of one node with respect to its input | Building block of the chain rule |
| Upstream gradient | Gradient arriving at a node from later nodes | Multiplied by the local derivative |
| Error signal $\boldsymbol{\delta}^{(\ell)}$ | $\partial L / \partial \mathbf{Z}^{(\ell)}$ | Passed backwards layer by layer |
| Backpropagation | Efficient computation of all gradients using stored forward values | Produces $\nabla_\theta L$ |
| Gradient descent | $\theta \leftarrow \theta - \eta\nabla L$ | Uses the gradient to learn |
| Mini-batch | A small random subset used per update | Practical, GPU-friendly estimate of the gradient |
| Epoch | One full pass over the training data | Unit of training progress |
| Iteration | One parameter update | Unit of optimisation progress |
| Learning rate $\eta$ | Step size scale | Most influential hyperparameter |
| Gradient check | Compare analytic and numerical gradients | Verifies hand-written backward code |

---

## Exercises

1. **Graph by hand.** For $L = (w_1x_1 + w_2x_2 + b - y)^2$ with $x = [1, 3]$, $y = 4$, $w = [0.5, 0.5]$, $b = 0$, draw the computational graph, run the forward pass, and compute $\partial L/\partial w_1$, $\partial L/\partial w_2$, $\partial L/\partial b$ using the local-derivative rule. Verify with autograd.
2. **Sum over paths.** Let $L = (x + y)\cdot(x - y)$. Compute $\partial L/\partial x$ at $x = 3$, $y = 2$ by the two-path rule, then by first simplifying $L = x^2 - y^2$. Check with autograd.
3. **Epochs and iterations.** A dataset has 2,500 examples and the batch size is 64. How many iterations are in one epoch? How many parameter updates occur in 20 epochs? What is the size of the last batch in each epoch?
4. **Learning rate.** Run the one-dimensional descent of Chapter 4, Section 4.3.2 with $\eta = 0.05$, $0.5$, and $1.1$. Predict the behaviour before running.
5. **Output error.** For a 3-class softmax output with logits $[2.0, 1.0, 0.0]$ and true class 1 (index 0), compute $\mathbf{p} - \mathbf{y}$ by hand. Check with autograd using `F.cross_entropy`.
6. **Backward by hand.** Take the $2 \to 3 \to 1$ network of Section 8.5.6 and the input $\mathbf{x} = [0, 1]$ with label $y = 0$. Carry out the forward pass and the complete backward pass by hand, then check with the code.
7. **Shapes.** A network has layers $10 \to 20 \to 5 \to 1$ and a batch of $N = 32$. Give the shapes of $\boldsymbol{\delta}^{(\ell)}$, $\partial L/\partial\mathbf{W}^{(\ell)}$, and $\partial L/\partial\mathbf{b}^{(\ell)}$ for each layer.
8. **Vanishing.** In a 10-layer network with sigmoid activations, the local activation derivatives are at most $0.25$. Give an upper bound on the factor contributed by the activations to the gradient reaching layer 1, ignoring the weights. How does this change with ReLU?
9. **Code.** Modify the `MLP` class to use ReLU instead of tanh in the hidden layer. What changes in `forward` and `backward`? Run the gradient check again. (Hint: points where $z$ is exactly 0 can make the numerical check disagree.)
10. **Batch size experiment.** Train the circle problem with `batch_size` equal to 400, 32, and 1. Compare the loss after a fixed number of epochs and the number of updates per epoch. What do you observe about noise and speed?
11. **Bug hunt.** In the from-scratch `step`, suppose you accidentally wrote `self.W1 += lr * self.dW1`. Predict what would happen to the loss, and test it.
12. **Reflection.** In your own words, explain the difference between backpropagation and gradient descent, and why backpropagation needs the forward pass to store its intermediate values.
