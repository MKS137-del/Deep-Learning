# Chapter 7: Neural Networks: Forward Pass

**Level 1: Essential Foundations | Tag: Essential**

## Chapter Roadmap

Chapter 6 ended with a single neuron and its limits. This chapter assembles neurons into **layers** and layers into a **network**, and explains how an input travels through it to produce a prediction (the **forward pass**). We also complete two promises made earlier: a proper treatment of **activation functions** and of **loss functions**.

By the end of this chapter you will be able to:

- Describe the input, hidden, and output layers of a feedforward network, and compute every shape in it.
- Carry out a complete forward pass by hand, in NumPy, and in PyTorch.
- Explain why nonlinear activation functions are essential, and compare sigmoid, tanh, ReLU, Leaky ReLU, and softmax (including their derivatives).
- Choose an output activation and a loss function for regression, binary classification, and multi-class classification.
- Compute mean squared error, binary cross-entropy, and categorical cross-entropy by hand and in PyTorch.
- Explain the difference between shallow and deep networks, and state what the universal approximation theorem does and does not say.

**What this chapter does not do.** It explains how a network *computes* a prediction and how *wrong* that prediction is. How the weights are *improved* (backpropagation) is Chapter 8.

> **Honesty note.** The code in this chapter has not been executed by me. Every "Expected output" was derived by hand. Where a block uses random numbers, the text states what to expect only in terms of shapes or approximate behaviour. Run each block and compare.

---

## 7.1 The Architecture of a Feedforward Network

### 7.1.1 Formal definition

A **feedforward neural network** (also called a **multi-layer perceptron, MLP**, or **fully connected network**) is a function built by chaining layers. Information flows in one direction, from input to output, with no loops. Each layer consists of neurons (Chapter 6), and in a **fully connected (dense)** layer every neuron receives every output of the previous layer.

| Layer type | Role |
|---|---|
| **Input layer** | Holds the raw input features $\mathbf{x}$. It performs no computation and has no parameters |
| **Hidden layer(s)** | Intermediate layers that compute new representations. "Hidden" because they are neither input nor output |
| **Output layer** | Produces the final prediction $\hat{\mathbf{y}}$ |

```text
 Input layer      Hidden layer 1     Hidden layer 2     Output layer
  (4 features)      (5 neurons)        (3 neurons)        (2 neurons)

     x1 ----\         h1 ----\           g1 ----\           y1
     x2 -----\--->    h2 -----\--->      g2 -----\--->      y2
     x3 -----/        h3 -----/          g3 -----/
     x4 ----/         h4 ---/
                      h5 --/
   (every neuron connects to every neuron in the next layer)
```

### 7.1.2 Terminology

| Term | Meaning |
|---|---|
| **Width** of a layer | Number of neurons in it |
| **Depth** of a network | Number of layers that have parameters (hidden layers plus the output layer). The input layer is usually not counted |
| **Shallow** network | Few layers (often one hidden layer) |
| **Deep** network | Many hidden layers (this is the "deep" in Deep Learning) |
| **Parameters** | All weights and biases |
| **Hyperparameters** | Choices made before training (number of layers, widths, learning rate) |

Counting conventions vary between authors, so when you read "a three-layer network", check whether the input layer is included.

### 7.1.3 The mathematics of one layer, restated

From Chapter 3, with a batch $\mathbf{X}$ of $N$ samples stored one per row, layer $\ell$ computes:

$$\mathbf{Z}^{(\ell)} = \mathbf{A}^{(\ell-1)}\,\mathbf{W}^{(\ell)} + \mathbf{b}^{(\ell)}, \qquad \mathbf{A}^{(\ell)} = \varphi^{(\ell)}\!\big(\mathbf{Z}^{(\ell)}\big), \qquad \mathbf{A}^{(0)} = \mathbf{X}$$

| Symbol | Shape | Meaning |
|---|---|---|
| $\ell$ | | Layer index, $1, \dots, L$. The superscript in parentheses is an index, not a power |
| $\mathbf{A}^{(\ell-1)}$ | $(N, n_{\ell-1})$ | Input to layer $\ell$: the previous layer's activations |
| $\mathbf{W}^{(\ell)}$ | $(n_{\ell-1}, n_\ell)$ | Weights of layer $\ell$ ($n_\ell$ neurons, each with $n_{\ell-1}$ weights) |
| $\mathbf{b}^{(\ell)}$ | $(n_\ell,)$ | Biases, broadcast across the batch |
| $\mathbf{Z}^{(\ell)}$ | $(N, n_\ell)$ | Pre-activations |
| $\varphi^{(\ell)}$ | | Activation function, applied **element-wise** |
| $\mathbf{A}^{(\ell)}$ | $(N, n_\ell)$ | Activations: this layer's output |
| $n_\ell$ | | Number of neurons in layer $\ell$ ($n_0$ is the number of input features) |

The network's prediction is the last layer's output: $\hat{\mathbf{Y}} = \mathbf{A}^{(L)}$.

**Parameter count of layer $\ell$:** $n_{\ell-1} \cdot n_\ell + n_\ell$ (Chapter 3, Section 3.6.6).

### 7.1.4 Simple explanation and intuition

Each layer takes the previous layer's description of the data and re-describes it in terms of new, hopefully more useful, features. The first layer works from raw inputs; later layers work from features of features. This is the "hierarchical representation" idea from Chapter 1.

The activation function is what makes each re-description genuinely *new*. Without it, every layer would be a linear map and the whole network would collapse into one (Section 3.5.3).

---

## 7.2 Forward Propagation

### 7.2.1 Definition

**Forward propagation** (the **forward pass**) is the process of computing the network's output from its input by applying each layer in order, from the first to the last. The same procedure is used during training (to compute the loss) and during **inference** (to make predictions with a trained model).

**Algorithm:**

1. Set $\mathbf{A}^{(0)} = \mathbf{X}$.
2. For each layer $\ell = 1, \dots, L$:
   - compute $\mathbf{Z}^{(\ell)} = \mathbf{A}^{(\ell-1)}\mathbf{W}^{(\ell)} + \mathbf{b}^{(\ell)}$,
   - compute $\mathbf{A}^{(\ell)} = \varphi^{(\ell)}(\mathbf{Z}^{(\ell)})$.
3. Output $\hat{\mathbf{Y}} = \mathbf{A}^{(L)}$.

Viewed as one function, this is exactly the composition from Chapter 1:

$$f(\mathbf{x};\theta) = \varphi^{(L)}\Big(\cdots\varphi^{(2)}\big(\varphi^{(1)}(\mathbf{x}\mathbf{W}^{(1)} + \mathbf{b}^{(1)})\,\mathbf{W}^{(2)} + \mathbf{b}^{(2)}\big)\cdots\Big)$$

**For training, keep the intermediate values.** The values $\mathbf{Z}^{(\ell)}$ and $\mathbf{A}^{(\ell)}$ computed on the way forward are needed to compute gradients on the way back (Chapter 8). Frameworks such as PyTorch store them automatically.

### 7.2.2 Worked example: a $2 \to 3 \to 1$ network, one sample

Architecture: 2 inputs, one hidden layer of 3 neurons with **ReLU**, and one output neuron with **sigmoid**. Parameters:

$$\mathbf{W}^{(1)} = \begin{bmatrix} 0.5 & -1.0 & 0.0 \\ 1.0 & 0.5 & -0.5 \end{bmatrix},\quad \mathbf{b}^{(1)} = \begin{bmatrix} 0.0 & 0.5 & -1.0 \end{bmatrix},\quad \mathbf{W}^{(2)} = \begin{bmatrix} 1.0 \\ -2.0 \\ 0.5 \end{bmatrix},\quad \mathbf{b}^{(2)} = \begin{bmatrix} 0.25 \end{bmatrix}$$

Shapes: $\mathbf{W}^{(1)}$ is $(2, 3)$, $\mathbf{b}^{(1)}$ is $(3,)$, $\mathbf{W}^{(2)}$ is $(3, 1)$, $\mathbf{b}^{(2)}$ is $(1,)$.

Input: $\mathbf{x} = [1, 2]$.

**Hidden layer pre-activation $\mathbf{z}^{(1)} = \mathbf{x}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}$:**

- Neuron 1: $1(0.5) + 2(1.0) + 0.0 = 2.5$
- Neuron 2: $1(-1.0) + 2(0.5) + 0.5 = 0.5$
- Neuron 3: $1(0.0) + 2(-0.5) - 1.0 = -2.0$

So $\mathbf{z}^{(1)} = [2.5,\ 0.5,\ -2.0]$.

**Hidden activation (ReLU, $\max(0, z)$):** $\mathbf{a}^{(1)} = [2.5,\ 0.5,\ 0]$.

**Output pre-activation $z^{(2)} = \mathbf{a}^{(1)}\mathbf{W}^{(2)} + b^{(2)}$:**

$$2.5(1.0) + 0.5(-2.0) + 0(0.5) + 0.25 = 2.5 - 1.0 + 0 + 0.25 = 1.75$$

**Output activation (sigmoid):** $\hat{y} = \sigma(1.75) = \dfrac{1}{1 + e^{-1.75}} = \dfrac{1}{1.1738} \approx 0.8520$.

**Parameter count:** layer 1 has $2 \times 3 + 3 = 9$, layer 2 has $3 \times 1 + 1 = 4$, total $13$.

### 7.2.3 The same computation in NumPy

```python
import numpy as np

x = np.array([[1.0, 2.0]])                         # shape (1, 2): one sample

W1 = np.array([[0.5, -1.0,  0.0],
               [1.0,  0.5, -0.5]])                 # shape (2, 3)
b1 = np.array([0.0, 0.5, -1.0])                    # shape (3,)

W2 = np.array([[ 1.0],
               [-2.0],
               [ 0.5]])                            # shape (3, 1)
b2 = np.array([0.25])                              # shape (1,)

z1 = x @ W1 + b1                                   # (1,2) @ (2,3) -> (1,3)
a1 = np.maximum(0, z1)                             # ReLU, element-wise
z2 = a1 @ W2 + b2                                  # (1,3) @ (3,1) -> (1,1)
a2 = 1 / (1 + np.exp(-z2))                         # sigmoid

print(z1.tolist())
print(a1.tolist())
print(z2.tolist())
print(f"{a2[0, 0]:.4f}")
```

| Code | Explanation |
|---|---|
| `x`, shape `(1, 2)` | A batch containing one sample. Keeping the batch dimension makes the code identical for any batch size |
| `x @ W1 + b1` | The layer formula. `b1`, shape `(3,)`, is broadcast over the batch (Chapter 2) |
| `np.maximum(0, z1)` | ReLU: compares every element with 0 and keeps the larger |
| `a1 @ W2 + b2` | The output layer's weighted sum, shape `(1, 1)` |
| `1 / (1 + np.exp(-z2))` | The sigmoid formula, applied element-wise |
| `.tolist()` | Prints nested Python lists, which avoids NumPy's spacing differences |
| `a2[0, 0]` | Extracts the single number from the `(1, 1)` array |

**Expected output (matches the hand calculation):**

```text
[[2.5, 0.5, -2.0]]
[[2.5, 0.5, 0.0]]
[[1.75]]
0.8520
```

### 7.2.4 The same computation in PyTorch

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 3),      # layer 1: 2 inputs -> 3 neurons
    nn.ReLU(),            # hidden activation
    nn.Linear(3, 1),      # layer 2: 3 inputs -> 1 neuron
    nn.Sigmoid(),         # output activation
)

# Copy our hand-chosen numbers into the layers.
# nn.Linear stores weights as (out_features, in_features), so we transpose.
with torch.no_grad():
    model[0].weight.copy_(torch.tensor(W1.T))
    model[0].bias.copy_(torch.tensor(b1))
    model[2].weight.copy_(torch.tensor(W2.T))
    model[2].bias.copy_(torch.tensor(b2))

x_t = torch.tensor(x, dtype=torch.float32)         # shape (1, 2)
out = model(x_t)

print(out.detach())
print(sum(p.numel() for p in model.parameters()))
```

| Code | Explanation |
|---|---|
| `nn.Sequential(...)` | Chains modules and runs them in order. Chapter 9 introduces custom model classes; `Sequential` is the quickest way to express a plain stack |
| `nn.Linear(2, 3)` | A dense layer $\mathbf{Z} = \mathbf{A}\mathbf{W} + \mathbf{b}$ with 2 inputs and 3 neurons |
| `nn.ReLU()`, `nn.Sigmoid()` | Activation layers with no parameters |
| `model[0]`, `model[2]` | Index into the sequence. The activations occupy positions 1 and 3 |
| `weight.copy_(torch.tensor(W1.T))` | In-place copy. `W1.T` has shape `(3, 2)`, which is `nn.Linear`'s storage layout (Chapter 3, Section 3.6.5). NumPy's float64 values are converted to float32 on copy |
| `torch.no_grad()` | We are setting parameters by hand, not training, so nothing needs tracking |
| `model(x_t)` | Runs the forward pass |
| `out.detach()` | Prints the values without gradient bookkeeping |
| `p.numel()` summed | Counts all 13 parameters |

**Expected output:**

```text
tensor([[0.8520]])
13
```

### 7.2.5 A batch, and shape tracing

Nothing changes when the batch has $N$ samples; only the leading dimension grows. For a randomly initialised network, the **shapes** (not the values) are predictable:

```python
torch.manual_seed(0)

net = nn.Sequential(
    nn.Linear(4, 5), nn.ReLU(),
    nn.Linear(5, 3), nn.ReLU(),
    nn.Linear(3, 2),
)

h = torch.randn(8, 4)                      # batch of 8 samples, 4 features
print("input", tuple(h.shape))
for layer in net:
    h = layer(h)
    print(type(layer).__name__, tuple(h.shape))

print(sum(p.numel() for p in net.parameters()))
```

| Code | Explanation |
|---|---|
| `torch.randn(8, 4)` | A random input batch: values are meaningless, but the shapes are real |
| `for layer in net:` | Iterates over the modules in the `Sequential` in order |
| `type(layer).__name__` | The layer's class name, for readable output |

Parameter count: $(4 \times 5 + 5) + (5 \times 3 + 3) + (3 \times 2 + 2) = 25 + 18 + 8 = 51$.

**Expected output:**

```text
input (8, 4)
Linear (8, 5)
ReLU (8, 5)
Linear (8, 3)
ReLU (8, 3)
Linear (8, 2)
51
```

**Habit to build:** before running any model, write down the shape after every layer. Most architecture bugs are shape bugs.

### 7.2.6 Cost of the forward pass

Each dense layer costs about $N \cdot n_{\ell-1} \cdot n_\ell$ multiply-adds (Chapter 3, Section 3.6.9). The activation functions are cheap by comparison. This is why the matrix multiplications dominate the running time, and why GPUs help (Chapter 2).

---

## 7.3 Activation Functions

### 7.3.1 Why a nonlinearity is essential

Chapter 3 proved that stacking linear layers collapses into a single linear layer. Chapter 6 showed that a single neuron cannot solve XOR. Both problems have the same fix: a **nonlinear activation function** between layers.

**Numerical demonstration: XOR with a ReLU hidden layer.** Take two hidden neurons with weights and biases:

$$\mathbf{W}^{(1)} = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}, \quad \mathbf{b}^{(1)} = [0,\ -1], \quad \mathbf{W}^{(2)} = \begin{bmatrix} 1 \\ -2 \end{bmatrix}, \quad b^{(2)} = 0$$

with ReLU on the hidden layer and no activation on the output. For each input $(x_1, x_2)$, the hidden pre-activations are $[x_1 + x_2,\ x_1 + x_2 - 1]$:

| $(x_1, x_2)$ | $\mathbf{z}^{(1)}$ | After ReLU $\mathbf{a}^{(1)}$ | Output $a_1 - 2a_2$ |
|---|---|---|---|
| $(0,0)$ | $[0, -1]$ | $[0, 0]$ | $0$ |
| $(0,1)$ | $[1, 0]$ | $[1, 0]$ | $1$ |
| $(1,0)$ | $[1, 0]$ | $[1, 0]$ | $1$ |
| $(1,1)$ | $[2, 1]$ | $[2, 1]$ | $2 - 2 = 0$ |

The outputs are $0, 1, 1, 0$: **XOR is solved.** The ReLU is what makes it work: without it, the network would compute $(x_1 + x_2) - 2(x_1 + x_2 - 1) = 2 - (x_1 + x_2)$, a linear function that cannot produce XOR.

```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)   # shape (4, 2)

W1 = np.array([[1.0, 1.0],
               [1.0, 1.0]])
b1 = np.array([0.0, -1.0])
W2 = np.array([[1.0], [-2.0]])

H = np.maximum(0, X @ W1 + b1)          # hidden layer with ReLU, shape (4, 2)
out = H @ W2                            # output, shape (4, 1)

print(H.tolist())
print(out.flatten().tolist())
```

**Expected output:**

```text
[[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 1.0]]
[0.0, 1.0, 1.0, 0.0]
```

### 7.3.2 What we want from an activation function

| Property | Why it matters |
|---|---|
| **Nonlinear** | Otherwise layers collapse into one |
| **Differentiable (almost everywhere)** | Needed for gradients (Chapter 4). The step function fails this usefully (Chapter 6) |
| **Cheap to compute** | It is applied to every neuron in every layer |
| **Gradients that do not vanish or explode** | Deep networks multiply many local derivatives (Chapter 4, Section 4.5.4) |
| **Suitable output range** | For output layers: probabilities in $(0,1)$, or unbounded for regression |

### 7.3.3 Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big)$$

**Derivation of the derivative.** Write $\sigma(z) = (1 + e^{-z})^{-1}$. By the chain rule (Chapter 4):

$$\sigma'(z) = -(1 + e^{-z})^{-2}\cdot(-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}}\cdot\frac{e^{-z}}{1 + e^{-z}} = \sigma(z)\big(1 - \sigma(z)\big)$$

since $1 - \sigma(z) = \dfrac{e^{-z}}{1 + e^{-z}}$.

| Property | Value |
|---|---|
| Output range | $(0, 1)$ |
| Maximum derivative | $0.25$, at $z = 0$ |
| Behaviour for large $\lvert z\rvert$ | **Saturates**: the curve flattens and the derivative approaches 0 |
| Zero-centred? | No (outputs are always positive) |

**Saturation, numerically:** $\sigma'(0) = 0.25$, but $\sigma'(5) = 0.9933 \times 0.0067 \approx 0.0066$, a gradient nearly 40 times smaller. A saturated neuron learns very slowly.

**Use today:** mainly in **output layers** for binary or multi-label probabilities, and as a gate inside LSTMs and GRUs (Chapter 16). It is rarely the default for hidden layers in deep networks.

### 7.3.4 Hyperbolic tangent (tanh)

$$\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}, \qquad \tanh'(z) = 1 - \tanh^2(z)$$

| Property | Value |
|---|---|
| Output range | $(-1, 1)$ |
| Maximum derivative | $1$, at $z = 0$ |
| Zero-centred? | **Yes**, which usually makes optimisation easier than with sigmoid |
| Saturation | Yes, for large $\lvert z\rvert$ |
| Relation to sigmoid | $\tanh(z) = 2\sigma(2z) - 1$ (a rescaled, shifted sigmoid) |

**Use today:** inside recurrent networks (Chapter 16) and in some output layers where values in $(-1,1)$ are wanted.

### 7.3.5 ReLU (Rectified Linear Unit)

$$\text{ReLU}(z) = \max(0, z), \qquad \text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0 \end{cases}$$

(At exactly $z = 0$ the derivative is undefined; PyTorch uses $0$.)

| Property | Value |
|---|---|
| Output range | $[0, \infty)$ |
| Cost | Extremely cheap: one comparison |
| Gradient for $z > 0$ | Exactly 1, so gradients do not shrink as they pass through active neurons |
| Sparsity | Neurons with negative input output exactly 0, so many activations can be zero (sparse representations) |
| Weakness | **Dying ReLU**: a neuron whose input is negative for *every* example outputs 0 and has gradient 0, so it can never recover |

**Use today:** the **default choice for hidden layers** in most feedforward and convolutional networks.

### 7.3.6 Leaky ReLU

$$\text{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \le 0 \end{cases}, \qquad \text{derivative} = \begin{cases} 1 & z > 0 \\ \alpha & z < 0 \end{cases}$$

with a small constant $\alpha$ (PyTorch's `nn.LeakyReLU` defaults to $\alpha = 0.01$). Negative inputs keep a small, nonzero slope, so the neuron can still receive gradient and recover. This addresses dying ReLU at the cost of one extra hyperparameter.

### 7.3.7 Other activations you will meet

| Activation | Definition | Where it appears |
|---|---|---|
| **GELU** | $z\,\Phi(z)$, where $\Phi$ is the standard normal cumulative distribution function | Common in Transformer models (Chapters 17 and 18) |
| **SiLU / Swish** | $z\,\sigma(z)$ | Some modern CNNs and language models |
| **ELU, softplus** | Smooth variants of ReLU | Occasional use |

Tag: **recommended** to recognise; the details are not needed for the foundations.

### 7.3.8 Softmax

Softmax is used on the **output layer** for multi-class classification. It converts a vector of $K$ raw scores (**logits**) $\mathbf{z} = [z_1, \dots, z_K]$ into a categorical probability distribution (Chapter 5):

$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

| Symbol | Meaning |
|---|---|
| $z_k$ | The logit (raw score) for class $k$ |
| $e^{z_k}$ | Exponentiation makes every score positive and amplifies differences |
| $\sum_j e^{z_j}$ | The normaliser: sum over all classes |
| Output | $K$ numbers, each in $(0,1)$, summing to exactly 1 |

**Properties:**

- Unlike the activations above, softmax acts on the **whole vector**, not element by element.
- Adding the same constant to every logit does not change the output (it cancels in the ratio). This allows a **numerically stable** implementation: subtract the maximum logit first so the exponentials cannot overflow.
- For two classes, softmax reduces to sigmoid: $\text{softmax}([z_1, z_2])_1 = \sigma(z_1 - z_2)$.
- The name is a "soft" version of arg-max: the largest logit receives the largest probability, but the others are not forced to zero.

**Worked example.** Logits $[2, 1, 0]$: exponentials $[7.389, 2.718, 1]$, sum $11.107$, probabilities $[0.6652, 0.2447, 0.0900]$ (as in Chapter 5).

### 7.3.9 Practical Python

First the values of each activation at a few points:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def leaky_relu(z, alpha=0.01):
    return z if z > 0 else alpha * z

for zi in [-2.0, -1.0, 0.0, 1.0, 2.0]:
    print(f"z={zi:5.1f}  sigmoid={sigmoid(zi):.4f}  tanh={np.tanh(zi):.4f}  "
          f"relu={max(0.0, zi):.1f}  leaky={leaky_relu(zi):.2f}")
```

**Expected output:**

```text
z= -2.0  sigmoid=0.1192  tanh=-0.9640  relu=0.0  leaky=-0.02
z= -1.0  sigmoid=0.2689  tanh=-0.7616  relu=0.0  leaky=-0.01
z=  0.0  sigmoid=0.5000  tanh=0.0000  relu=0.0  leaky=0.00
z=  1.0  sigmoid=0.7311  tanh=0.7616  relu=1.0  leaky=1.00
z=  2.0  sigmoid=0.8808  tanh=0.9640  relu=2.0  leaky=2.00
```

Next, check the derivatives with autograd (Chapter 4). Because the four inputs are independent, summing the outputs before calling `backward()` gives each element's own derivative:

```python
z = torch.tensor([-2.0, 0.0, 2.0, 5.0], requires_grad=True)
torch.sigmoid(z).sum().backward()
print(torch.round(z.grad, decimals=4))

z2 = torch.tensor([-1.0, 1.0], requires_grad=True)
torch.relu(z2).sum().backward()
print(z2.grad)
```

| Code | Explanation |
|---|---|
| `torch.sigmoid(z).sum().backward()` | `sum()` makes a scalar; each element of `z.grad` is then $\sigma'(z_i)$ |
| `torch.round(..., decimals=4)` | Rounds for readable output |
| `torch.relu(z2)` | ReLU on `-1` gives derivative 0; on `+1` gives 1 |

**Expected output** (from $\sigma(1-\sigma)$: $0.1192 \times 0.8808 \approx 0.1050$, $0.25$, $0.1050$, $0.0066$):

```text
tensor([0.1050, 0.2500, 0.1050, 0.0066])
tensor([0., 1.])
```

Finally, softmax, including the stability trick:

```python
def softmax(z):
    z = z - np.max(z)                   # subtract the maximum for numerical stability
    e = np.exp(z)
    return e / e.sum()

print(np.round(softmax(np.array([2.0, 1.0, 0.0])), 4).tolist())
print(np.round(softmax(np.array([1000.0, 1001.0, 1002.0])), 4).tolist())
```

| Code | Explanation |
|---|---|
| `z - np.max(z)` | Shifts the largest logit to 0, so $e^{z}$ never exceeds 1 and cannot overflow |
| `e / e.sum()` | Normalises to sum 1 |

The second call would give `nan` without the shift, since $e^{1000}$ overflows to infinity. With the shift it equals the softmax of $[-2, -1, 0]$, which is the softmax of $[0, 1, 2]$.

**Expected output:**

```text
[0.6652, 0.2447, 0.09]
[0.09, 0.2447, 0.6652]
```

(PyTorch's `torch.softmax` and `F.cross_entropy` perform this stabilisation for you.)

### 7.3.10 Comparison: sigmoid vs tanh vs ReLU

| Aspect | Sigmoid | Tanh | ReLU |
|---|---|---|---|
| Formula | $\dfrac{1}{1+e^{-z}}$ | $\dfrac{e^z - e^{-z}}{e^z + e^{-z}}$ | $\max(0, z)$ |
| Range | $(0, 1)$ | $(-1, 1)$ | $[0, \infty)$ |
| Zero-centred | No | Yes | No |
| Max derivative | $0.25$ | $1$ | $1$ |
| Saturates for large $\lvert z \rvert$ | Yes (both ends) | Yes (both ends) | Only on the negative side (gradient 0) |
| Vanishing gradient risk in deep nets | High | Moderate | Low (for active neurons) |
| Cost | Exponential | Exponential | Comparison |
| Typical role | Binary output, gates | Recurrent units, bounded outputs | **Default for hidden layers** |
| Main weakness | Saturation; slow, not zero-centred | Saturation | Dying neurons |

**Rule of thumb:** use ReLU (or a variant such as Leaky ReLU, or GELU in Transformers) in hidden layers; choose the output activation by the task (Section 7.4.5).

### 7.3.11 Common misconceptions about activations

| Misconception | Correction |
|---|---|
| "The activation function is applied to the weights." | It is applied to the pre-activation $z$ of each neuron, not to the weights |
| "Every layer needs the same activation." | Hidden and output layers usually differ |
| "Softmax is applied neuron by neuron." | It acts on the whole logit vector |
| "ReLU has no derivative, so it cannot be trained." | It is differentiable everywhere except exactly at 0, which is rare and handled by convention |
| "Sigmoid is obsolete." | It is still the standard for binary probability outputs and for gates |
| "A bigger network can skip the nonlinearity." | Without it, depth adds nothing (Section 3.5.3) |

---

## 7.4 Loss Functions

### 7.4.1 Definition and purpose

A **loss function** $\ell(\hat{y}, y)$ measures how wrong one prediction $\hat{y}$ is relative to the true target $y$. The **cost** (or **objective**) minimised in training is the average loss over a batch or dataset. Training is the search for parameters that make it small (Chapter 4), and by Chapter 5 every standard loss can be read as a negative log-likelihood under some assumption about the output.

A loss must be **differentiable** with respect to the network's output so that gradients exist. This is why we train with cross-entropy and report accuracy (Chapter 5, Section 5.7.6).

### 7.4.2 Mean Squared Error (MSE): for regression

$$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}\left(\hat{y}_i - y_i\right)^2$$

| Symbol | Meaning |
|---|---|
| $N$ | Number of predictions (examples, or examples times outputs) |
| $\hat{y}_i$ | Prediction $i$ |
| $y_i$ | True value $i$ |
| $(\hat{y}_i - y_i)^2$ | Squared error: always $\ge 0$, and large errors are penalised disproportionately |

**Origin:** it is the negative log-likelihood under Gaussian noise with fixed variance (Chapter 5, Section 5.6.4). Its derivative with respect to a prediction is $\dfrac{2(\hat{y}_i - y_i)}{N}$ (used in Chapter 8).

**Related losses:**

| Loss | Formula | Behaviour |
|---|---|---|
| **MAE** (mean absolute error, L1 loss) | $\dfrac{1}{N}\sum \lvert \hat{y}_i - y_i \rvert$ | Penalises errors linearly; less sensitive to outliers |
| **RMSE** | $\sqrt{\text{MSE}}$ | Same units as the target; often reported |
| **Huber / smooth L1** | Quadratic for small errors, linear for large | A compromise; optional topic |

**Worked example.** Predictions $[2.5,\ 0.0,\ 2.1,\ 7.8]$, targets $[3.0,\ -0.5,\ 2.0,\ 8.0]$:

| $i$ | $\hat{y}_i - y_i$ | Squared | Absolute |
|---|---|---|---|
| 1 | $-0.5$ | $0.25$ | $0.5$ |
| 2 | $0.5$ | $0.25$ | $0.5$ |
| 3 | $0.1$ | $0.01$ | $0.1$ |
| 4 | $-0.2$ | $0.04$ | $0.2$ |

$\text{MSE} = 0.55 / 4 = 0.1375$, $\ \text{MAE} = 1.3 / 4 = 0.325$, $\ \text{RMSE} = \sqrt{0.1375} \approx 0.3708$.

**Sensitivity to outliers.** If the last prediction were wrong by $10$ instead of $0.2$, the errors become $[0.5, 0.5, 0.1, 10]$: MSE $= (0.25 + 0.25 + 0.01 + 100)/4 = 25.1275$ while MAE $= 11.1/4 = 2.775$. One bad point dominates MSE far more than MAE.

```python
preds = torch.tensor([2.5, 0.0, 2.1, 7.8])
targets = torch.tensor([3.0, -0.5, 2.0, 8.0])

mse_manual = ((preds - targets) ** 2).mean()
mse_builtin = nn.MSELoss()(preds, targets)
mae_builtin = nn.L1Loss()(preds, targets)

print(round(mse_manual.item(), 4))
print(round(mse_builtin.item(), 4))
print(round(mae_builtin.item(), 4))
```

| Code | Explanation |
|---|---|
| `((preds - targets) ** 2).mean()` | The formula written directly |
| `nn.MSELoss()` | A loss module: first create it, then call it with `(predictions, targets)` |
| `nn.L1Loss()` | PyTorch's name for mean absolute error |

**Expected output:**

```text
0.1375
0.1375
0.325
```

**Shape pitfall (Chapter 2, Section 2.10.1).** If predictions have shape `(N, 1)` and targets have shape `(N,)`, subtraction silently broadcasts to `(N, N)` and the "loss" is wrong. PyTorch's `MSELoss` warns when shapes differ; make them match with `.squeeze(1)` or `.reshape(-1, 1)`.

### 7.4.3 Binary Cross-Entropy (BCE): for yes/no outputs

For a label $y \in \{0, 1\}$ and a predicted probability $p = \hat{y}$ from a sigmoid:

$$\text{BCE} = -\frac{1}{N}\sum_{i=1}^{N}\Big[\,y_i\ln p_i + (1 - y_i)\ln(1 - p_i)\,\Big]$$

| Symbol | Meaning |
|---|---|
| $p_i$ | The model's probability that example $i$ is class 1 |
| $y_i\ln p_i$ | Active only when the label is 1: penalises a low $p_i$ |
| $(1 - y_i)\ln(1-p_i)$ | Active only when the label is 0: penalises a high $p_i$ |

**Origin:** the negative log-likelihood of a Bernoulli model (Chapter 5).

**Worked example.** Labels $y = [1, 0, 1]$ and probabilities $p = [0.9, 0.2, 0.6]$:

| $i$ | Relevant probability | Loss $= -\ln(\cdot)$ |
|---|---|---|
| 1 ($y=1$) | $p = 0.9$ | $0.1054$ |
| 2 ($y=0$) | $1 - p = 0.8$ | $0.2231$ |
| 3 ($y=1$) | $p = 0.6$ | $0.5108$ |

Mean: $(0.1054 + 0.2231 + 0.5108)/3 = 0.8393/3 \approx 0.2798$.

```python
y = torch.tensor([1.0, 0.0, 1.0])
p = torch.tensor([0.9, 0.2, 0.6])

bce_manual = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
bce_probs = F.binary_cross_entropy(p, y)
bce_logits = F.binary_cross_entropy_with_logits(torch.logit(p), y)

print(round(bce_manual.item(), 4))
print(round(bce_probs.item(), 4))
print(round(bce_logits.item(), 4))
```

| Code | Explanation |
|---|---|
| `F.binary_cross_entropy(p, y)` | Takes **probabilities** (already passed through a sigmoid) |
| `torch.logit(p)` | The inverse of the sigmoid: $\ln\frac{p}{1-p}$. It converts probabilities to the raw logits that the next function expects |
| `F.binary_cross_entropy_with_logits(logits, y)` | Takes **raw logits** and applies the sigmoid internally in a numerically stable way. **This is the version to use in practice** |
| `y` is a float tensor | Binary targets must be floats of the same shape as the logits |

`F` is `torch.nn.functional`, imported as `import torch.nn.functional as F`.

**Expected output:**

```text
0.2798
0.2798
0.2798
```

**Why not MSE for classification?** With a sigmoid output $a = \sigma(z)$:

- BCE gives $\dfrac{\partial L}{\partial z} = a - y$.
- MSE gives $\dfrac{\partial L}{\partial z} = 2(a - y)\,\sigma'(z)$, which contains the factor $\sigma'(z)$ and therefore **vanishes when the neuron is saturated**.

Take a confidently wrong neuron: $z = -5$, so $a \approx 0.0067$, with true label $y = 1$.

| Loss | Gradient $\partial L/\partial z$ | Meaning |
|---|---|---|
| BCE | $a - y \approx -0.9933$ | Strong corrective signal |
| MSE | $2(a - y)\,\sigma'(z) \approx 2(-0.9933)(0.0066) \approx -0.0132$ | About 75 times weaker: learning stalls exactly when the model is most wrong |

```python
z = torch.tensor(-5.0, requires_grad=True)
loss_mse = (torch.sigmoid(z) - 1.0) ** 2
loss_mse.backward()
print(round(z.grad.item(), 4))

z = torch.tensor(-5.0, requires_grad=True)
loss_bce = F.binary_cross_entropy_with_logits(z, torch.tensor(1.0))
loss_bce.backward()
print(round(z.grad.item(), 4))
```

**Expected output:**

```text
-0.0132
-0.9933
```

### 7.4.4 Categorical Cross-Entropy: for multi-class outputs

With logits $\mathbf{z}_i \in \mathbb{R}^{K}$ and true class index $y_i$ for example $i$:

$$\text{CE} = -\frac{1}{N}\sum_{i=1}^{N}\ln\big[\text{softmax}(\mathbf{z}_i)\big]_{y_i} = \frac{1}{N}\sum_{i=1}^{N}\left[\ln\sum_{j=1}^{K}e^{z_{ij}} - z_{i,y_i}\right]$$

The second form shows what the loss does: it pushes the correct class's logit up *relative to* the log-sum-exp of all logits. It is the negative log-likelihood of a categorical model (Chapter 5, Section 5.7.2).

**Worked example.** Two samples, three classes. Logits $\begin{bmatrix} 2.0 & 1.0 & 0.0 \\ 0.5 & 2.5 & 0.1 \end{bmatrix}$, true classes $[0, 1]$.

- Sample 1: $\ln(e^2 + e^1 + e^0) - 2 = \ln(11.1073) - 2 = 2.4076 - 2 = 0.4076$.
- Sample 2: $\ln(e^{0.5} + e^{2.5} + e^{0.1}) - 2.5 = \ln(14.9364) - 2.5 = 2.7038 - 2.5 = 0.2038$.
- Mean: $(0.4076 + 0.2038)/2 \approx 0.3057$.

```python
logits = torch.tensor([[2.0, 1.0, 0.0],
                       [0.5, 2.5, 0.1]])          # shape (2, 3): N=2 samples, K=3 classes
targets = torch.tensor([0, 1])                    # shape (2,): class indices, int64

per_sample = F.cross_entropy(logits, targets, reduction="none")
mean_loss = F.cross_entropy(logits, targets)

print(per_sample)
print(round(mean_loss.item(), 4))
```

| Code | Explanation |
|---|---|
| `logits`, shape `(N, K)` | Raw network outputs; **no softmax applied** |
| `targets`, shape `(N,)` | Integer class indices (dtype `int64`), not one-hot vectors |
| `reduction="none"` | Returns one loss per sample instead of the average |
| Default reduction | Mean over the batch |
| `nn.CrossEntropyLoss()` | The module version of `F.cross_entropy`; same inputs |

**Expected output:**

```text
tensor([0.4076, 0.2038])
0.3057
```

**Sanity check you can use on any new classifier.** Before training, a network with small random weights outputs near-uniform probabilities, so the loss should be close to $\ln K$ (about $2.30$ for $K = 10$, about $0.69$ for $K = 2$). If the initial loss is far from $\ln K$, suspect a bug in the data, labels, or output layer.

### 7.4.5 Choosing the output layer and loss

| Task | Output neurons | Output activation | Loss | PyTorch loss (expects) |
|---|---|---|---|---|
| Regression (one or more real numbers) | $d$ | None (identity) | MSE (or MAE, Huber) | `nn.MSELoss` on predictions |
| Binary classification | 1 | Sigmoid | BCE | `nn.BCEWithLogitsLoss` on **logits**, float targets |
| Multi-class (exactly one class) | $K$ | Softmax | Categorical CE | `nn.CrossEntropyLoss` on **logits**, integer targets |
| Multi-label (any number of classes can be true) | $K$ | Sigmoid on each | BCE per label | `nn.BCEWithLogitsLoss` on **logits**, float 0/1 targets |

**Important PyTorch convention.** When you train with `BCEWithLogitsLoss` or `CrossEntropyLoss`, the model's last layer is a plain `nn.Linear` with **no** sigmoid or softmax. You apply `torch.sigmoid` or `torch.softmax` only when you want probabilities at prediction time. Putting the activation inside the model *and* using these losses applies it twice.

### 7.4.6 Comparison: loss functions vs activation functions

These two ideas are often confused because both involve functions of the network's outputs.

| | Activation function | Loss function |
|---|---|---|
| Purpose | Shapes what a layer outputs | Scores how wrong the final output is |
| Where it sits | Inside the network, after each layer | At the very end, after the output layer |
| Input | A neuron's pre-activation $z$ | The prediction and the true target |
| Output | Same shape as its input | One number (after averaging) |
| Has parameters? | Usually none | None |
| Chosen according to | Layer role (hidden vs output) and training behaviour | The type of problem and output distribution |
| Examples | ReLU, sigmoid, softmax | MSE, BCE, cross-entropy |
| Used at inference? | Yes | No (there is no target) |

**Related but different: loss vs metric.** The loss is what the optimiser minimises, so it must be differentiable. A **metric** such as accuracy, precision, or $R^2$ is what you report to judge the model (Chapter 12), and need not be differentiable.

---

## 7.5 Shallow vs Deep Networks

### 7.5.1 The universal approximation theorem

**Informal statement.** A feedforward network with **one hidden layer** containing a sufficient number of neurons and a suitable nonlinear activation can approximate any continuous function on a bounded region to any desired accuracy. Results of this kind were proved by Cybenko (1989) for sigmoid activations and by Hornik and others at around the same time for broader classes.

**What it tells us:** a single hidden layer is, in principle, expressive enough for a very wide family of functions. Networks are not limited the way a single neuron is (Chapter 6).

**What it does NOT tell us:**

| Not guaranteed | Why it matters |
|---|---|
| How many neurons are needed | It can be astronomically many for some functions |
| That training will *find* the right weights | Existence is not the same as discoverability (Chapters 8 to 11) |
| That the network will *generalise* to new data | Fitting the training set is not the goal (Chapter 12) |
| That a shallow network is *efficient* | Depth often achieves the same with far fewer parameters |

### 7.5.2 Why depth helps

- **Composition of simple features.** Real-world data is compositional: edges form shapes, shapes form parts, parts form objects (Chapter 1, Section 1.3.4). Each layer can reuse the features built by the previous one.
- **Efficiency.** There are theoretical results showing that certain functions can be represented by deep networks with far fewer neurons than any shallow network needs. In practice, deeper networks usually reach better accuracy for a given parameter budget on perception tasks.
- **Feature reuse.** Early layers learn general features that many later neurons share.

### 7.5.3 What depth costs

| Cost | Where it is addressed |
|---|---|
| Vanishing and exploding gradients | Chapter 10 |
| Harder optimisation, more hyperparameters | Chapters 10 and 11 |
| Greater risk of overfitting without enough data | Chapters 10 and 12 |
| More computation and memory | Chapters 2, 3, and 24 |
| Reduced interpretability | Chapter 24 |

### 7.5.4 Comparison: shallow vs deep

| Aspect | Shallow (one hidden layer) | Deep (many hidden layers) |
|---|---|---|
| Representational power | Universal in principle | Universal, usually far more parameter-efficient |
| Feature learning | One level of features | Hierarchy of features |
| Training difficulty | Easier | Needs careful initialisation, normalisation, and optimisers |
| Data needed | Often less | Often more (or a pretrained model) |
| Good for | Small or tabular problems; baselines | Images, speech, text, and other complex data |
| When to prefer | Start here to establish a baseline | Move here when the shallow model underfits |

**Practical advice:** begin with a small network and increase width or depth only while validation performance keeps improving.

### 7.5.5 Counting parameters in code

```python
shallow = nn.Sequential(
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 10),
)

deeper = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10),
)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

print(count_parameters(shallow))
print(count_parameters(deeper))
```

By hand: the shallow network has $(784 \times 128 + 128) + (128 \times 10 + 10) = 100{,}480 + 1{,}290 = 101{,}770$. The deeper one has $(784 \times 256 + 256) + (256 \times 128 + 128) + (128 \times 10 + 10) = 200{,}960 + 32{,}896 + 1{,}290 = 235{,}146$.

**Expected output:**

```text
101770
235146
```

---

## 7.6 Putting It Together: A Complete Forward Pass for a Classifier

Design for a task: classify $28 \times 28$ greyscale digit images into 10 classes.

| Design choice | Decision | Reason |
|---|---|---|
| Input | Flatten each image to 784 numbers | A dense layer expects a vector per sample |
| Hidden layer | 128 neurons, ReLU | Default hidden activation |
| Output layer | 10 neurons, **no** activation in the model | One logit per class |
| Loss | `CrossEntropyLoss` | Multi-class, one correct class |
| Parameters | 101,770 | From Section 7.5.5 |

```python
torch.manual_seed(0)

model = nn.Sequential(
    nn.Flatten(),                 # (N, 1, 28, 28) -> (N, 784)
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

images = torch.randn(32, 1, 28, 28)               # fake batch: N=32, C=1, H=28, W=28
labels = torch.randint(0, 10, (32,))              # fake labels in 0..9, shape (32,)

logits = model(images)
loss = F.cross_entropy(logits, labels)

print(tuple(logits.shape))
print(round(loss.item(), 1))
```

| Code | Explanation |
|---|---|
| `nn.Flatten()` | Keeps the batch dimension and flattens the rest: $1 \times 28 \times 28 = 784$ |
| `torch.randn(32, 1, 28, 28)` | Random stand-in for real images in `(N, C, H, W)` layout (Chapter 2) |
| `torch.randint(0, 10, (32,))` | Random class labels; `high` (10) is excluded |
| `logits` | Raw scores, shape `(32, 10)` |
| `F.cross_entropy(logits, labels)` | Average loss over the 32 samples |

**Expected output.** The shape is certain. The loss depends on the random weights, but for an untrained network with near-uniform predictions it should be close to $\ln 10 \approx 2.30$, so it should print a value near `2.3` (it may differ slightly, such as `2.2` or `2.4`):

```text
(32, 10)
2.3
```

This is the sanity check from Section 7.4.4. **Nothing has been learned here**: the weights are random and the labels are random. The next chapter shows how to reduce this loss.

---

## 7.7 Common Errors and Misconceptions

| Mistake or misconception | Correction |
|---|---|
| Omitting activations between hidden layers | The network collapses to one linear map (Section 3.5.3) |
| Putting `Softmax` in the model and then using `nn.CrossEntropyLoss` | The loss applies log-softmax itself; pass raw logits |
| Putting `Sigmoid` in the model and then using `BCEWithLogitsLoss` | Same problem: it applies the sigmoid itself |
| Passing one-hot targets to `CrossEntropyLoss` in the standard usage | Targets should be integer class indices of dtype `int64` |
| Integer or mismatched targets for `BCEWithLogitsLoss` | Targets must be floats with the same shape as the logits |
| An activation on the final layer of a regression network | Usually unnecessary; it limits the output range (a ReLU output cannot be negative, a sigmoid output cannot exceed 1) |
| Output size not matching the number of classes | The final layer needs exactly $K$ outputs for $K$ classes |
| Shape `(N, 1)` predictions vs shape `(N,)` targets | Silent broadcasting or a warning; align the shapes |
| "Deeper is always better." | Depth helps only with enough data and good training practice |
| "The universal approximation theorem means one hidden layer is enough in practice." | It guarantees existence, not efficiency, learnability, or generalisation |
| "A trained network's softmax output is the true probability." | It is the model's score, which may be badly calibrated (Chapter 5, Section 5.10.2) |
| "Initial loss does not matter." | Compare it with $\ln K$: a large gap flags bugs early |
| "Hidden neurons are described by the programmer." | Their meaning emerges from training; we only choose the architecture |

---

## 7.8 Comparison Summary

| Concept | Definition or formula | Role |
|---|---|---|
| Layer | $\mathbf{Z} = \mathbf{A}_{\text{prev}}\mathbf{W} + \mathbf{b}$, then $\mathbf{A} = \varphi(\mathbf{Z})$ | One step of the forward pass |
| Forward pass | Apply layers in order | Prediction and loss computation |
| Sigmoid | $1/(1+e^{-z})$ | Binary probability output, gates |
| Tanh | $(e^z - e^{-z})/(e^z + e^{-z})$ | Zero-centred bounded activation |
| ReLU | $\max(0, z)$ | Default hidden activation |
| Leaky ReLU | $z$ if $z>0$ else $\alpha z$ | Avoids dying neurons |
| Softmax | $e^{z_k}/\sum_j e^{z_j}$ | Multi-class probabilities |
| MSE | $\frac1N\sum(\hat y - y)^2$ | Regression loss |
| BCE | $-[y\ln p + (1-y)\ln(1-p)]$ | Binary classification loss |
| Cross-entropy | $-\ln\text{softmax}(\mathbf{z})_y$ | Multi-class classification loss |
| Universal approximation | One wide hidden layer can approximate any continuous function | Existence, not efficiency |
| Depth | Many layers of composed features | Efficiency and hierarchy |

---

## Exercises

1. **Forward pass by hand.** For the network in Section 7.2.2, compute the output for the input $\mathbf{x} = [0, 1]$. Check with the NumPy code.
2. **Shapes.** A network has layers $20 \to 64 \to 32 \to 5$. Give the shape of every weight matrix and bias vector, the number of parameters in each layer, and the total. Verify with `nn.Sequential` and `count_parameters`.
3. **Batch shapes.** For a batch of 100 samples through the network in Exercise 2, give the shape of $\mathbf{A}^{(\ell)}$ after each layer.
4. **Activation derivatives.** Compute $\sigma'(z)$, $\tanh'(z)$, and $\text{ReLU}'(z)$ at $z = -3$, $z = 0.5$, and $z = 3$ by hand. Check with autograd.
5. **Collapse.** Build a two-layer network by hand with no activation function between the layers, and show numerically that it equals a single layer with weights $\mathbf{W}^{(1)}\mathbf{W}^{(2)}$. (Use the row convention $\mathbf{x}\mathbf{W}^{(1)}\mathbf{W}^{(2)}$.)
6. **Softmax.** Compute softmax by hand for logits $[3, 1, 0.2]$. Then show that adding $100$ to every logit leaves the result unchanged.
7. **MSE and MAE.** For predictions $[1, 2, 3, 4]$ and targets $[1, 2, 3, 14]$, compute MSE and MAE. Which one is more affected by the outlier, and by roughly what factor compared with a version with the last target equal to 4.5?
8. **BCE.** For $y = [1, 1, 0, 0]$ and $p = [0.8, 0.4, 0.3, 0.1]$, compute the BCE by hand and verify with `F.binary_cross_entropy`.
9. **Cross-entropy.** For logits $\begin{bmatrix} 1 & 3 & 0 \end{bmatrix}$ and true class $2$ (the second class, index 1), compute the loss by hand using both forms of the formula in Section 7.4.4.
10. **Initial loss.** Build a classifier for 5 classes, run a random batch through it, and check that the initial loss is near $\ln 5 \approx 1.61$.
11. **Design.** For each task, choose the number of output neurons, the output activation, and the loss: (a) predicting house prices, (b) deciding whether an email is spam, (c) assigning a news article to exactly one of 8 topics, (d) tagging a photo with any subset of 20 labels.
12. **Reflection.** In your own words, explain the difference between the universal approximation theorem and the practical advantage of depth.
