# Chapter 6: From Neuron to Perceptron

**Level 1: Essential Foundations | Tag: Essential**

## Chapter Roadmap

The mathematics is now in place: vectors and dot products (Chapter 3), gradients (Chapter 4), and probability and loss (Chapter 5). This chapter builds the first real learning machine, the **artificial neuron**, and shows both what it can do and where it hits a wall.

By the end of this chapter you will be able to:

- Explain the biological analogy and where it stops being accurate.
- Compute the output of an artificial neuron from its inputs, weights, bias, and activation.
- Explain what weights and biases each do, geometrically and intuitively.
- Build logic gates (AND, OR, NAND) by choosing weights by hand.
- Train a perceptron with the perceptron learning rule and trace it step by step.
- Define **linear separability** and prove that XOR cannot be solved by a single perceptron.
- Show how a second layer solves XOR, which motivates deep networks.
- Explain why the step function must be replaced by a smooth function for gradient-based training, and connect a sigmoid neuron to logistic regression.
- Distinguish a perceptron from a modern artificial neural network.

**Prerequisite gap covered here (from the syllabus):** logistic regression as a "single neuron" bridge.

> **Honesty note.** The code in this chapter has not been executed by me. Every "Expected output" was derived by hand by tracing the algorithm. Where results depend on random initialisation, the text says so. Run each block and compare.

---

## 6.1 Biological Inspiration

### 6.1.1 The biological neuron, simplified

A neuron in the brain is a cell with four main parts relevant to our analogy:

| Part | Biological role |
|---|---|
| **Dendrites** | Branches that receive electrical signals from other neurons |
| **Synapses** | The junctions where signals pass between neurons. Their *strength* varies |
| **Cell body (soma)** | Combines (integrates) the incoming signals |
| **Axon** | Carries the output signal to other neurons |

**Simplified behaviour.** A neuron collects signals from many other neurons. Some connections excite it, others inhibit it, and some are stronger than others. If the combined input is strong enough, the neuron **fires** and sends a signal onward. Learning in the brain is associated with changes in the strengths of these connections (synaptic plasticity).

A commonly cited estimate puts the human brain at roughly 86 billion neurons, each connected to many others. Intelligence arises from the pattern and strength of these connections, not from any single neuron.

### 6.1.2 The mapping to an artificial neuron

| Biological | Artificial | Symbol |
|---|---|---|
| Incoming signals | Input values | $x_1, \dots, x_n$ |
| Synaptic strength (excitatory or inhibitory) | **Weights** (positive or negative) | $w_1, \dots, w_n$ |
| Cell body integrating signals | **Weighted sum** | $\sum w_i x_i$ |
| Firing threshold | **Bias** and activation function | $b$, $\varphi$ |
| Axon output | Output value | $a$ |

### 6.1.3 How far the analogy goes

The analogy was the *starting point* of the field, but it should not be pushed far.

| Misconception | Correction |
|---|---|
| "Artificial neural networks simulate the brain." | An artificial neuron is a few arithmetic operations. Real neurons involve complex chemistry, timing, and structure that these models ignore. |
| "Deep Learning works because it copies how brains learn." | Networks are trained by gradient descent and backpropagation (Chapter 8). There is no established evidence that brains learn the same way. |
| "More neurons means more intelligence." | Capability depends on architecture, data, and training, not on neuron count alone. |

Think of the biological picture as a source of **vocabulary and motivation** ("neuron", "weight", "activation"), and of the mathematics as the actual definition.

---

## 6.2 The Artificial Neuron

### 6.2.1 Formal definition

An **artificial neuron** takes an input vector $\mathbf{x} \in \mathbb{R}^{n}$ and computes in two stages:

$$z = \mathbf{w} \cdot \mathbf{x} + b = \sum_{i=1}^{n} w_i x_i + b \qquad\qquad a = \varphi(z)$$

| Symbol | Meaning |
|---|---|
| $\mathbf{x} = [x_1, \dots, x_n]$ | The inputs (features, or outputs of earlier neurons) |
| $\mathbf{w} = [w_1, \dots, w_n]$ | The **weights**: one learnable number per input |
| $b$ | The **bias**: one learnable number per neuron |
| $z$ | The **pre-activation** (also called the logit or net input) |
| $\varphi$ | The **activation function**: a fixed, usually nonlinear function |
| $a$ | The neuron's output (its **activation**) |

The dot product is from Chapter 3. Section 3.3.3 already computed $z = 1.5$ for $\mathbf{w} = [0.5, -1, 2]$, $\mathbf{x} = [4, 3, 1]$, $b = 0.5$.

```text
   x1 ----w1---\
                \
   x2 ----w2----(  Sum: z = w.x + b  )----[ activation phi ]----> a
                /               ^
   x3 ----w3---/                |
                               bias b
```

### 6.2.2 Simple explanation

A neuron is a **weighted vote followed by a decision**. Each input casts a vote scaled by its weight; the bias shifts how easily the neuron says yes; the activation turns the total into an output.

### 6.2.3 What weights and biases do

**Weights** express how much each input matters, and in which direction:

| Weight | Meaning |
|---|---|
| Large positive $w_i$ | Input $i$ strongly pushes the neuron toward "on" |
| Large negative $w_i$ | Input $i$ strongly pushes it toward "off" |
| $w_i \approx 0$ | Input $i$ is effectively ignored |

**Bias** is a baseline: the neuron's tendency to fire when all inputs are zero. With a threshold-style activation, a bias of $b$ is equivalent to a firing threshold of $-b$: the neuron fires when $\mathbf{w}\cdot\mathbf{x} \ge -b$.

**Number of learnable parameters:** a neuron with $n$ inputs has $n$ weights plus 1 bias, so $n + 1$ parameters. A neuron reading a $28 \times 28$ image flattened to 784 inputs has $784 + 1 = 785$ parameters; ten such neurons have $784 \times 10 + 10 = 7{,}850$ (the formula from Section 3.6.6).

**The "always-on input" trick.** Treat the bias as a weight on an extra input that is always $1$: define $\tilde{\mathbf{x}} = [1, x_1, \dots, x_n]$ and $\tilde{\mathbf{w}} = [b, w_1, \dots, w_n]$. Then $z = \tilde{\mathbf{w}} \cdot \tilde{\mathbf{x}}$. This explains why the learning rule in Section 6.5 updates the bias just like a weight whose input is 1.

### 6.2.4 Practical Python

```python
import numpy as np

def step(z):
    return 1 if z >= 0 else 0

def neuron(x, w, b, activation):
    z = np.dot(w, x) + b
    return z, activation(z)

x = np.array([4.0, 3.0, 1.0])
w = np.array([0.5, -1.0, 2.0])
b = 0.5

z, a = neuron(x, w, b, step)
print(z)
print(a)
```

| Code | Explanation |
|---|---|
| `def step(z)` | The step (threshold) activation: output 1 when $z \ge 0$, else 0 |
| `np.dot(w, x) + b` | The pre-activation $z = \mathbf{w}\cdot\mathbf{x} + b$ |
| `activation` passed as an argument | Functions can be passed to functions (Chapter 4's `numerical_derivative` did the same) |
| `return z, a` | Returns both values so we can inspect them |

**Hand calculation:** $z = 0.5(4) + (-1)(3) + 2(1) + 0.5 = 1.5$; since $1.5 \ge 0$, the step outputs $1$.

**Expected output:**

```text
1.5
1
```

---

## 6.3 The Perceptron

### 6.3.1 Formal definition

The **perceptron**, proposed by Frank Rosenblatt in 1958, is an artificial neuron whose activation is the **step function** (Heaviside function):

$$\hat{y} = \text{step}(\mathbf{w}\cdot\mathbf{x} + b), \qquad \text{step}(z) = \begin{cases} 1 & z \ge 0 \\ 0 & z < 0 \end{cases}$$

(The convention at exactly $z = 0$ varies between textbooks; here we output 1.)

### 6.3.2 Why it was a landmark

It was one of the first models that could **learn its weights from examples** instead of having them programmed. That idea, a function with adjustable numbers fitted to data (Chapter 1), is the seed of everything in this book.

### 6.3.3 Geometric meaning: a decision boundary

The perceptron divides the input space into two regions. The boundary is where $z = 0$:

$$w_1 x_1 + w_2 x_2 + b = 0$$

In two dimensions this is a **straight line**; in three, a plane; in $n$ dimensions, a **hyperplane**.

- The weight vector $\mathbf{w}$ is **perpendicular** to the boundary and points toward the "1" side.
- The bias $b$ **shifts** the boundary away from the origin without rotating it.
- Points on one side get $\hat{y} = 1$, points on the other get $\hat{y} = 0$.

**Worked example.** Take $\mathbf{w} = [1, 1]$ and $b = -1.5$. The boundary is $x_1 + x_2 = 1.5$.

| Point | $z = x_1 + x_2 - 1.5$ | Output |
|---|---|---|
| $(0, 0)$ | $-1.5$ | 0 |
| $(1, 0)$ | $-0.5$ | 0 |
| $(1, 1)$ | $0.5$ | 1 |

Only the point $(1,1)$ lies on the "1" side. We have just built an **AND gate**.

---

## 6.4 Logic Gates with Hand-Chosen Weights

Because the perceptron is a weighted vote with a threshold, we can pick weights that implement logic.

| Gate | Weights $\mathbf{w}$ | Bias $b$ | Boundary |
|---|---|---|---|
| AND | $[1, 1]$ | $-1.5$ | $x_1 + x_2 = 1.5$ |
| OR | $[1, 1]$ | $-0.5$ | $x_1 + x_2 = 0.5$ |
| NAND | $[-1, -1]$ | $1.5$ | $-x_1 - x_2 + 1.5 = 0$ |

**Verification by hand** (all four input pairs, listed as $(x_1, x_2) \to z \to$ output):

| Input | AND $z$ | AND out | OR $z$ | OR out | NAND $z$ | NAND out |
|---|---|---|---|---|---|---|
| $(0,0)$ | $-1.5$ | 0 | $-0.5$ | 0 | $1.5$ | 1 |
| $(0,1)$ | $-0.5$ | 0 | $0.5$ | 1 | $0.5$ | 1 |
| $(1,0)$ | $-0.5$ | 0 | $0.5$ | 1 | $0.5$ | 1 |
| $(1,1)$ | $0.5$ | 1 | $1.5$ | 1 | $-0.5$ | 0 |

```python
gates = {
    "AND":  (np.array([1.0, 1.0]), -1.5),
    "OR":   (np.array([1.0, 1.0]), -0.5),
    "NAND": (np.array([-1.0, -1.0]), 1.5),
}
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

for name, (w, b) in gates.items():
    outputs = [step(np.dot(w, x) + b) for x in inputs]
    print(name, outputs)
```

| Code | Explanation |
|---|---|
| `gates` dictionary | Maps a name to a `(weights, bias)` pair |
| `for name, (w, b) in gates.items()` | Loops over entries, unpacking the pair directly |
| `[step(...) for x in inputs]` | A list comprehension applying the neuron to each input pair |
| `np.dot(w, x)` | Accepts a tuple `x` and treats it as an array |

**Expected output:**

```text
AND [0, 0, 0, 1]
OR [0, 1, 1, 1]
NAND [1, 1, 1, 0]
```

**Why NAND matters:** NAND is *universal*: any digital circuit can be built from NAND gates alone. A network of perceptrons can therefore compute anything a digital computer can, which is why layers of neurons are such a powerful idea. A single perceptron is far less powerful, as Section 6.7 shows.

---

## 6.5 The Perceptron Learning Rule

### 6.5.1 Formal rule

Instead of choosing weights by hand, we can learn them. For each training example $(\mathbf{x}, y)$ with $y \in \{0, 1\}$:

1. Compute the prediction $\hat{y} = \text{step}(\mathbf{w}\cdot\mathbf{x} + b)$.
2. Compute the error $e = y - \hat{y}$, which is $0$, $+1$, or $-1$.
3. Update:

$$\mathbf{w} \leftarrow \mathbf{w} + \eta\, e\, \mathbf{x}, \qquad b \leftarrow b + \eta\, e$$

| Symbol | Meaning |
|---|---|
| $e = y - \hat{y}$ | The error. $0$ means correct; no change |
| $\eta$ | The learning rate (Chapter 4) |
| $e\,\mathbf{x}$ | The input vector, scaled by the error and its sign |

### 6.5.2 Intuition

| Situation | $e$ | Effect of the update |
|---|---|---|
| Predicted 0 but the answer is 1 | $+1$ | Add $\eta\mathbf{x}$ to $\mathbf{w}$: the score $\mathbf{w}\cdot\mathbf{x}$ for this input rises (raising it by $\eta\lVert\mathbf{x}\rVert^2$, using Chapter 3's dot product), and the bias rises too |
| Predicted 1 but the answer is 0 | $-1$ | Subtract $\eta\mathbf{x}$: the score for this input falls |
| Correct | $0$ | Nothing changes |

The perceptron only learns from its **mistakes**. This rule is not gradient descent on a differentiable loss (the step function has no useful derivative, Section 6.8), but it has the same flavour: nudge the parameters in the direction that fixes the current error.

### 6.5.3 Worked example: learning OR from scratch

Data: $(0,0)\to 0,\ (0,1)\to 1,\ (1,0)\to 1,\ (1,1)\to 1$. Start with $\mathbf{w} = [0, 0]$, $b = 0$, $\eta = 1$, and process the examples in this order each epoch.

**Epoch 1**

| Example | $z$ | $\hat{y}$ | $y$ | $e$ | New $\mathbf{w}$ | New $b$ |
|---|---|---|---|---|---|---|
| $(0,0)$ | $0$ | 1 (since $z \ge 0$) | 0 | $-1$ | $[0, 0]$ | $-1$ |
| $(0,1)$ | $-1$ | 0 | 1 | $+1$ | $[0, 1]$ | $0$ |
| $(1,0)$ | $0$ | 1 | 1 | $0$ | $[0, 1]$ | $0$ |
| $(1,1)$ | $1$ | 1 | 1 | $0$ | $[0, 1]$ | $0$ |

**Epoch 2** (start: $\mathbf{w} = [0,1]$, $b = 0$)

| Example | $z$ | $\hat{y}$ | $y$ | $e$ | New $\mathbf{w}$ | New $b$ |
|---|---|---|---|---|---|---|
| $(0,0)$ | $0$ | 1 | 0 | $-1$ | $[0, 1]$ | $-1$ |
| $(0,1)$ | $0$ | 1 | 1 | $0$ | $[0, 1]$ | $-1$ |
| $(1,0)$ | $-1$ | 0 | 1 | $+1$ | $[1, 1]$ | $0$ |
| $(1,1)$ | $2$ | 1 | 1 | $0$ | $[1, 1]$ | $0$ |

**Epoch 3** (start: $\mathbf{w} = [1,1]$, $b = 0$): only $(0,0)$ is wrong ($z = 0 \Rightarrow \hat{y} = 1$ but $y = 0$), so $b \leftarrow -1$. Final: $\mathbf{w} = [1, 1]$, $b = -1$.

**Epoch 4:** $(0,0)$ gives $z = -1 \to 0$ (correct), $(0,1)$ and $(1,0)$ give $z = 0 \to 1$ (correct), $(1,1)$ gives $z = 1 \to 1$ (correct). **Zero errors: training has converged.**

The learned boundary is $x_1 + x_2 - 1 = 0$. Compare with the hand-built OR gate ($b = -0.5$): the boundary is slightly different but the classification of the four points is identical. There are many valid solutions.

### 6.5.4 Practical Python

```python
def train_perceptron(X, y, lr=1.0, max_epochs=100, verbose=True):
    w = np.zeros(X.shape[1])          # one weight per input feature
    b = 0.0
    history = []                      # number of mistakes in each epoch

    for epoch in range(1, max_epochs + 1):
        errors = 0
        for x_i, y_i in zip(X, y):
            y_hat = step(np.dot(w, x_i) + b)
            error = y_i - y_hat
            if error != 0:
                w = w + lr * error * x_i
                b = b + lr * error
                errors += 1
        history.append(errors)
        if verbose:
            print(f"epoch {epoch}: w = {w}, b = {b}, errors = {errors}")
        if errors == 0:
            break
    return w, b, history

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

w, b, history = train_perceptron(X, y_or)
```

| Code | Explanation |
|---|---|
| `X.shape[1]` | Number of columns of `X`: the number of input features (2 here) |
| `np.zeros(X.shape[1])` | Weights start at zero (fine for a perceptron; Chapter 10 explains why zero initialisation is a **bad** idea for deeper networks) |
| `zip(X, y)` | Pairs each input row with its label |
| `error = y_i - y_hat` | $y - \hat{y}$, so $+1$, $-1$, or $0$ |
| `w = w + lr * error * x_i` | $\mathbf{w} \leftarrow \mathbf{w} + \eta e \mathbf{x}$. Shapes: scalar times `(2,)` array |
| `b = b + lr * error` | The bias update (input fixed at 1) |
| `if errors == 0: break` | Stop once a full pass has no mistakes |

**Expected output (matches the hand trace above):**

```text
epoch 1: w = [0. 1.], b = 0.0, errors = 2
epoch 2: w = [1. 1.], b = 0.0, errors = 2
epoch 3: w = [1. 1.], b = -1.0, errors = 1
epoch 4: w = [1. 1.], b = -1.0, errors = 0
```

### 6.5.5 Convergence

**Perceptron convergence theorem (informal statement).** If the training data is **linearly separable** (some line or hyperplane perfectly separates the two classes), the perceptron learning rule is guaranteed to find a separating boundary in a finite number of updates. If the data is **not** linearly separable, the rule never settles: it keeps updating forever.

---

## 6.6 Linear Separability

**Definition.** Two classes of points are **linearly separable** if a single straight line (in 2-D; a hyperplane in higher dimensions) can place all points of one class on one side and all points of the other class on the other side.

| Problem | Linearly separable? |
|---|---|
| AND | Yes |
| OR | Yes |
| NAND | Yes |
| XOR | **No** |
| Points inside a circle vs outside it (Chapter 1) | No |

A single neuron with a threshold activation can represent **only** linearly separable problems. The same is true of a single neuron with a smooth activation, as we will see in Section 6.8.

---

## 6.7 The XOR Problem

### 6.7.1 Statement

**XOR** (exclusive OR) outputs 1 when exactly one input is 1:

| $x_1$ | $x_2$ | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Plotted in the plane, the two 1s sit at opposite corners $(0,1)$ and $(1,0)$, and the two 0s sit at the other corners $(0,0)$ and $(1,1)$. No single straight line can put the 1s on one side and the 0s on the other.

### 6.7.2 Proof that no perceptron solves XOR

Suppose weights $w_1, w_2$ and bias $b$ existed. The four rows would require:

| Row | Requirement | Inequality |
|---|---|---|
| $(0,0) \to 0$ | $z < 0$ | $b < 0$ |
| $(0,1) \to 1$ | $z \ge 0$ | $w_2 + b \ge 0$ |
| $(1,0) \to 1$ | $z \ge 0$ | $w_1 + b \ge 0$ |
| $(1,1) \to 0$ | $z < 0$ | $w_1 + w_2 + b < 0$ |

Add the middle two inequalities: $w_1 + w_2 + 2b \ge 0$, which can be rewritten as $w_1 + w_2 + b \ge -b$. From the first row, $b < 0$, so $-b > 0$, which gives $w_1 + w_2 + b > 0$. This contradicts the fourth row, which demands $w_1 + w_2 + b < 0$. **No such weights exist.** $\blacksquare$

### 6.7.3 Historical note

In 1969, Marvin Minsky and Seymour Papert published *Perceptrons*, which analysed these limits rigorously. It is widely credited with contributing to a decline in interest and funding for neural-network research through the 1970s, although historians disagree about how large its influence actually was. Interest returned once multi-layer networks could be trained efficiently with backpropagation (Chapter 8).

### 6.7.4 Watching the perceptron fail

```python
y_xor = np.array([0, 1, 1, 0])

w, b, history = train_perceptron(X, y_xor, max_epochs=100, verbose=False)

print("epochs run:", len(history))
print("any epoch with zero errors:", 0 in history)
```

| Code | Explanation |
|---|---|
| `verbose=False` | Suppresses the per-epoch printout (it would be 100 lines) |
| `len(history)` | Number of epochs actually run |
| `0 in history` | `True` if any epoch had zero mistakes |

Since no weights can classify all four XOR points, no epoch can finish with zero mistakes, so the loop always runs to the limit.

**Expected output:**

```text
epochs run: 100
any epoch with zero errors: False
```

### 6.7.5 The way out: add a layer

XOR can be written as a combination of gates that a perceptron *can* represent:

$$\text{XOR}(x_1, x_2) = \text{AND}\big(\text{OR}(x_1, x_2),\ \text{NAND}(x_1, x_2)\big)$$

This uses **two layers**: a hidden layer computing OR and NAND, then an output neuron computing AND of those two results.

| $(x_1, x_2)$ | $h_1 = $ OR | $h_2 = $ NAND | Output $=$ AND$(h_1, h_2)$ |
|---|---|---|---|
| $(0,0)$ | 0 | 1 | 0 |
| $(0,1)$ | 1 | 1 | 1 |
| $(1,0)$ | 1 | 1 | 1 |
| $(1,1)$ | 1 | 0 | 0 |

**Why it works.** In the original space $(x_1, x_2)$ the classes are not separable. The hidden layer maps each point to a new space $(h_1, h_2)$:

- $(0,0) \mapsto (0, 1)$
- $(0,1) \mapsto (1, 1)$
- $(1,0) \mapsto (1, 1)$
- $(1,1) \mapsto (1, 0)$

In the new space, both XOR-1 points land on $(1,1)$, and the XOR-0 points land on $(0,1)$ and $(1,0)$. These **are** linearly separable (for example by the line $h_1 + h_2 = 1.5$). **The hidden layer has learned a new representation that makes the problem easy.** This is exactly the idea of representation learning from Chapter 1, in its smallest form.

```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])        # shape (4, 2)

# Hidden layer: column 0 is the OR neuron, column 1 is the NAND neuron
W1 = np.array([[ 1.0, -1.0],
               [ 1.0, -1.0]])                          # shape (2, 2)
b1 = np.array([-0.5, 1.5])                             # shape (2,)

H = (X @ W1 + b1 >= 0).astype(int)                     # shape (4, 2)

# Output layer: AND of the two hidden neurons
W2 = np.array([1.0, 1.0])                              # shape (2,)
b2 = -1.5

out = (H @ W2 + b2 >= 0).astype(int)                   # shape (4,)

print(H)
print(out)
```

| Code | Explanation |
|---|---|
| `W1`, shape `(2, 2)` | Each **column** holds one hidden neuron's weights (Chapter 3's $(d_{\text{in}}, d_{\text{out}})$ layout). Column 0 is $[1,1]$ (OR); column 1 is $[-1,-1]$ (NAND) |
| `X @ W1 + b1` | The layer formula $\mathbf{X}\mathbf{W} + \mathbf{b}$, shape `(4,2)` |
| `>= 0` then `.astype(int)` | Applies the step function to every element at once: a boolean array converted to 0/1 |
| `H @ W2 + b2` | Shape `(4, 2) @ (2,)` gives `(4,)`: the AND neuron applied to each row of `H` |

**Expected output** (the pairs in `H` match the table of $(h_1, h_2)$ above):

```text
[[0 1]
 [1 1]
 [1 1]
 [1 0]]
[0 1 1 0]
```

We chose these weights **by hand**. Learning them automatically through a hidden layer requires a way to assign blame to hidden neurons: backpropagation (Chapter 8). It also requires an activation function that has useful gradients, which we examine next.

---

## 6.8 From Step to Smooth: The Bridge to Logistic Regression

### 6.8.1 The problem with the step function

Gradient-based learning (Chapter 4) needs a derivative that carries information. The step function is flat except for a jump at $0$:

- Its derivative is $0$ everywhere it exists, so a small change in a weight causes **no change in the loss** and gives no gradient signal.
- At the jump the derivative is undefined.

That is why the perceptron needed its special learning rule, which does not generalise to multi-layer networks.

### 6.8.2 The sigmoid neuron

Replace the step with the **sigmoid** (logistic) function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

| Property | Value |
|---|---|
| Range | Strictly between 0 and 1 |
| $\sigma(0)$ | $0.5$ |
| Large positive $z$ | Approaches 1 |
| Large negative $z$ | Approaches 0 |
| Derivative | $\sigma(z)\,(1 - \sigma(z))$, smooth and nonzero everywhere (derived in Chapter 7) |

The neuron's output $a = \sigma(\mathbf{w}\cdot\mathbf{x} + b)$ can now be read as a **probability**: $P(y = 1 \mid \mathbf{x})$ under a Bernoulli model (Chapter 5). Training maximises the likelihood, which means minimising the **binary cross-entropy** loss:

$$L = -\big[\,y\ln a + (1 - y)\ln(1 - a)\,\big]$$

**This single sigmoid neuron trained with binary cross-entropy is logistic regression**, a model you may already know from classical ML. A "deep" network is built from many such units. The Chapter 1 statement that your classical ML knowledge carries over is concrete here.

| | Step neuron (perceptron) | Sigmoid neuron (logistic regression) |
|---|---|---|
| Output | Hard 0 or 1 | A probability between 0 and 1 |
| Differentiable | No useful gradient | Yes |
| Training rule | Perceptron rule (mistake-driven) | Gradient descent on cross-entropy |
| Tells you confidence | No | Yes |
| Decision boundary | Linear | **Still linear** (where $z = 0$, i.e. $a = 0.5$) |

The last row is crucial: a smooth activation fixes the *training* problem but not the *capacity* problem. A single sigmoid neuron still cannot solve XOR.

### 6.8.3 Worked example

Using the neuron from Section 6.2.4 ($z = 1.5$):

$$a = \sigma(1.5) = \frac{1}{1 + e^{-1.5}} = \frac{1}{1 + 0.2231} \approx 0.8176$$

- If the true label is $y = 1$: $L = -\ln(0.8176) \approx 0.2014$ (small, the model was right and fairly confident).
- If the true label is $y = 0$: $L = -\ln(1 - 0.8176) = -\ln(0.1824) \approx 1.7014$ (large, confident and wrong).

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

a = sigmoid(1.5)
print(round(a, 4))
print(round(-np.log(a), 4))
print(round(-np.log(1 - a), 4))
```

**Expected output:**

```text
0.8176
0.2014
1.7014
```

**A gradient worth remembering.** For a sigmoid output with binary cross-entropy, the derivative of the loss with respect to the pre-activation is simply

$$\frac{\partial L}{\partial z} = a - y$$

(prediction minus target). For $y = 1$ here that is $0.8176 - 1 = -0.1824$: negative, meaning *increasing $z$ lowers the loss*, as it should. We check this with autograd (Chapter 4):

```python
import torch
import torch.nn.functional as F

z = torch.tensor(1.5, requires_grad=True)
y = torch.tensor(1.0)

loss = F.binary_cross_entropy_with_logits(z, y)
loss.backward()

print(round(loss.item(), 4))
print(z.grad)
```

| Code | Explanation |
|---|---|
| `F.binary_cross_entropy_with_logits(z, y)` | Applies the sigmoid and the binary cross-entropy together in a numerically stable way. It takes the **raw** $z$ (the logit), so you must **not** apply a sigmoid first |
| `z.grad` | $\partial L / \partial z$ after `backward()` |

**Expected output:**

```text
0.2014
tensor(-0.1824)
```

### 6.8.4 Training a single sigmoid neuron in PyTorch

A complete small example: learn OR, then try XOR, with one neuron.

```python
import torch.nn as nn

X_t = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])   # shape (4, 2)

def train_single_neuron(targets, steps=2000, lr=0.5):
    torch.manual_seed(0)
    layer = nn.Linear(2, 1)                          # one neuron: 2 weights + 1 bias
    y_t = torch.tensor(targets, dtype=torch.float32).reshape(4, 1)

    for _ in range(steps):
        logits = layer(X_t)                          # shape (4, 1)
        loss = F.binary_cross_entropy_with_logits(logits, y_t)
        layer.zero_grad()                            # clear old gradients
        loss.backward()                              # compute new gradients
        with torch.no_grad():
            for p in layer.parameters():
                p -= lr * p.grad                     # gradient descent step

    with torch.no_grad():
        preds = (torch.sigmoid(layer(X_t)) >= 0.5).long().flatten()
    return preds

or_preds = train_single_neuron([0, 1, 1, 1])
xor_preds = train_single_neuron([0, 1, 1, 0])

print(or_preds)
print("XOR correct:", (xor_preds == torch.tensor([0, 1, 1, 0])).sum().item(), "of 4")
```

| Code | Explanation |
|---|---|
| `nn.Linear(2, 1)` | One neuron with two inputs: weight shape `(1, 2)`, bias shape `(1,)` (Chapter 3). Random initial values, hence the seed |
| `.reshape(4, 1)` | Targets must match the logits' shape `(4, 1)` |
| `layer.zero_grad()` | Clears accumulated gradients (Chapter 4, Section 4.7.4) |
| `loss.backward()` | Computes gradients for the weight and bias |
| `p -= lr * p.grad` inside `no_grad()` | The gradient descent update $\theta \leftarrow \theta - \eta\nabla L$ |
| `torch.sigmoid(...) >= 0.5` | Turns probabilities into hard predictions |
| `.long().flatten()` | Integer dtype and shape `(4,)` for printing and comparison |

**Expected behaviour (not executed; trained by gradient descent from a random start, so run it to confirm):**

- OR is linearly separable, so the neuron should reach all four correct: `tensor([0, 1, 1, 1])`.
- XOR is not, so the neuron can classify **at most 3 of 4** points correctly, however long it trains. The printed count will be 3 or fewer, never 4.

This repeats, with a smooth neuron and a proper loss, what Section 6.7.4 showed with the perceptron: more careful training does not remove a limitation that comes from the model's *shape*.

---

## 6.9 Perceptron vs Artificial Neural Network

### 6.9.1 Definitions

- A **perceptron** is a single threshold neuron (step activation), possibly arranged as one layer of such neurons.
- An **artificial neural network (ANN)** is a network of neurons arranged in layers, with **nonlinear** activations, whose weights are trained by gradient-based methods. When it has at least one hidden layer it is a **multi-layer perceptron (MLP)**, also called a feedforward network.

(The name "multi-layer perceptron" is historical. Modern MLPs use smooth or piecewise-linear activations such as sigmoid or ReLU rather than the step function.)

### 6.9.2 Comparison

| Aspect | Perceptron | Sigmoid neuron (logistic regression) | Multi-layer ANN (MLP) |
|---|---|---|---|
| Layers | One neuron or one layer | One neuron | Input, one or more hidden layers, output |
| Activation | Step | Sigmoid | Sigmoid, tanh, ReLU, and others (Chapter 7) |
| Output | Hard 0/1 | Probability | Probabilities or real values |
| Learning rule | Perceptron rule | Gradient descent | Backpropagation + gradient descent (Chapter 8) |
| Decision boundary | Straight line / hyperplane | Straight line / hyperplane | Complex, curved |
| Solves XOR? | No | No | **Yes** |
| Learns new features? | No | No | **Yes**, in hidden layers |
| Typical use today | Teaching, historical | Baseline classifier | The basic building block of deep learning |

**When is each preferable?** A single neuron or logistic regression is an excellent **baseline**: fast, interpretable, and often surprisingly strong on tabular data (Chapter 1). Use hidden layers when the relationship between inputs and outputs is nonlinear and a linear model measurably underperforms.

### 6.9.3 What a hidden layer adds, in one sentence

A hidden layer lets the network build its **own features** from the inputs, so the final neuron faces an easier, often linearly separable problem. Chapter 7 develops this fully.

---

## 6.10 Advantages, Limitations, and Applications

### 6.10.1 Advantages

- Very simple, fast, and cheap to train and evaluate.
- Weights are interpretable: their sign and size show each feature's influence.
- The perceptron rule has a convergence guarantee on separable data.
- It is the unit from which all larger networks are built.

### 6.10.2 Limitations

- Cannot represent non-linearly-separable problems (XOR, circles).
- The step perceptron gives no confidence information and cannot be trained with gradients.
- On non-separable data the perceptron rule never converges.
- Sensitive to the order of examples and to feature scaling (Chapter 5's standardisation helps).

### 6.10.3 Applications

| Use | Notes |
|---|---|
| Linear classification baselines (spam filtering, simple text classification) | Linear models on good features are often strong |
| Historical pattern recognition (early character recognition) | The original use |
| Logistic-regression output layers | The final layer of many classifiers is exactly a sigmoid or softmax neuron layer |
| Building blocks | Every neuron in a deep network is this unit with a different activation |

---

## 6.11 Common Misconceptions and Errors

| Misconception or mistake | Correction |
|---|---|
| "A neuron is a miniature brain cell." | It is a weighted sum plus an activation function. The biological link is only an inspiration |
| "The bias is optional." | Without a bias the boundary must pass through the origin, which limits what the neuron can represent |
| "A perceptron and logistic regression are the same." | They share the linear form $\mathbf{w}\cdot\mathbf{x}+b$ but differ in activation, output, and training |
| "If I train longer or with a smaller learning rate, a single neuron will solve XOR." | The limitation is structural (Section 6.7.2); training cannot overcome it |
| "A hidden layer alone is enough." | Without a **nonlinear** activation, stacked layers collapse to one linear map (Section 3.5.3). The nonlinearity is essential |
| "The decision boundary of a sigmoid neuron is curved." | It is the straight line $z = 0$; only the transition across it is smooth |
| Passing a sigmoid output into `F.binary_cross_entropy_with_logits` | That function expects the raw logit and applies the sigmoid itself |
| "Zero initial weights are fine for all networks." | Fine for a single perceptron, but problematic for hidden layers (Chapter 10) |

---

## 6.12 Comparison Summary

| Concept | Formula or rule | Role |
|---|---|---|
| Artificial neuron | $a = \varphi(\mathbf{w}\cdot\mathbf{x} + b)$ | The basic computational unit |
| Weights | $\mathbf{w}$ | Importance and direction of each input |
| Bias | $b$ | Shifts the decision boundary; baseline tendency |
| Step activation | $1$ if $z \ge 0$ else $0$ | Hard decision; not trainable by gradients |
| Sigmoid activation | $1/(1 + e^{-z})$ | Smooth probability-like output |
| Perceptron rule | $\mathbf{w} \leftarrow \mathbf{w} + \eta(y - \hat{y})\mathbf{x}$ | Mistake-driven learning for threshold neurons |
| Linear separability | A single hyperplane separates the classes | What a single neuron can solve |
| Hidden layer | Maps inputs to a new representation | Lets networks solve XOR and beyond |

---

## Exercises

1. **Neuron by hand.** For $\mathbf{x} = [2, -1, 3]$, $\mathbf{w} = [0.5, 1, -0.5]$, $b = 1$: compute $z$, the step output, and the sigmoid output $\sigma(z)$.
2. **Design a gate.** Choose weights and a bias for a perceptron that implements NOT (single input: output 1 when $x = 0$, output 0 when $x = 1$). Verify both inputs.
3. **Boundary.** For $\mathbf{w} = [2, 1]$ and $b = -4$, write the decision boundary equation, find where it crosses each axis, and state which side outputs 1.
4. **Trace the rule.** Using $\eta = 1$, zero initial weights, and the order $(0,0), (0,1), (1,0), (1,1)$, trace the perceptron rule by hand for AND (targets $0,0,0,1$) until an epoch has no errors. Verify with `train_perceptron`.
5. **Learning rate.** Run `train_perceptron` on OR with `lr=0.1` and `lr=10`. Does the number of epochs change? Explain why (hint: the update is proportional to `lr`, and the data is separable).
6. **Proof practice.** Repeat the XOR impossibility argument for XNOR (outputs $1,0,0,1$ for the four inputs in order) and show it is also impossible for one perceptron.
7. **Two layers.** Build XNOR (the negation of XOR) with the same two-layer structure by changing only the output neuron's weights and bias. Verify all four rows.
8. **Loss values.** For a sigmoid neuron with $z = -2$ and true label $y = 1$, compute $a$, the binary cross-entropy loss, and the gradient $a - y$. Check with `F.binary_cross_entropy_with_logits` and autograd.
9. **Parameter counting.** How many parameters does a layer of 16 neurons reading 784 inputs have? How many if 10 such neurons follow it?
10. **Reflection.** In your own words, explain why a smooth activation function is necessary for training with gradient descent but is not enough on its own to solve XOR.
