# Chapter 4: Math II: Calculus and Gradients

**Level 1: Essential Foundations | Tag: Essential**

## Chapter Roadmap

In Chapter 1 we found good parameters by brute-force search and saw that it cannot scale. The replacement is **calculus**: at the current parameter values, it tells us which way to nudge every parameter to make the error smaller.

By the end of this chapter you will be able to:

- Explain what a derivative measures and compute simple derivatives by hand.
- Check a derivative numerically with a finite difference.
- Explain partial derivatives and the gradient, and why the negative gradient points downhill.
- Apply the chain rule to composite functions, including the step-by-step pattern used by backpropagation.
- Take a complete "learning step" by hand: forward value, loss, gradients, parameter update.
- Use PyTorch autograd to compute gradients and recognise its most common pitfall.

**Scope.** Only the calculus needed to understand training is covered. Integrals, series, and second-order methods are not needed here. The full backpropagation algorithm is built in Chapter 8; this chapter prepares every tool it uses.

> **Honesty note.** The code in this chapter has not been executed by me. All "Expected output" blocks were derived by hand. Run each block and compare.

---

## 4.1 Why Calculus?

Training a network means finding parameters $\theta$ that make a **loss** (a number measuring how wrong the predictions are) small. A network can have millions of parameters, so we need an efficient answer to this question:

> *If I change this parameter by a tiny amount, how much does the loss change, and in which direction?*

That quantity is a **derivative**. Collected over all parameters, the derivatives form the **gradient**. Computing it efficiently through many layers is done with the **chain rule**. These three ideas, derivative, gradient, and chain rule, are the whole mathematical engine of Deep Learning.

| Question | Calculus tool |
|---|---|
| How sensitive is the output to one input? | Derivative |
| How sensitive is the loss to each of many parameters? | Partial derivatives, collected into the gradient |
| How do I get sensitivities through a stack of layers? | Chain rule |
| Which way should I move the parameters? | Opposite to the gradient |

---

## 4.2 Derivatives

### 4.2.1 Formal definition

The **derivative** of a function $f$ at the point $x$ is:

$$f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

| Symbol | Meaning |
|---|---|
| $f(x)$ | The function being studied |
| $h$ | A small change in the input |
| $f(x+h) - f(x)$ | How much the output changed |
| $\dfrac{f(x+h) - f(x)}{h}$ | Change in output per unit change in input (the average slope over a tiny interval) |
| $\lim_{h \to 0}$ | Let $h$ shrink toward zero, so the average slope becomes the exact slope at $x$ |
| $f'(x)$ or $\dfrac{df}{dx}$ | Two notations for the same thing: the derivative |

### 4.2.2 Simple explanation

The derivative is the **slope** of the function at a point, or equivalently its **rate of change**.

**Analogy.** If $f(t)$ is the distance a car has travelled at time $t$, then $f'(t)$ is its speed: the speedometer reading at that instant.

**What the sign tells you:**

| Value of $f'(x)$ | Meaning at that point |
|---|---|
| Positive | The function is increasing: raising $x$ raises $f$ |
| Negative | The function is decreasing: raising $x$ lowers $f$ |
| Zero | The function is momentarily flat (a possible minimum, maximum, or saddle) |
| Large in magnitude | $f$ is very sensitive to $x$ here |

### 4.2.3 Why it exists and what problem it solves

Our goal is to *lower* the loss. If the derivative of the loss with respect to a parameter is positive, **decreasing** that parameter lowers the loss. If it is negative, **increasing** it lowers the loss. The derivative tells us the direction and, roughly, how strongly.

### 4.2.4 Worked example: derivative of $x^2$ from the definition

Let $f(x) = x^2$. Then:

$$\frac{f(x+h) - f(x)}{h} = \frac{(x+h)^2 - x^2}{h} = \frac{x^2 + 2xh + h^2 - x^2}{h} = \frac{2xh + h^2}{h} = 2x + h$$

As $h \to 0$, this becomes $2x$. So $f'(x) = 2x$.

At $x = 3$ the slope is $6$. Concretely, a small step of $h = 0.01$ should raise $f$ by about $6 \times 0.01 = 0.06$:

- $f(3) = 9$ and $f(3.01) = 9.0601$, a rise of $0.0601$, very close to $0.06$.

This leads to the key approximation used throughout training:

$$f(x + h) \approx f(x) + f'(x)\, h \qquad \text{(for small } h\text{)}$$

**The function looks like a straight line near any point, and the derivative is that line's slope.**

### 4.2.5 Rules you need

| Function | Derivative | Example |
|---|---|---|
| Constant $c$ | $0$ | $\frac{d}{dx}(5) = 0$ |
| $x^n$ | $n\,x^{n-1}$ | $\frac{d}{dx}(x^3) = 3x^2$ |
| $c \cdot f(x)$ | $c \cdot f'(x)$ | $\frac{d}{dx}(4x^2) = 8x$ |
| $f(x) + g(x)$ | $f'(x) + g'(x)$ | $\frac{d}{dx}(x^2 + 3x) = 2x + 3$ |
| $e^x$ | $e^x$ | |
| $\ln x$ | $\dfrac{1}{x}$ | |
| Product $f(x)\,g(x)$ | $f'(x)\,g(x) + f(x)\,g'(x)$ | $\frac{d}{dx}(x \cdot x^2) = 1 \cdot x^2 + x \cdot 2x = 3x^2$ |

The chain rule, the most important rule for Deep Learning, gets its own section (4.5).

**Worked example.** For $f(x) = 3x^2 + 2x + 1$: $f'(x) = 6x + 2$, so $f'(2) = 14$.

### 4.2.6 Checking derivatives numerically

We can approximate the derivative using the definition with a small, fixed $h$. The **central difference** version is more accurate than the one-sided version:

$$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}$$

```python
def numerical_derivative(func, x, h=1e-5):
    return (func(x + h) - func(x - h)) / (2 * h)

def square(x):
    return x ** 2

print(round(numerical_derivative(square, 3.0), 4))
print(round(numerical_derivative(lambda x: 3 * x**2 + 2 * x + 1, 2.0), 4))
print(round(numerical_derivative(lambda x: x ** 3, 2.0), 4))
```

| Code | Explanation |
|---|---|
| `def numerical_derivative(func, x, h=1e-5)` | Takes a function as an argument. `h=1e-5` means $0.00001$; it is a default value you can override |
| `func(x + h)` and `func(x - h)` | Evaluates the function slightly to the right and left of $x$ |
| `/ (2 * h)` | Divides the difference by the width of the interval, $2h$ |
| `lambda x: ...` | A short, unnamed function written inline |
| `round(..., 4)` | Rounds to 4 decimal places to hide tiny floating-point noise |

**Hand-derived results:** $2x = 6$ at $x=3$; $6x + 2 = 14$ at $x=2$; $3x^2 = 12$ at $x=2$.

**Expected output:**

```text
6.0
14.0
12.0
```

**Why not use this for training?** Each derivative needs two extra evaluations of the function. A network with a million parameters would need two million extra forward passes *per step*. Also, a very small $h$ causes floating-point round-off errors. Numerical differentiation is used only to **check** other methods (Section 4.8).

---

## 4.3 Using the Derivative to Minimise a Function

### 4.3.1 The idea

If the slope is positive, move left; if negative, move right. A single formula does both:

$$x_{\text{new}} = x - \eta \, f'(x)$$

| Symbol | Meaning |
|---|---|
| $x$ | The current value |
| $f'(x)$ | The slope at $x$ |
| $\eta$ (eta) | The **learning rate**: a small positive number controlling the step size |
| $x_{\text{new}}$ | The updated value |

This is **gradient descent** in one dimension. Chapter 8 develops it fully; here we just see how derivatives drive it.

### 4.3.2 Worked example

Minimise $f(x) = (x - 3)^2$. Its derivative is $f'(x) = 2(x - 3)$, and the minimum is clearly at $x = 3$. Start at $x = 0$ with $\eta = 0.1$:

- Step 1: $f'(0) = 2(0 - 3) = -6$, so $x = 0 - 0.1(-6) = 0.6$.
- Step 2: $f'(0.6) = 2(0.6 - 3) = -4.8$, so $x = 0.6 - 0.1(-4.8) = 1.08$.
- Step 3: $f'(1.08) = -3.84$, so $x = 1.08 + 0.384 = 1.464$.

The update simplifies: $x_{\text{new}} - 3 = (1 - 2\eta)(x - 3)$. With $\eta = 0.1$, the distance to the minimum is multiplied by $0.8$ at every step.

```python
x = 0.0
learning_rate = 0.1

for step in range(11):
    loss = (x - 3.0) ** 2
    print(f"step {step:2d}: x = {x:.4f}, loss = {loss:.4f}")
    gradient = 2.0 * (x - 3.0)
    x = x - learning_rate * gradient
```

| Code | Explanation |
|---|---|
| `x = 0.0` | Starting guess |
| `loss = (x - 3.0) ** 2` | The function we want to minimise |
| `gradient = 2.0 * (x - 3.0)` | The derivative worked out by hand |
| `x = x - learning_rate * gradient` | The update rule |
| `{step:2d}`, `{x:.4f}` | Format codes: integer padded to width 2; float with 4 decimals |

**Expected output (derived from $x_t = 3 - 3(0.8)^t$ and loss $= 9(0.64)^t$):**

```text
step  0: x = 0.0000, loss = 9.0000
step  1: x = 0.6000, loss = 5.7600
step  2: x = 1.0800, loss = 3.6864
step  3: x = 1.4640, loss = 2.3593
step  4: x = 1.7712, loss = 1.5099
step  5: x = 2.0170, loss = 0.9664
step  6: x = 2.2136, loss = 0.6185
step  7: x = 2.3709, loss = 0.3958
step  8: x = 2.4967, loss = 0.2533
step  9: x = 2.5973, loss = 0.1621
step 10: x = 2.6779, loss = 0.1038
```

### 4.3.3 The learning rate matters

Because the distance to the minimum is multiplied by $(1 - 2\eta)$ each step for this particular function:

| $\eta$ | Factor $(1 - 2\eta)$ | Behaviour |
|---|---|---|
| 0.1 | 0.8 | Smooth, steady approach (above) |
| 0.5 | 0 | Lands exactly on the minimum in one step |
| 1.0 | $-1$ | Bounces between two values forever, never settling |
| 1.1 | $-1.2$ | The distance grows by 20% each step: **divergence** |
| 0.01 | 0.98 | Moves in the right direction but very slowly |

The exact numbers belong to this simple function, but the lesson is general: **too small a learning rate is slow, too large a rate can overshoot or diverge**. Chapter 8 returns to this.

### 4.3.4 Common misconceptions

| Misconception | Correction |
|---|---|
| "A zero derivative means we found the minimum." | It means the function is flat there. It could be a minimum, a maximum, or a saddle point. |
| "A bigger step is always better." | Overshooting can make the loss rise or diverge (see the table above). |
| "The derivative tells us where the minimum is." | It tells us only the **local** slope. Descent follows it step by step; it does not jump to the answer. |

---

## 4.4 Partial Derivatives and the Gradient

### 4.4.1 Functions of many variables

A loss depends on *many* parameters, not one. Think of $L(\theta_1, \theta_2, \dots, \theta_n)$. We need the sensitivity to each parameter separately.

### 4.4.2 Partial derivatives

**Definition.** The **partial derivative** of $f$ with respect to $x_i$ is the ordinary derivative taken while treating all other variables as constants. It is written $\dfrac{\partial f}{\partial x_i}$. The curly $\partial$ (as opposed to $d$) signals that other variables exist and are being held fixed.

**Worked example.** Let

$$f(x, y) = x^2 + xy + y^2$$

- With respect to $x$ (treat $y$ as a constant): $\dfrac{\partial f}{\partial x} = 2x + y$.
  The term $x^2$ gives $2x$; the term $xy$ gives $y$ (as $y$ is a constant multiplier); the term $y^2$ is a constant and gives $0$.
- With respect to $y$ (treat $x$ as a constant): $\dfrac{\partial f}{\partial y} = x + 2y$.

At the point $(x, y) = (1, 2)$:

$$\frac{\partial f}{\partial x} = 2(1) + 2 = 4, \qquad \frac{\partial f}{\partial y} = 1 + 2(2) = 5$$

**Interpretation:** at $(1, 2)$, increasing $x$ slightly raises $f$ at a rate of 4 per unit, and increasing $y$ slightly raises $f$ at a rate of 5 per unit.

### 4.4.3 The gradient

**Definition.** The **gradient** of $f$ is the vector of all its partial derivatives:

$$\nabla f = \begin{bmatrix} \dfrac{\partial f}{\partial x_1} \\[2mm] \dfrac{\partial f}{\partial x_2} \\ \vdots \\ \dfrac{\partial f}{\partial x_n} \end{bmatrix}$$

| Symbol | Meaning |
|---|---|
| $\nabla$ | "nabla" or "del": the gradient operator |
| $\nabla f$ | A vector with one entry per input variable |
| $\dfrac{\partial f}{\partial x_i}$ | How sensitive $f$ is to variable $x_i$ alone |

For our example at $(1, 2)$: $\nabla f = [4,\ 5]$.

**Geometric meaning.** Picture $f$ as a landscape, where the height at each position is the function value.

- The gradient points in the direction of **steepest ascent** (uphill).
- The **negative** gradient, $-\nabla f$, points in the direction of **steepest descent** (downhill).
- The length of the gradient measures **how steep** the slope is.

**Why the negative gradient reduces the function.** For a small change $\Delta$ in the inputs, the multi-variable version of the approximation from Section 4.2.4 is:

$$f(\mathbf{x} + \boldsymbol{\Delta}) \approx f(\mathbf{x}) + \nabla f \cdot \boldsymbol{\Delta}$$

This uses the dot product from Chapter 3. If we choose $\boldsymbol{\Delta} = -\eta \nabla f$:

$$f(\mathbf{x} - \eta \nabla f) \approx f(\mathbf{x}) - \eta\, \lVert \nabla f \rVert_2^2$$

Since $\lVert \nabla f \rVert_2^2 \ge 0$, the function decreases (for small $\eta$).

### 4.4.4 The gradient descent update

For parameters $\boldsymbol{\theta}$ and loss $L$:

$$\boldsymbol{\theta}_{\text{new}} = \boldsymbol{\theta} - \eta\, \nabla_{\boldsymbol{\theta}} L$$

This is the same rule as Section 4.3.1, applied to every parameter simultaneously. **This single line is how neural networks learn.** All the later sophistication (momentum, Adam, schedules) in Chapter 11 modifies how the step is computed, not this basic idea.

### 4.4.5 Worked example: one descent step

For $f(x, y) = x^2 + xy + y^2$ at $(1, 2)$: $f = 1 + 2 + 4 = 7$ and $\nabla f = [4, 5]$. Take $\eta = 0.01$:

$$(x, y)_{\text{new}} = (1 - 0.01 \cdot 4,\ 2 - 0.01 \cdot 5) = (0.96,\ 1.95)$$

$$f(0.96, 1.95) = 0.9216 + 1.872 + 3.8025 = 6.5961$$

The first-order prediction is $7 - 0.01(4^2 + 5^2) = 7 - 0.41 = 6.59$. The true value, $6.5961$, is slightly higher because the surface curves; the approximation holds for small steps. The function decreased, as promised.

### 4.4.6 Practical Python

First a numerical gradient check, then PyTorch's automatic computation:

```python
import numpy as np
import torch

def f_np(p):
    return p[0] ** 2 + p[0] * p[1] + p[1] ** 2

def numerical_gradient(func, point, h=1e-5):
    point = np.array(point, dtype=float)
    grad = np.zeros_like(point)
    for i in range(len(point)):
        step = np.zeros_like(point)
        step[i] = h
        grad[i] = (func(point + step) - func(point - step)) / (2 * h)
    return grad

print(np.round(numerical_gradient(f_np, [1.0, 2.0]), 4))

# The same gradient using PyTorch autograd
p = torch.tensor([1.0, 2.0], requires_grad=True)
loss = p[0] ** 2 + p[0] * p[1] + p[1] ** 2
loss.backward()
print(p.grad)

# One gradient descent step
learning_rate = 0.01
with torch.no_grad():
    p -= learning_rate * p.grad

print(p.detach())
new_loss = p[0] ** 2 + p[0] * p[1] + p[1] ** 2
print(round(new_loss.item(), 4))
```

| Code | Explanation |
|---|---|
| `np.zeros_like(point)` | An array of zeros with the same shape as `point` |
| `step[i] = h` | A vector that is zero everywhere except position $i$, so only variable $i$ is nudged |
| `(func(point + step) - func(point - step)) / (2 * h)` | The central difference for one partial derivative |
| `requires_grad=True` | Tells PyTorch to track operations on `p` |
| `loss.backward()` | Computes $\partial\,\text{loss}/\partial p$ for every tracked element and stores it |
| `p.grad` | The gradient, with the same shape as `p` |
| `with torch.no_grad():` | Do not record the following operations for differentiation. Parameter updates should not themselves be differentiated |
| `p -= learning_rate * p.grad` | The in-place update $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\nabla L$ |
| `p.detach()` | Prints the values without gradient bookkeeping |
| `.item()` | Converts a one-element tensor to a Python number |

**Expected output:**

```text
[4. 5.]
tensor([4., 5.])
tensor([0.9600, 1.9500])
6.5961
```

---

## 4.5 The Chain Rule

### 4.5.1 Formal definition

If $y = f(u)$ and $u = g(x)$, so $y = f(g(x))$, then:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

| Symbol | Meaning |
|---|---|
| $x$ | The original input |
| $u = g(x)$ | An intermediate quantity computed from $x$ |
| $y = f(u)$ | The final output, computed from $u$ |
| $\dfrac{du}{dx}$ | How fast $u$ changes when $x$ changes |
| $\dfrac{dy}{du}$ | How fast $y$ changes when $u$ changes |
| Product | The overall sensitivity is the product of the sensitivities along the chain |

### 4.5.2 Intuition

**Analogy: gears.** Suppose turning the first gear once turns the second gear 3 times, and turning the second gear once turns the third gear 2 times. Then turning the first gear once turns the third gear $3 \times 2 = 6$ times. Rates of change **multiply** along a chain.

**Why it exists.** A neural network is a chain of functions (Chapter 1: $f_3(f_2(f_1(x)))$). The loss depends on early parameters only *through* the later layers. The chain rule is what lets us pass sensitivity information backward through that chain.

### 4.5.3 Worked example 1

Let $y = (3x + 1)^2$. Define $u = 3x + 1$, so $y = u^2$.

- $\dfrac{dy}{du} = 2u$
- $\dfrac{du}{dx} = 3$
- $\dfrac{dy}{dx} = 2u \cdot 3 = 6u = 6(3x + 1)$

At $x = 1$: $u = 4$, so $\dfrac{dy}{dx} = 6 \cdot 4 = 24$.

**Numerical sanity check.** $y(1) = 16$ and $y(1.001) = (4.003)^2 = 16.024009$. The change per unit input is $0.024009 / 0.001 = 24.009$, close to $24$.

```python
def y_of_x(x):
    return (3 * x + 1) ** 2

analytic = 6 * (3 * 1.0 + 1)
numerical = numerical_derivative(y_of_x, 1.0)

print(analytic)
print(round(numerical, 4))
```

This reuses `numerical_derivative` from Section 4.2.6.

**Expected output:**

```text
24.0
24.0
```

### 4.5.4 Longer chains

With three steps $y = f_3(u_2),\ u_2 = f_2(u_1),\ u_1 = f_1(x)$:

$$\frac{dy}{dx} = \frac{dy}{du_2} \cdot \frac{du_2}{du_1} \cdot \frac{du_1}{dx}$$

Each factor is a **local** derivative: it involves only one step and is easy to compute. A network with 100 layers is a chain of 100 steps, and the chain rule multiplies 100 local derivatives.

**Preview of a problem.** If many of those local derivatives are small, the product becomes tiny; if many are large, it explodes. For example, the sigmoid function (Chapter 7) has a derivative never larger than $0.25$, so the product of ten such factors is at most $0.25^{10} \approx 9.5 \times 10^{-7}$. This is the **vanishing gradient problem** (Chapter 10), and it follows directly from the chain rule.

### 4.5.5 Multiple paths: add the contributions

If a variable influences the output through **more than one route**, the chain rule is applied along each route and the results are **added**.

**Worked example.** Let $L = u \cdot v$ with $u = x^2$ and $v = x + 1$. Then $x$ reaches $L$ through two paths (through $u$ and through $v$):

$$\frac{dL}{dx} = \frac{\partial L}{\partial u}\frac{du}{dx} + \frac{\partial L}{\partial v}\frac{dv}{dx} = v \cdot 2x + u \cdot 1 = (x+1)(2x) + x^2 = 3x^2 + 2x$$

Check by direct expansion: $L = x^2(x + 1) = x^3 + x^2$, so $\dfrac{dL}{dx} = 3x^2 + 2x$. Same result. At $x = 2$: $3(4) + 4 = 16$.

This "add over paths" rule matters because in real networks one value (such as an input feature or a shared weight) often feeds several later computations. Backpropagation (Chapter 8) accumulates gradients exactly this way.

---

## 4.6 A Complete Learning Step by Hand

Now we combine everything on the smallest possible model: one input, one weight, one bias.

### 4.6.1 Setup

$$\hat{y} = w x + b, \qquad L = (\hat{y} - y)^2$$

| Symbol | Meaning |
|---|---|
| $x$ | Input |
| $w, b$ | The weight and bias: the parameters we want to learn |
| $\hat{y}$ | Prediction |
| $y$ | True target |
| $L$ | Loss: squared error for this one example |

Take $x = 2$, $y = 7$, starting parameters $w = 1$, $b = 0$, and $\eta = 0.05$.

### 4.6.2 Forward pass (compute values left to right)

- $\hat{y} = 1 \cdot 2 + 0 = 2$
- $L = (2 - 7)^2 = 25$

### 4.6.3 Backward pass (apply the chain rule right to left)

The dependency chain is $(w, b) \to \hat{y} \to L$.

- Local derivative of the loss: $\dfrac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y) = 2(2 - 7) = -10$.
- Local derivatives of the prediction: $\dfrac{\partial \hat{y}}{\partial w} = x = 2$ and $\dfrac{\partial \hat{y}}{\partial b} = 1$.
- Chain rule:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w} = (-10)(2) = -20, \qquad \frac{\partial L}{\partial b} = (-10)(1) = -10$$

Both gradients are negative: increasing $w$ or $b$ lowers the loss, which makes sense, since the prediction ($2$) is below the target ($7$).

### 4.6.4 Update

$$w \leftarrow 1 - 0.05(-20) = 2, \qquad b \leftarrow 0 - 0.05(-10) = 0.5$$

### 4.6.5 Check that the loss fell

New prediction: $\hat{y} = 2 \cdot 2 + 0.5 = 4.5$. New loss: $(4.5 - 7)^2 = 6.25$, down from $25$.

**One step of gradient descent took the loss from 25 to 6.25. Repeating it is training.**

### 4.6.6 In plain Python, then repeated

```python
x, y_true = 2.0, 7.0
w, b = 1.0, 0.0
learning_rate = 0.05

for step in range(6):
    # Forward pass
    y_hat = w * x + b
    loss = (y_hat - y_true) ** 2
    print(f"step {step}: y_hat = {y_hat:.4f}, loss = {loss:.4f}")

    # Backward pass (chain rule)
    dL_dyhat = 2.0 * (y_hat - y_true)
    dL_dw = dL_dyhat * x
    dL_db = dL_dyhat * 1.0

    # Update
    w = w - learning_rate * dL_dw
    b = b - learning_rate * dL_db
```

| Code | Explanation |
|---|---|
| `y_hat = w * x + b` | Forward pass: the model's prediction |
| `loss = (y_hat - y_true) ** 2` | Squared error |
| `dL_dyhat` | $\partial L / \partial \hat{y}$, the first local derivative |
| `dL_dw = dL_dyhat * x` | Chain rule: multiply by $\partial\hat{y}/\partial w = x$ |
| `dL_db = dL_dyhat * 1.0` | Chain rule: multiply by $\partial\hat{y}/\partial b = 1$ |
| `w = w - learning_rate * dL_dw` | Update both parameters from the gradients computed *before* either changed |

**Expected output (derived by hand):** in this example the prediction error is halved at every step, so the loss falls by a factor of four.

```text
step 0: y_hat = 2.0000, loss = 25.0000
step 1: y_hat = 4.5000, loss = 6.2500
step 2: y_hat = 5.7500, loss = 1.5625
step 3: y_hat = 6.3750, loss = 0.3906
step 4: y_hat = 6.6875, loss = 0.0977
step 5: y_hat = 6.8438, loss = 0.0244
```

(The halving is a feature of these particular numbers, since $1 - \eta \cdot 2 \cdot (x^2 + 1) = 1 - 0.05 \cdot 2 \cdot 5 = 0.5$. It is not a general law.)

---

## 4.7 Automatic Differentiation with PyTorch

### 4.7.1 Definition and motivation

**Automatic differentiation (autograd)** records the elementary operations performed during the forward pass, then applies the chain rule through them in reverse to deliver exact derivatives. You write only the forward computation; PyTorch produces the gradients. Chapter 8 explains the mechanism as a computational graph.

### 4.7.2 Checking our hand calculation

```python
import torch

x = torch.tensor(2.0)
y_true = torch.tensor(7.0)

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

y_hat = w * x + b
loss = (y_hat - y_true) ** 2
loss.backward()

print(w.grad)
print(b.grad)
```

| Code | Explanation |
|---|---|
| `x`, `y_true` | Plain data. They do not need gradients |
| `requires_grad=True` on `w` and `b` | These are the learnable parameters, so PyTorch tracks them |
| `loss.backward()` | Runs the chain rule backward from `loss` to every tracked tensor |
| `w.grad`, `b.grad` | $\partial L / \partial w$ and $\partial L / \partial b$ |

**Expected output (matches Section 4.6.3):**

```text
tensor(-20.)
tensor(-10.)
```

### 4.7.3 The shape rule for gradients

**A gradient always has the same shape as the parameter it belongs to**, because it holds one partial derivative per element.

```python
torch.manual_seed(0)

X = torch.randn(4, 3)                       # batch of 4 samples, 3 features
Y = torch.randn(4, 2)                       # 4 targets with 2 values each
W = torch.randn(3, 2, requires_grad=True)   # weights
b = torch.zeros(2, requires_grad=True)      # biases

loss = ((X @ W + b - Y) ** 2).mean()
loss.backward()

print(W.grad.shape)
print(b.grad.shape)
```

| Code | Explanation |
|---|---|
| `X @ W + b` | The layer from Chapter 3: shape `(4, 3) @ (3, 2)` gives `(4, 2)`, with `b` broadcast |
| `... - Y` | Prediction minus target, shape `(4, 2)` |
| `** 2` and `.mean()` | Mean squared error over all 8 entries: a single number (a scalar), which `backward()` requires |

**Expected output:**

```text
torch.Size([3, 2])
torch.Size([2])
```

### 4.7.4 Pitfall: gradients accumulate

Each call to `backward()` **adds** to `.grad` rather than replacing it. If you forget to reset gradients between training steps, they pile up and training breaks.

```python
x = torch.tensor(3.0, requires_grad=True)

y = x ** 2
y.backward()
print(x.grad)

y = x ** 2
y.backward()
print(x.grad)

x.grad.zero_()
print(x.grad)
```

| Code | Explanation |
|---|---|
| First `backward()` | $dy/dx = 2x = 6$, stored in `x.grad` |
| Second `backward()` | Another $6$ is **added**, giving $12$ |
| `x.grad.zero_()` | Resets the stored gradient to zero in place (the trailing underscore means in-place) |

**Expected output:**

```text
tensor(6.)
tensor(12.)
tensor(0.)
```

In Chapter 9's training loop, clearing gradients every iteration is a required step.

### 4.7.5 Other autograd details to know now

| Fact | Consequence |
|---|---|
| `backward()` on a non-scalar tensor needs extra arguments | Reduce the output to a scalar loss first (for example with `.mean()` or `.sum()`) |
| Operations inside `torch.no_grad()` are not tracked | Use it for parameter updates and for evaluation (inference) to save memory |
| `tensor.detach()` returns a tensor with no gradient history | Use it before converting to NumPy or when you want to stop gradients flowing |
| Only floating-point tensors can require gradients | Integer tensors cannot be differentiated |

---

## 4.8 Comparing Ways to Compute Derivatives

| Method | How it works | Accuracy | Cost for $n$ parameters | Used for |
|---|---|---|---|---|
| **Symbolic** | Derive a formula with algebra (by hand or with a computer algebra system) | Exact | Depends on formula | Understanding, simple models |
| **Numerical** (finite differences) | Evaluate $f$ at nearby points | Approximate; sensitive to $h$ and round-off | About $2n$ extra function evaluations | **Checking** other methods |
| **Automatic** (autograd) | Apply the chain rule to the recorded elementary operations | Exact up to floating-point precision | A small multiple of one forward pass, for all $n$ gradients together | **Training** |

**Gradient checking** is a valuable debugging technique: compare `autograd` against `numerical_gradient` on a small input. They should agree closely (to several decimal places). A mismatch usually means a bug in a custom operation or a hand-written backward pass.

---

## 4.9 Derivatives You Will Meet Later

You do not need to memorise these now; they are listed so the chain rule has concrete targets in Chapter 7. Each is derived there.

| Function | Formula | Derivative | Notes |
|---|---|---|---|
| Sigmoid $\sigma(x)$ | $\dfrac{1}{1 + e^{-x}}$ | $\sigma(x)\,(1 - \sigma(x))$ | Maximum value $0.25$ at $x = 0$ |
| Tanh | $\tanh(x)$ | $1 - \tanh^2(x)$ | Maximum value $1$ at $x = 0$ |
| ReLU | $\max(0, x)$ | $1$ if $x > 0$, else $0$ | Not differentiable exactly at $0$; PyTorch uses $0$ there |
| Squared error | $(\hat{y} - y)^2$ | $2(\hat{y} - y)$ with respect to $\hat{y}$ | Used in Section 4.6 |

A quick autograd check of the sigmoid value at $x=0$ (sigmoid $= 0.5$, so the derivative is $0.5 \times 0.5 = 0.25$):

```python
x0 = torch.tensor(0.0, requires_grad=True)
torch.sigmoid(x0).backward()
print(x0.grad)
```

**Expected output:**

```text
tensor(0.2500)
```

---

## 4.10 Limitations and Common Misconceptions

### 4.10.1 Limitations of the gradient

- **Local information only.** The gradient describes the slope at the current point. It does not reveal the global shape of the loss surface.
- **Flat regions.** Where gradients are zero or tiny (plateaus, saddle points, saturated activations), progress stalls.
- **Not guaranteed to find the best solution.** Neural-network losses are not simple bowls. Training finds parameters that work well in practice, but this is an empirical observation, not a guarantee. Later chapters on optimisers and initialisation address the practical difficulties.

### 4.10.2 Misconceptions and errors

| Misconception or mistake | Correction |
|---|---|
| "The gradient is a number." | For many parameters it is a vector (or tensor) with the same shape as the parameters |
| "The gradient points toward the minimum." | It points toward steepest local **ascent**. We step in the **opposite** direction, and that leads downhill locally, not necessarily straight to the minimum |
| "$\partial$ and $d$ mean the same thing." | $d$ is for functions of one variable; $\partial$ means other variables are held fixed |
| Forgetting the minus sign in the update | Using $\theta + \eta\nabla L$ climbs the loss instead of descending it |
| Not clearing gradients between steps | Gradients accumulate in PyTorch (Section 4.7.4) |
| Calling `backward()` on a non-scalar | Reduce to a scalar loss first |
| "Autograd means I don't need to understand calculus." | You still need it to read errors, design losses, and diagnose vanishing or exploding gradients |

---

## 4.11 Real-World Role of These Ideas

| Where it appears | How calculus is used |
|---|---|
| Every neural-network training run | Gradient of the loss with respect to all parameters, then an update step |
| Fine-tuning a pretrained model | The same gradient descent, starting from pretrained weights (Chapters 14 and 18) |
| Debugging "training doesn't improve" | Inspecting gradient sizes for vanishing or exploding values (Chapter 10) |
| Custom loss functions | Writing the forward formula; autograd supplies the derivative |
| Adversarial examples | Using the gradient with respect to the **input** instead of the weights (Chapter 1's limitations) |

---

## Exercises

1. **Derivatives.** Find $f'(x)$ for (a) $f(x) = 5x^3 - 2x + 7$, (b) $f(x) = x^2 e^x$ (use the product rule), (c) $f(x) = (2x - 5)^4$ (use the chain rule). Check each at $x = 1$ with `numerical_derivative`.
2. **Direction.** For $f(x) = (x-3)^2$, which direction should $x$ move when $x = 5$? When $x = 1$? Confirm using the sign of $f'(x)$.
3. **Learning rate.** Modify the loop in Section 4.3.2 to use $\eta = 1.1$ and then $\eta = 0.5$. Predict the first five values of $x$ by hand before running.
4. **Partial derivatives.** For $f(x, y) = 3x^2 y + y^3$, find $\partial f/\partial x$ and $\partial f/\partial y$, evaluate the gradient at $(1, 2)$, and check it with `numerical_gradient`.
5. **Descent step.** Using the gradient from Exercise 4, take one step with $\eta = 0.01$. Compute $f$ before and after, and compare with the first-order prediction $f - \eta\lVert\nabla f\rVert^2$.
6. **Chain rule.** Differentiate $y = \sqrt{x^2 + 1}$ by writing $u = x^2 + 1$ and $y = u^{1/2}$. Evaluate at $x = 2$.
7. **A learning step.** Redo Section 4.6 by hand with $x = 3$, $y = 10$, $w = 1$, $b = 1$, $\eta = 0.02$. Give $\hat{y}$, $L$, both gradients, the updated parameters, and the new loss. Verify with autograd.
8. **Accumulation.** Write a short PyTorch snippet showing that calling `backward()` twice without zeroing gradients doubles `.grad`, then fix it.
9. **Vanishing product.** Compute $0.25^{n}$ for $n = 5, 10, 20$ and describe what this implies for the gradient reaching the first layer of a 20-layer network whose local derivatives are all $0.25$.
