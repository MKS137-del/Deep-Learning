# Chapter 3: Math I: Linear Algebra

**Level 1: Essential Foundations | Tag: Essential**

## Chapter Roadmap

Chapter 2 showed you how to *handle* tensors in code. This chapter explains what the main operations *mean*, so that when you later read `X @ W + b` you understand what is being computed and why the shapes must line up.

By the end of this chapter you will be able to:

- Distinguish scalars, vectors, matrices, and tensors, and read their notation.
- Compute and interpret the dot product, vector length (norm), and cosine similarity.
- Multiply matrices by hand and predict the result shape.
- Explain how a neural-network layer is a matrix multiplication plus a bias.
- Count the parameters of a layer and track shapes through a multi-layer network.
- Explain why stacking linear layers without anything in between gains nothing (the motivation for Chapter 7).

**Scope.** This chapter teaches only the linear algebra that later chapters use. Topics such as determinants, matrix inverses, eigenvalues, and singular value decomposition are deliberately skipped. If a later chapter needs one (for example the idea of matrix rank when we reach LoRA in Chapter 23), it will be introduced there.

> **Honesty note.** The code in this chapter has not been executed by me. All "Expected output" blocks were derived by hand. Run each block and compare.

---

## 3.1 Why Linear Algebra?

### 3.1.1 The core idea

Neural networks operate on numbers, and almost everything they do to those numbers is one of two things:

1. **Multiply by a matrix and add a vector** (this chapter).
2. **Apply a simple nonlinear function to each number** (Chapter 7).

A network with a hundred layers is, at heart, these two steps repeated. Linear algebra is the language of the first step.

### 3.1.2 Where linear algebra appears

| Deep Learning concept | Linear algebra object |
|---|---|
| One input example (a house, a patient, a pixel row) | Vector |
| A whole dataset or a batch of examples | Matrix |
| Weights of a layer | Matrix |
| Biases of a layer | Vector |
| A batch of colour images | 4-D tensor |
| One neuron's computation | Dot product |
| One whole layer's computation | Matrix multiplication |
| "How similar are two word embeddings?" | Dot product / cosine similarity (Chapter 15) |
| Attention scores in Transformers | Dot products between vectors (Chapter 17) |

---

## 3.2 Scalars, Vectors, Matrices, and Tensors

### 3.2.1 Formal definitions

- A **scalar** is a single number. Notation: $x \in \mathbb{R}$ (the symbol $\mathbb{R}$ means "the set of real numbers" and $\in$ means "is an element of").
- A **vector** is an ordered list of $n$ numbers. Notation: $\mathbf{x} \in \mathbb{R}^{n}$ (a list of $n$ real numbers). Its $i$-th element is written $x_i$.
- A **matrix** is a rectangular grid of numbers with $m$ rows and $n$ columns. Notation: $\mathbf{A} \in \mathbb{R}^{m \times n}$. The element in row $i$, column $j$ is written $A_{ij}$.
- A **tensor** is the general case: a multi-dimensional grid of numbers. Scalars, vectors, and matrices are tensors of rank 0, 1, and 2.

**Notation used in this book:** lowercase italic for scalars ($x$), bold lowercase for vectors ($\mathbf{x}$), bold uppercase for matrices ($\mathbf{A}$, $\mathbf{W}$, $\mathbf{X}$).

### 3.2.2 Simple explanation

| Object | Everyday picture | Example |
|---|---|---|
| Scalar | One number | Temperature: $37.5$ |
| Vector | A list (one row of a spreadsheet) | One house: $[120,\ 3,\ 10]$ (area in square metres, bedrooms, age in years) |
| Matrix | A table (the whole spreadsheet) | Three houses stacked as rows |
| Tensor | A stack of tables | A batch of colour images |

**A vector as a point or arrow.** The vector $[3, 2]$ can be pictured as an arrow from the origin to the point 3 units right and 2 units up. A vector with 120 numbers is a point in a 120-dimensional space. We cannot draw it, but the mathematics works identically, and this "data as points in space" picture underlies much of machine learning.

### 3.2.3 Data as vectors and matrices

Suppose each house is described by three features: area (square metres), bedrooms, and age (years).

| House | Area | Bedrooms | Age |
|---|---|---|---|
| 1 | 120 | 3 | 10 |
| 2 | 80 | 2 | 25 |
| 3 | 200 | 5 | 2 |

- Each **row** is a vector in $\mathbb{R}^{3}$: one sample.
- The whole table is a matrix $\mathbf{X} \in \mathbb{R}^{3 \times 3}$ with $N = 3$ samples (rows) and $F = 3$ features (columns).

**Convention used throughout this book:** a dataset or batch is stored as $\mathbf{X} \in \mathbb{R}^{N \times F}$, with **one sample per row**. This matches PyTorch.

### 3.2.4 Shapes and rank in code

```python
import numpy as np

s = np.array(5.0)                       # scalar
v = np.array([120.0, 3.0, 10.0])        # vector
X = np.array([[120.0, 3.0, 10.0],
              [ 80.0, 2.0, 25.0],
              [200.0, 5.0,  2.0]])      # matrix
T = np.zeros((2, 3, 4))                 # 3-D tensor

for name, arr in [("s", s), ("v", v), ("X", X), ("T", T)]:
    print(name, "ndim =", arr.ndim, " shape =", arr.shape)
```

| Code | Explanation |
|---|---|
| `np.array(5.0)` | A zero-dimensional array holding one number |
| `arr.ndim` | The rank: how many dimensions |
| `arr.shape` | Size of each dimension; the empty tuple `()` means a scalar |
| `for name, arr in [...]` | Loops over (label, array) pairs so we can print them uniformly |

**Expected output:**

```text
s ndim = 0  shape = ()
v ndim = 1  shape = (3,)
X ndim = 2  shape = (3, 3)
T ndim = 3  shape = (2, 3, 4)
```

### 3.2.5 Row vectors vs column vectors

Mathematically, a vector can be written as a column (shape $n \times 1$) or a row (shape $1 \times n$). In NumPy and PyTorch a plain 1-D array of shape `(n,)` is neither. This matters for matrix multiplication.

```python
v = np.array([1.0, 2.0, 3.0])
print(v.shape)                      # (3,)
print(v.reshape(3, 1).shape)        # (3, 1)  column vector
print(v.reshape(1, 3).shape)        # (1, 3)  row vector
```

**Expected output:**

```text
(3,)
(3, 1)
(1, 3)
```

### 3.2.6 Common misconceptions

| Misconception | Correction |
|---|---|
| "A tensor in Deep Learning is the same object as a tensor in physics." | In Deep Learning, "tensor" simply means a multi-dimensional array. The physics meaning involves transformation rules that we never use. |
| "A vector of shape `(3,)` is a column vector." | It is a 1-D array. It has no row or column orientation until you reshape it or use it in a matrix product. |
| "More dimensions means harder mathematics." | A 4-D image batch is just several matrices organised together. Operations act on specific dimensions. |

---

## 3.3 Vector Operations

### 3.3.1 Addition and scalar multiplication

For vectors of the same length, **addition** is element by element, and **scalar multiplication** scales every element:

$$(\mathbf{a} + \mathbf{b})_i = a_i + b_i, \qquad (c\,\mathbf{a})_i = c \cdot a_i$$

| Symbol | Meaning |
|---|---|
| $\mathbf{a}, \mathbf{b}$ | Two vectors of the same length |
| $a_i, b_i$ | Their $i$-th elements |
| $c$ | A scalar |

Geometrically, adding vectors places one arrow's tail at the other's tip; multiplying by $c$ stretches the arrow by a factor of $c$ (and flips its direction if $c < 0$).

### 3.3.2 Element-wise (Hadamard) product

The **element-wise product** multiplies matching positions and returns a vector of the same length:

$$(\mathbf{a} \odot \mathbf{b})_i = a_i \, b_i$$

The symbol $\odot$ denotes element-wise multiplication. In NumPy and PyTorch this is the `*` operator. **It is not the dot product.**

### 3.3.3 The dot product

**Formal definition.** For two vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^{n}$ of equal length:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i \, b_i = a_1 b_1 + a_2 b_2 + \dots + a_n b_n$$

| Symbol | Meaning |
|---|---|
| $\mathbf{a} \cdot \mathbf{b}$ | The dot product. The result is a **single number** (a scalar) |
| $n$ | Length of both vectors; they must be equal |
| $a_i, b_i$ | $i$-th elements |
| $\sum_{i=1}^{n}$ | Add the terms for $i = 1$ up to $i = n$ |

**Simple explanation.** Multiply matching elements, then add everything up.

**Why it exists.** It condenses two lists of numbers into one number that measures how strongly they "agree": large and positive when large values line up with large values, negative when they oppose, near zero when unrelated.

**Intuition: a weighted checklist.** Suppose a neuron decides whether a house is "good value". The vector $\mathbf{w}$ holds how much each feature matters (its weights), and $\mathbf{x}$ holds the house's features. The dot product $\mathbf{w} \cdot \mathbf{x}$ is the total weighted score.

**A worked example.** Let $\mathbf{w} = [0.5,\ -1,\ 2]$ and $\mathbf{x} = [4,\ 3,\ 1]$:

$$\mathbf{w} \cdot \mathbf{x} = (0.5)(4) + (-1)(3) + (2)(1) = 2 - 3 + 2 = 1$$

Adding a bias $b = 0.5$ gives the neuron's pre-activation value:

$$z = \mathbf{w} \cdot \mathbf{x} + b = 1 + 0.5 = 1.5$$

**This single formula, $z = \mathbf{w} \cdot \mathbf{x} + b$, is the computation of one artificial neuron** (Chapter 6).

### 3.3.4 Vector length (norm)

The **L2 norm** (Euclidean length) of a vector measures how long the arrow is:

$$\lVert \mathbf{a} \rVert_2 = \sqrt{\sum_{i=1}^{n} a_i^2} = \sqrt{\mathbf{a} \cdot \mathbf{a}}$$

The **L1 norm** adds absolute values instead:

$$\lVert \mathbf{a} \rVert_1 = \sum_{i=1}^{n} |a_i|$$

| Symbol | Meaning |
|---|---|
| $\lVert \mathbf{a} \rVert_2$ | L2 norm: straight-line length |
| $\lVert \mathbf{a} \rVert_1$ | L1 norm: sum of absolute values |
| $\lvert a_i \rvert$ | Absolute value of $a_i$ |

Norms appear in regularisation (Chapter 10), where large weights are penalised, and in normalising vectors to unit length.

### 3.3.5 Geometric meaning of the dot product and cosine similarity

The dot product also equals:

$$\mathbf{a} \cdot \mathbf{b} = \lVert \mathbf{a} \rVert_2 \, \lVert \mathbf{b} \rVert_2 \cos\theta$$

where $\theta$ is the angle between the two arrows. Rearranging gives **cosine similarity**, which compares *direction* regardless of length:

$$\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert_2 \, \lVert \mathbf{b} \rVert_2}$$

| Value of $\cos\theta$ | Meaning |
|---|---|
| $+1$ | Same direction ($\theta = 0^\circ$) |
| $0$ | Perpendicular, unrelated ($\theta = 90^\circ$) |
| $-1$ | Opposite directions ($\theta = 180^\circ$) |

Embedding models (Chapter 15) represent words as vectors so that related words point in similar directions, and attention (Chapter 17) uses dot products to score how relevant one position is to another.

### 3.3.6 Practical Python

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

print(a + b)                       # element-wise addition
print(2 * a)                       # scalar multiplication
print(a * b)                       # element-wise (Hadamard) product
print(np.dot(a, b))                # dot product
print(np.linalg.norm(a))           # L2 norm
print(np.linalg.norm(a, ord=1))    # L1 norm
```

| Code | Explanation |
|---|---|
| `a + b` | Adds matching elements: $[1+4,\ 2+5,\ 3+6]$ |
| `2 * a` | Multiplies each element by 2 |
| `a * b` | Element-wise product: $[4,\ 10,\ 18]$ |
| `np.dot(a, b)` | $1(4) + 2(5) + 3(6) = 4 + 10 + 18 = 32$ |
| `np.linalg.norm(a)` | $\sqrt{1 + 4 + 9} = \sqrt{14} \approx 3.7417$ |
| `ord=1` | Selects the L1 norm: $1 + 2 + 3 = 6$ |

**Expected output** (decimal places may differ slightly in the last digits depending on your NumPy version):

```text
[5. 7. 9.]
[2. 4. 6.]
[ 4. 10. 18.]
32.0
3.7416573867739413
6.0
```

Now a single neuron and cosine similarity:

```python
w = np.array([0.5, -1.0, 2.0])
x = np.array([4.0, 3.0, 1.0])
bias = 0.5

z = np.dot(w, x) + bias
print(z)

def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

print(round(cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 1.0])), 4))
print(round(cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 4))
print(round(cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0])), 4))
```

| Code | Explanation |
|---|---|
| `np.dot(w, x) + bias` | The neuron computation from the worked example: $1 + 0.5$ |
| `cosine_similarity` | Implements $\frac{\mathbf{u}\cdot\mathbf{v}}{\lVert\mathbf{u}\rVert\lVert\mathbf{v}\rVert}$ directly |
| First call | $[1,0]$ vs $[1,1]$: dot $= 1$, lengths $1$ and $\sqrt{2}$, so $1/\sqrt{2} \approx 0.7071$ (an angle of $45^\circ$) |
| Second call | Perpendicular vectors: dot $= 0$ |
| Third call | Opposite directions: dot $= -1$, lengths 1 and 1 |
| `round(..., 4)` | Rounds to 4 decimal places for readable output |

**Expected output:**

```text
1.5
0.7071
0.0
-1.0
```

### 3.3.7 Comparison: three kinds of "multiplication"

| Operation | NumPy / PyTorch | Inputs | Output | Example with $\mathbf{a} = [1,2,3]$, $\mathbf{b} = [4,5,6]$ |
|---|---|---|---|---|
| Element-wise product | `a * b` | Same shape (or broadcastable) | Same shape as inputs | $[4, 10, 18]$ |
| Dot product | `a @ b` or `np.dot(a, b)` | Two 1-D vectors, equal length | A scalar | $32$ |
| Matrix multiplication | `A @ B` | Matrices with compatible inner dimension | A matrix | Section 3.4 |

Mixing up `*` and `@` is among the most common beginner errors. Remember: `*` keeps the shape; `@` combines and **sums over a shared dimension**.

---

## 3.4 Matrices and Matrix Multiplication

### 3.4.1 Transpose

The **transpose** of a matrix flips rows and columns. If $\mathbf{A} \in \mathbb{R}^{m \times n}$, then $\mathbf{A}^{\top} \in \mathbb{R}^{n \times m}$ and:

$$(\mathbf{A}^{\top})_{ij} = A_{ji}$$

For example,

$$\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \quad\Rightarrow\quad \mathbf{A}^{\top} = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$$

The shape `(2, 3)` becomes `(3, 2)`. In code: `A.T`.

### 3.4.2 Matrix-vector product

Multiplying a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ by a vector $\mathbf{x} \in \mathbb{R}^{n}$ gives a vector in $\mathbb{R}^{m}$. **Each output element is the dot product of one row of $\mathbf{A}$ with $\mathbf{x}$.**

$$\mathbf{A}\mathbf{x} = \begin{bmatrix} \text{row}_1 \cdot \mathbf{x} \\ \text{row}_2 \cdot \mathbf{x} \\ \vdots \\ \text{row}_m \cdot \mathbf{x} \end{bmatrix}$$

Example with $\mathbf{x} = [1, 0, -1]$:

$$\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix} = \begin{bmatrix} 1(1) + 2(0) + 3(-1) \\ 4(1) + 5(0) + 6(-1) \end{bmatrix} = \begin{bmatrix} -2 \\ -2 \end{bmatrix}$$

**Interpretation:** each row of $\mathbf{A}$ is one neuron's weight vector. The matrix-vector product runs **all neurons of a layer on one input at once**.

### 3.4.3 Matrix multiplication

**Formal definition.** For $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{B} \in \mathbb{R}^{n \times p}$, the product $\mathbf{C} = \mathbf{A}\mathbf{B}$ is in $\mathbb{R}^{m \times p}$, with entries:

$$C_{ij} = \sum_{k=1}^{n} A_{ik} \, B_{kj}$$

| Symbol | Meaning |
|---|---|
| $m$ | Number of rows of $\mathbf{A}$ (and of $\mathbf{C}$) |
| $n$ | Columns of $\mathbf{A}$ = rows of $\mathbf{B}$. **This inner dimension must match** and is summed over |
| $p$ | Number of columns of $\mathbf{B}$ (and of $\mathbf{C}$) |
| $i, j$ | Row index and column index of the output entry |
| $k$ | The running index in the sum, from 1 to $n$ |
| $A_{ik}$ | Element in row $i$, column $k$ of $\mathbf{A}$ |
| $B_{kj}$ | Element in row $k$, column $j$ of $\mathbf{B}$ |

**In words:** entry $C_{ij}$ is the **dot product of row $i$ of $\mathbf{A}$ with column $j$ of $\mathbf{B}$**.

**Shape rule:**

$$(m \times n)\ \cdot\ (n \times p)\ \longrightarrow\ (m \times p)$$

The two inner numbers must be equal and they disappear; the outer numbers form the result.

### 3.4.4 Step-by-step worked example

Let

$$\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \ (2 \times 3), \qquad \mathbf{B} = \begin{bmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{bmatrix} \ (3 \times 2)$$

Inner dimensions: 3 and 3 match. Result shape: $2 \times 2$.

| Entry | Row of A | Column of B | Computation | Value |
|---|---|---|---|---|
| $C_{11}$ | $[1, 2, 3]$ | $[7, 9, 11]$ | $1(7) + 2(9) + 3(11) = 7 + 18 + 33$ | $58$ |
| $C_{12}$ | $[1, 2, 3]$ | $[8, 10, 12]$ | $1(8) + 2(10) + 3(12) = 8 + 20 + 36$ | $64$ |
| $C_{21}$ | $[4, 5, 6]$ | $[7, 9, 11]$ | $4(7) + 5(9) + 6(11) = 28 + 45 + 66$ | $139$ |
| $C_{22}$ | $[4, 5, 6]$ | $[8, 10, 12]$ | $4(8) + 5(10) + 6(12) = 32 + 50 + 72$ | $154$ |

$$\mathbf{C} = \begin{bmatrix} 58 & 64 \\ 139 & 154 \end{bmatrix}$$

**Cost.** Each of the $m \times p$ output entries needs $n$ multiplications, so the product needs $m \cdot n \cdot p = 2 \cdot 3 \cdot 2 = 12$ multiplications (and about as many additions). Networks do this for enormous matrices, which is why matrix multiplication speed (and therefore GPUs, Chapter 2) matters so much.

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[ 7,  8],
              [ 9, 10],
              [11, 12]])

C = A @ B
print(C)
print(C.shape)

print(A.T)
print(np.allclose((A @ B).T, B.T @ A.T))
```

| Code | Explanation |
|---|---|
| `A @ B` | Matrix multiplication, shape `(2, 3) @ (3, 2)` gives `(2, 2)` |
| `A.T` | Transpose, shape `(3, 2)` |
| `(A @ B).T` and `B.T @ A.T` | Tests the identity $(\mathbf{A}\mathbf{B})^{\top} = \mathbf{B}^{\top}\mathbf{A}^{\top}$ (note the reversed order) |
| `np.allclose` | `True` if two arrays are equal within floating-point tolerance |

**Expected output:**

```text
[[ 58  64]
 [139 154]]
(2, 2)
[[1 4]
 [2 5]
 [3 6]]
True
```

### 3.4.5 Properties you should know

| Property | Statement | Notes |
|---|---|---|
| **Not commutative** | Usually $\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}$ | Even when both shapes allow it. Often one order is not even defined |
| **Associative** | $(\mathbf{A}\mathbf{B})\mathbf{C} = \mathbf{A}(\mathbf{B}\mathbf{C})$ | Used in Section 3.5.3 |
| **Distributive** | $\mathbf{A}(\mathbf{B} + \mathbf{C}) = \mathbf{A}\mathbf{B} + \mathbf{A}\mathbf{C}$ | |
| **Transpose of a product** | $(\mathbf{A}\mathbf{B})^{\top} = \mathbf{B}^{\top}\mathbf{A}^{\top}$ | Order reverses. Used in backpropagation (Chapter 8) |
| **Identity** | $\mathbf{A}\mathbf{I} = \mathbf{I}\mathbf{A} = \mathbf{A}$ | $\mathbf{I}$ has ones on the diagonal, zeros elsewhere |

**Numerical demonstration that order matters.** Let

$$\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$$

$$\mathbf{A}\mathbf{B} = \begin{bmatrix} 2 & 1 \\ 4 & 3 \end{bmatrix} \qquad \mathbf{B}\mathbf{A} = \begin{bmatrix} 3 & 4 \\ 1 & 2 \end{bmatrix}$$

Multiplying by $\mathbf{B}$ on the right swaps the *columns* of $\mathbf{A}$; on the left it swaps the *rows*. The results differ.

---

## 3.5 Matrices as Transformations

### 3.5.1 Intuition

A matrix is not just a table; it is a **function** that takes a vector in and gives a vector out: $\mathbf{x} \mapsto \mathbf{A}\mathbf{x}$. In two dimensions you can see what it does.

- **Rotation by $90^\circ$ counter-clockwise:** $\mathbf{R} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$. Applied to $[1, 0]$ (pointing right) it gives $[0, 1]$ (pointing up).
- **Scaling:** $\mathbf{S} = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$ stretches the first coordinate by 2 and the second by 3.

```python
R = np.array([[0.0, -1.0],
              [1.0,  0.0]])
S = np.array([[2.0, 0.0],
              [0.0, 3.0]])

print(R @ np.array([1.0, 0.0]))
print(R @ np.array([0.0, 1.0]))
print(S @ np.array([1.0, 1.0]))
```

**Hand calculation:** $\mathbf{R}[1,0]^{\top} = [0(1) + (-1)(0),\ 1(1) + 0(0)] = [0, 1]$; $\mathbf{R}[0,1]^{\top} = [-1, 0]$; $\mathbf{S}[1,1]^{\top} = [2, 3]$.

**Expected output:**

```text
[0. 1.]
[-1.  0.]
[2. 3.]
```

**Why this matters.** The weight matrix of a layer transforms its input into a new representation: it can rotate, stretch, squash, and mix features. Training chooses a transformation that makes the final task easy. This is the geometric content of "learning representations" from Chapter 1.

### 3.5.2 Linear vs affine

A map of the form $\mathbf{x} \mapsto \mathbf{W}\mathbf{x}$ is **linear**. Adding a bias, $\mathbf{x} \mapsto \mathbf{W}\mathbf{x} + \mathbf{b}$, makes it **affine** (a linear map followed by a shift). Neural-network "linear layers" are affine in this precise sense; the name is used loosely.

### 3.5.3 Why stacking linear layers is not enough

Apply two linear layers in a row, with weight matrices $\mathbf{W}_1$ and $\mathbf{W}_2$:

$$\mathbf{W}_2(\mathbf{W}_1\mathbf{x}) = (\mathbf{W}_2\mathbf{W}_1)\,\mathbf{x}$$

by associativity. The two layers collapse into **one** layer with weight matrix $\mathbf{W}_2\mathbf{W}_1$. Biases do not rescue this: $\mathbf{W}_2(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (\mathbf{W}_2\mathbf{W}_1)\mathbf{x} + (\mathbf{W}_2\mathbf{b}_1 + \mathbf{b}_2)$, which is still a single affine map.

**Numerical check.** Take

$$\mathbf{W}_1 = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad \mathbf{W}_2 = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

- Step by step: $\mathbf{W}_1\mathbf{x} = [3, 7]$, then $\mathbf{W}_2[3, 7]^{\top} = [7, 3]$.
- Combined: $\mathbf{W}_2\mathbf{W}_1 = \begin{bmatrix} 3 & 4 \\ 1 & 2 \end{bmatrix}$ (this is the product $\mathbf{B}\mathbf{A}$ computed in Section 3.4.5), and $\begin{bmatrix} 3 & 4 \\ 1 & 2 \end{bmatrix}[1, 1]^{\top} = [7, 3]$.

Same answer.

```python
W1 = np.array([[1.0, 2.0],
               [3.0, 4.0]])
W2 = np.array([[0.0, 1.0],
               [1.0, 0.0]])
x = np.array([1.0, 1.0])

two_layers = W2 @ (W1 @ x)
one_layer = (W2 @ W1) @ x

print(two_layers)
print(one_layer)
print(W2 @ W1)
```

**Expected output:**

```text
[7. 3.]
[7. 3.]
[[3. 4.]
 [1. 2.]]
```

**Consequence.** A network built only from linear layers can only represent a single linear map, no matter how deep. To gain real power, we insert a **nonlinear activation function** between layers. That is the subject of Chapter 7.

---

## 3.6 Shapes in Neural-Network Layers

### 3.6.1 One neuron

For one input vector $\mathbf{x} \in \mathbb{R}^{d_{\text{in}}}$ and a weight vector $\mathbf{w} \in \mathbb{R}^{d_{\text{in}}}$:

$$z = \mathbf{w} \cdot \mathbf{x} + b$$

with scalar bias $b$ and scalar output $z$.

### 3.6.2 One layer of neurons, one sample

A layer with $d_{\text{out}}$ neurons stacks their weight vectors into a matrix. Using the batch-friendly row convention of this book:

$$\mathbf{z} = \mathbf{x}\mathbf{W} + \mathbf{b}$$

| Symbol | Shape | Meaning |
|---|---|---|
| $\mathbf{x}$ | $(1, d_{\text{in}})$ or $(d_{\text{in}},)$ | One input sample |
| $\mathbf{W}$ | $(d_{\text{in}}, d_{\text{out}})$ | Weights: **column $j$ holds neuron $j$'s weights** |
| $\mathbf{b}$ | $(d_{\text{out}},)$ | One bias per neuron |
| $\mathbf{z}$ | $(1, d_{\text{out}})$ or $(d_{\text{out}},)$ | One output value per neuron |

### 3.6.3 One layer, a whole batch

Stack $N$ samples as rows of $\mathbf{X}$. The same formula processes the entire batch in a single matrix multiplication:

$$\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}$$

| Symbol | Shape | Meaning |
|---|---|---|
| $\mathbf{X}$ | $(N, d_{\text{in}})$ | Batch of inputs |
| $\mathbf{W}$ | $(d_{\text{in}}, d_{\text{out}})$ | Weights |
| $\mathbf{b}$ | $(d_{\text{out}},)$ | Biases, **broadcast** across all $N$ rows (Chapter 2) |
| $\mathbf{Z}$ | $(N, d_{\text{out}})$ | Output for every sample |

Element by element, for sample $s$ and neuron $j$:

$$Z_{sj} = \sum_{i=1}^{d_{\text{in}}} X_{si}\, W_{ij} + b_j$$

This is exactly the matrix-multiplication formula from Section 3.4.3 plus a bias.

### 3.6.4 Worked numerical example

Take $N = 2$ samples, $d_{\text{in}} = 3$ inputs, $d_{\text{out}} = 2$ neurons:

$$\mathbf{X} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}, \quad \mathbf{W} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 0.5 & -1 \end{bmatrix}$$

Shape check: $(2 \times 3)(3 \times 2) \to (2 \times 2)$, and $\mathbf{b}$ of shape $(2,)$ broadcasts across the two rows.

**Step 1: $\mathbf{X}\mathbf{W}$**

- Sample 1: $[1(1) + 2(0) + 3(1),\ 1(0) + 2(1) + 3(1)] = [4,\ 5]$
- Sample 2: $[4(1) + 5(0) + 6(1),\ 4(0) + 5(1) + 6(1)] = [10,\ 11]$

**Step 2: add $\mathbf{b}$ to each row**

- Sample 1: $[4 + 0.5,\ 5 - 1] = [4.5,\ 4]$
- Sample 2: $[10 + 0.5,\ 11 - 1] = [10.5,\ 10]$

$$\mathbf{Z} = \begin{bmatrix} 4.5 & 4 \\ 10.5 & 10 \end{bmatrix}$$

### 3.6.5 The same layer in PyTorch

```python
import torch
import torch.nn as nn

X = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])           # (2, 3)
W = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]])                # (3, 2)
b = torch.tensor([0.5, -1.0])                 # (2,)

# Manual computation
Z = X @ W + b
print(Z)

# The same thing with PyTorch's built-in layer
layer = nn.Linear(in_features=3, out_features=2)
print(layer.weight.shape)
print(layer.bias.shape)

with torch.no_grad():
    layer.weight.copy_(W.T)
    layer.bias.copy_(b)

out = layer(X)
print(out.detach())
print(torch.allclose(out.detach(), Z))
print(sum(p.numel() for p in layer.parameters()))
```

| Code | Explanation |
|---|---|
| `X @ W + b` | The formula $\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}$ written directly |
| `nn.Linear(in_features=3, out_features=2)` | PyTorch's ready-made layer. It creates a weight matrix and bias vector with **random** initial values |
| `layer.weight.shape` | `torch.Size([2, 3])`: PyTorch stores the weight as `(out_features, in_features)`, the **transpose** of our $\mathbf{W}$ |
| `layer.bias.shape` | `torch.Size([2])` |
| `with torch.no_grad():` | A block in which PyTorch does not record operations for gradients. We use it to set parameters by hand without tracking that step |
| `layer.weight.copy_(W.T)` | Copies values into the existing parameter **in place** (trailing underscore means in-place in PyTorch). `W.T` has shape `(2, 3)` |
| `layer(X)` | Computes $\mathbf{X}\,\text{weight}^{\top} + \text{bias}$, which equals $\mathbf{X}\mathbf{W} + \mathbf{b}$ |
| `out.detach()` | Returns the values without the gradient-tracking information, for clean printing |
| `torch.allclose(...)` | `True` if the two tensors match within tolerance |
| `p.numel()` summed | Counts every learnable number: $3 \times 2$ weights $+\ 2$ biases $= 8$ |

**Expected output:**

```text
tensor([[ 4.5000,  4.0000],
        [10.5000, 10.0000]])
torch.Size([2, 3])
torch.Size([2])
tensor([[ 4.5000,  4.0000],
        [10.5000, 10.0000]])
True
8
```

**Convention note (a frequent source of confusion).**

| Convention | Formula | Weight shape | Used by |
|---|---|---|---|
| Column-vector (many textbooks and papers) | $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$ | $(d_{\text{out}}, d_{\text{in}})$ | Mathematical derivations |
| Row-batch (this book's derivations) | $\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}$ | $(d_{\text{in}}, d_{\text{out}})$ | Our hand calculations and NumPy implementations |
| PyTorch `nn.Linear` | $\mathbf{Z} = \mathbf{X}\,\mathbf{W}_{\text{stored}}^{\top} + \mathbf{b}$ | $(d_{\text{out}}, d_{\text{in}})$ | Real PyTorch models |

The mathematics is identical; only the storage layout differs. When a chapter shows a from-scratch implementation, it will state which convention it uses.

### 3.6.6 Parameter counting

For a layer with $d_{\text{in}}$ inputs and $d_{\text{out}}$ outputs:

$$\text{parameters} = d_{\text{in}} \cdot d_{\text{out}} + d_{\text{out}}$$

(the weight matrix plus the bias vector).

| Layer | $d_{\text{in}}$ | $d_{\text{out}}$ | Weights | Biases | Total |
|---|---|---|---|---|---|
| Layer 1 | 3 | 4 | $3 \times 4 = 12$ | 4 | 16 |
| Layer 2 | 4 | 2 | $4 \times 2 = 8$ | 2 | 10 |
| **Network total** | | | | | **26** |

### 3.6.7 Tracking shapes through a multi-layer network

```python
torch.manual_seed(0)

layer1 = nn.Linear(3, 4)
layer2 = nn.Linear(4, 2)

x = torch.randn(5, 3)        # batch of 5 samples, 3 features each
h = layer1(x)                # hidden representation
y = layer2(h)                # final output

print(x.shape)
print(h.shape)
print(y.shape)

total = sum(p.numel() for m in (layer1, layer2) for p in m.parameters())
print(total)
```

| Code | Explanation |
|---|---|
| `torch.manual_seed(0)` | Fixes the random numbers (Chapter 2) so results are repeatable |
| `nn.Linear(3, 4)` | Positional arguments are `(in_features, out_features)` |
| `torch.randn(5, 3)` | Random input batch, 5 samples with 3 features each |
| `layer1(x)` | `(5, 3)` times `(3, 4)` gives `(5, 4)` |
| `layer2(h)` | `(5, 4)` times `(4, 2)` gives `(5, 2)` |
| The `sum(...)` line | Loops over both layers and all of their parameters, counting elements |

**Expected output** (shapes and the count do not depend on the random values):

```text
torch.Size([5, 3])
torch.Size([5, 4])
torch.Size([5, 2])
26
```

```text
Input        Layer 1         Hidden         Layer 2         Output
(5, 3)  --> (3 -> 4) -->    (5, 4)   --> (4 -> 2) -->     (5, 2)
```

**Note:** this network has no activation function between the layers, so by Section 3.5.3 it is equivalent to one linear layer. Chapter 7 inserts the missing nonlinearity.

### 3.6.8 Batched and higher-dimensional cases

Real data has more than two dimensions. The rules extend naturally:

- **Linear layers act on the last dimension.** For sequence data of shape $(N, T, D)$ (batch, time steps, features), `nn.Linear(D, H)` produces shape $(N, T, H)$, applying the same weights at every time step.
- **Batched matrix multiplication.** When the leading dimensions match, `@` multiplies each pair of matrices independently: $(B, m, n)\ @\ (B, n, p) \to (B, m, p)$. Transformers use this heavily (Chapter 17).

```python
seq = torch.randn(2, 5, 3)         # (N=2, T=5, D=3)
proj = nn.Linear(3, 4)
print(proj(seq).shape)

A3 = torch.randn(2, 3, 4)
B3 = torch.randn(2, 4, 5)
print((A3 @ B3).shape)
```

**Expected output:**

```text
torch.Size([2, 5, 4])
torch.Size([2, 3, 5])
```

### 3.6.9 Cost of a layer

A matrix product $(m \times n)(n \times p)$ needs $m \cdot n \cdot p$ multiply-add operations. For a batch of $N = 64$ samples through a layer from 784 inputs to 128 outputs:

$$64 \times 784 \times 128 = 6{,}422{,}528 \text{ multiply-adds}$$

for that one layer on that one batch. Deep networks stack many such layers and run thousands of batches, so matrix multiplication dominates the total computation. That is why hardware designed for fast matrix arithmetic changed the field.

---

## 3.7 Advantages, Limitations, and Applications

### 3.7.1 Advantages of the linear-algebra formulation

- **Compact:** one expression `X @ W + b` describes the computation of thousands of neurons on a whole batch.
- **Fast:** highly optimised libraries and GPUs accelerate matrix multiplication.
- **Analysable:** shapes can be checked mentally before running anything.

### 3.7.2 Limitations

- A purely linear model can only represent linear (affine) relationships (Section 3.5.3).
- The formulas say nothing about *how to choose* good weights. That requires calculus (Chapter 4) and an optimisation procedure (Chapter 8).

### 3.7.3 Applications beyond neural layers

| Application | Linear algebra idea |
|---|---|
| Recommender systems | A user-by-item rating matrix, factorised into smaller matrices |
| Images | Greyscale images are matrices; colour images are 3-D tensors |
| Word and token embeddings | Each word is a row of an embedding matrix (Chapter 15) |
| Similarity search | Cosine similarity between vectors |
| Attention | Matrix products of query and key matrices (Chapter 17) |

---

## 3.8 Common Errors and Misconceptions

| Mistake or misconception | Correction |
|---|---|
| Using `*` when you mean `@` | `*` is element-wise and keeps shapes; `@` is matrix multiplication. They give different results and often different shapes |
| "Matrix multiplication is commutative." | Generally $\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}$ (Section 3.4.5) |
| "The dot product returns a vector." | It returns a single number |
| Shape error `(a, b) @ (c, d)` with $b \neq c$ | Check the inner dimensions; transpose with `.T` or correct a layer size |
| Confusing PyTorch's weight shape with the textbook's | `nn.Linear.weight` has shape `(out_features, in_features)` (Section 3.6.5) |
| "A 1-D tensor is a row vector." | Shape `(n,)` has no orientation. Use `reshape` or `unsqueeze` when orientation matters |
| `torch.dot` on 2-D tensors | `torch.dot` accepts only 1-D tensors of the same length and type. Use `@` or `torch.matmul` for matrices |
| "Two stacked linear layers are more powerful than one." | They collapse to one linear map (Section 3.5.3). Nonlinearity is required |

---

## 3.9 Comparison Summary

| Object / operation | Notation | Shape rule | Output | Role in Deep Learning |
|---|---|---|---|---|
| Scalar | $x$ | `()` | | A loss value, a learning rate |
| Vector | $\mathbf{x}$ | `(n,)` | | One sample, one neuron's weights, one bias vector |
| Matrix | $\mathbf{W}$ | `(m, n)` | | Weights of a layer, a batch of samples |
| Tensor | | `(d1, d2, ...)` | | Image batches, sequence batches |
| Element-wise product | $\mathbf{a} \odot \mathbf{b}$ | Same (or broadcastable) | Same shape | Masks, gating (Chapter 16) |
| Dot product | $\mathbf{a} \cdot \mathbf{b}$ | `(n,)` and `(n,)` | Scalar | One neuron; similarity scores |
| Matrix multiplication | $\mathbf{A}\mathbf{B}$ | `(m,n)` and `(n,p)` | `(m,p)` | A whole layer on a whole batch |
| Transpose | $\mathbf{A}^{\top}$ | `(m,n)` becomes `(n,m)` | | Convention changes, backpropagation |
| Norm | $\lVert\mathbf{a}\rVert$ | `(n,)` | Scalar | Regularisation, normalisation |

---

## Exercises

1. **Dot product.** Compute $\mathbf{a} \cdot \mathbf{b}$ for $\mathbf{a} = [2, -1, 3]$ and $\mathbf{b} = [4, 0, -2]$. Then compute $\lVert\mathbf{a}\rVert_2$ and $\lVert\mathbf{a}\rVert_1$.
2. **Cosine similarity.** For $\mathbf{u} = [3, 4]$ and $\mathbf{v} = [4, 3]$, compute the cosine similarity by hand and check it in code.
3. **Matrix product.** Compute $\mathbf{A}\mathbf{B}$ by hand for $\mathbf{A} = \begin{bmatrix} 2 & 0 \\ 1 & 3 \end{bmatrix}$ and $\mathbf{B} = \begin{bmatrix} 1 & 4 \\ 2 & 5 \end{bmatrix}$, then compute $\mathbf{B}\mathbf{A}$. Are they equal?
4. **Shape checking.** For each product, state whether it is valid and give the result shape: (a) $(4, 3) @ (3, 5)$, (b) $(4, 3) @ (4, 5)$, (c) $(10, 784) @ (784, 128)$, (d) $(2, 7, 3) @ (2, 3, 6)$.
5. **Layer by hand.** For $\mathbf{x} = [1, 2]$, $\mathbf{W} = \begin{bmatrix} 1 & -1 & 0 \\ 2 & 1 & 1 \end{bmatrix}$, $\mathbf{b} = [0, 1, -1]$, compute $\mathbf{z} = \mathbf{x}\mathbf{W} + \mathbf{b}$ and state its shape. Verify with `nn.Linear` by copying in $\mathbf{W}^{\top}$ and $\mathbf{b}$.
6. **Parameter counting.** How many learnable parameters does a network with layers $784 \to 128 \to 10$ have? Verify by building it with two `nn.Linear` layers and summing `numel()`.
7. **Collapse.** Pick two $2 \times 2$ matrices and a vector of your own. Show numerically that applying them one after the other equals applying their product once.
8. **Cost.** How many multiply-adds does a batch of 32 samples need through a layer from 512 inputs to 256 outputs?
