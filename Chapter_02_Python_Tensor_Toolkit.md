# Chapter 2: The Python and Tensor Toolkit

**Level 1: Essential Foundations | Tag: Essential**

## Chapter Roadmap

Chapter 1 explained *what* Deep Learning is. Before we can build anything, you need the tools. This chapter covers the Python and numerical skills that every later chapter assumes.

By the end of this chapter you will be able to:

- Set up a working environment locally or on Google Colab.
- Use the Python features that neural-network code relies on (functions, classes, inheritance).
- Create, index, reshape, and combine NumPy arrays.
- Predict the result shape of an operation using the **broadcasting** rules.
- Create and manipulate PyTorch **tensors**, and convert between NumPy and PyTorch.
- Explain the difference between CPU and GPU execution and write device-independent code.
- Make experiments reproducible by controlling random seeds.
- Diagnose the most common shape, dtype, and device errors.

**Prerequisite gaps covered here (from the Chapter 1 syllabus):** Python classes and functions, NumPy broadcasting, tensor shapes and indexing.

> **Honesty note.** Code blocks in this chapter have not been executed by me. Every "Expected output" was derived by hand. Where output depends on your machine (versions, speed, GPU), the text says so. Run each block yourself and compare.

---

## 2.1 Setting Up Your Environment

### 2.1.1 Two ways to work

| Option | Best for | Strengths | Weaknesses |
|---|---|---|---|
| **Local machine** (Python + virtual environment) | Learning, small experiments, full control | Files persist, works offline, no usage limits imposed by a service | Needs installation; GPU use needs a compatible NVIDIA card and drivers |
| **Google Colab** (browser notebook) | Quick start, free GPU access when available | No installation, GPU option, easy sharing | Sessions are temporary, GPU availability and usage limits are not guaranteed |

All foundational exercises in this book run on a **CPU**. A GPU becomes useful from the CNN chapters onward. Colab setup is covered in Section 2.8.

### 2.1.2 Local installation

Use a **virtual environment** so that this project's libraries do not conflict with other Python projects.

```bash
# Create an isolated environment named dl-env
python -m venv dl-env

# Activate it (macOS / Linux)
source dl-env/bin/activate

# Activate it (Windows, Command Prompt)
# dl-env\Scripts\activate

# Install the libraries used in this book's early chapters
pip install numpy matplotlib torch torchvision
```

| Command | Explanation |
|---|---|
| `python -m venv dl-env` | Creates a folder `dl-env` holding a private copy of the Python environment |
| `source dl-env/bin/activate` | Makes your terminal use that private environment. Your prompt usually shows `(dl-env)` afterwards |
| `pip install ...` | Downloads and installs the listed packages into the active environment |

**Important:** the plain `pip install torch` command is fine for CPU use. If you have an NVIDIA GPU and want GPU support, the exact command depends on your operating system and CUDA version. Use the install selector on the official PyTorch website (pytorch.org) to get the correct command rather than guessing.

### 2.1.3 Verifying the installation

```python
import sys
import numpy as np
import torch

print("Python :", sys.version.split()[0])
print("NumPy  :", np.__version__)
print("PyTorch:", torch.__version__)
print("CUDA GPU available:", torch.cuda.is_available())
```

| Code | Explanation |
|---|---|
| `sys.version.split()[0]` | `sys.version` is a long string; splitting on spaces and taking the first piece gives just the version number |
| `np.__version__`, `torch.__version__` | Version strings of the libraries |
| `torch.cuda.is_available()` | `True` only if PyTorch can use an NVIDIA GPU |

The output depends on your machine. Record these versions when you report a bug or share results; many "it works on my computer" problems are version differences.

---

## 2.2 Python Essentials for Deep Learning

You do not need advanced Python. You do need to be comfortable with **functions**, **list comprehensions**, and especially **classes**, because every PyTorch model is written as a class.

### 2.2.1 Functions and list comprehensions

```python
def mean_squared_error(y_true, y_pred):
    squared_errors = [(t - p) ** 2 for t, p in zip(y_true, y_pred)]
    return sum(squared_errors) / len(squared_errors)

print(mean_squared_error([3.0, 5.0], [2.0, 7.0]))
```

| Code | Explanation |
|---|---|
| `def mean_squared_error(y_true, y_pred):` | Defines a function with two inputs: lists of true values and predictions |
| `zip(y_true, y_pred)` | Pairs up the two lists element by element: `(3.0, 2.0)`, `(5.0, 7.0)` |
| `[(t - p) ** 2 for t, p in ...]` | A **list comprehension**: builds a new list by applying `(t - p) ** 2` to each pair |
| `sum(...) / len(...)` | Total of the squared errors divided by how many there are: the average |
| `return` | Sends the result back to the caller |

**Hand calculation:** errors are $3-2 = 1$ and $5-7 = -2$; squares are $1$ and $4$; mean is $(1+4)/2 = 2.5$.

**Expected output:**

```text
2.5
```

This is the MSE from Chapter 1, written in plain Python. Later we replace such loops with array operations, which are shorter and much faster.

### 2.2.2 Classes: the structure behind every model

A **class** bundles data (parameters) and behaviour (functions) into one object. A class is a blueprint; an **object** (instance) is one concrete thing built from it.

```python
class Neuron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        return self.weight * x + self.bias

    def __call__(self, x):
        return self.forward(x)


neuron = Neuron(weight=2.0, bias=1.0)
print(neuron.forward(3.0))
print(neuron(3.0))
```

| Code | Explanation |
|---|---|
| `class Neuron:` | Defines the blueprint |
| `def __init__(self, weight, bias):` | The **constructor**. Python calls it automatically when you create a `Neuron`. `self` refers to the object being built |
| `self.weight = weight` | Stores the value *on the object* so other methods can use it later |
| `def forward(self, x):` | A **method**: a function that belongs to the class. It computes $wx + b$ |
| `def __call__(self, x):` | A special method that lets you call the object like a function: `neuron(3.0)` |
| `Neuron(weight=2.0, bias=1.0)` | Creates one object, running `__init__` |

**Hand calculation:** $2.0 \times 3.0 + 1.0 = 7.0$.

**Expected output:**

```text
7.0
7.0
```

**Why this matters.** In PyTorch, you will write a model class with an `__init__` that creates the layers and a `forward` method that describes the computation. PyTorch's base class supplies the `__call__` behaviour for you (Chapter 9).

### 2.2.3 Inheritance

A new class can **inherit** from an existing one, reusing and extending its behaviour. PyTorch models inherit from `torch.nn.Module`.

```python
class ScaledNeuron(Neuron):
    def __init__(self, weight, bias, scale):
        super().__init__(weight, bias)   # run the parent's constructor
        self.scale = scale

    def forward(self, x):
        return self.scale * super().forward(x)


scaled = ScaledNeuron(weight=2.0, bias=1.0, scale=10.0)
print(scaled(3.0))
```

| Code | Explanation |
|---|---|
| `class ScaledNeuron(Neuron):` | `ScaledNeuron` inherits everything from `Neuron` |
| `super().__init__(weight, bias)` | Calls the parent's constructor so `weight` and `bias` are set up. **Forgetting this call is a classic bug** (you will meet it again with `nn.Module`) |
| `super().forward(x)` | Calls the parent's `forward` and then multiplies the result by `scale` |
| `scaled(3.0)` | Uses the inherited `__call__`, which calls this class's overridden `forward` |

**Hand calculation:** parent gives $7.0$; times $10$ is $70.0$.

**Expected output:**

```text
70.0
```

---

## 2.3 NumPy Arrays

### 2.3.1 Definition and motivation

**NumPy** is Python's foundational library for numerical computing. Its core object is the **`ndarray`** (N-dimensional array): a grid of numbers that all share one data type, stored in a compact block of memory.

**Why not Python lists?** Lists can hold mixed types, so Python must check each element's type during every operation. Arrays enforce a single type and run their loops in fast compiled code. This is called **vectorization**: you write one operation and it is applied to every element.

| Aspect | Python list | NumPy array |
|---|---|---|
| Element types | Mixed allowed | One shared type |
| Math on whole collection | Needs explicit loop | `a * 2`, `a + b` work directly |
| Speed on large numeric data | Slow | Much faster |
| Multi-dimensional support | Awkward (lists of lists) | Built in |

### 2.3.2 Creating arrays and reading their properties

```python
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.shape)
print(a.ndim)
print(a.size)
print(a.dtype)
```

| Property | Meaning | Value here |
|---|---|---|
| `a.shape` | Size along each dimension, as a tuple | `(2, 3)`: 2 rows, 3 columns |
| `a.ndim` | Number of dimensions (also called axes) | `2` |
| `a.size` | Total number of elements | `6` |
| `a.dtype` | Data type of the elements | `int64` on most Linux/macOS systems (can be `int32` on some Windows setups) |

**Expected output (the dtype line depends on your system):**

```text
(2, 3)
2
6
int64
```

**Other common ways to create arrays:**

```python
print(np.zeros((2, 3)))       # all zeros, shape (2, 3)
print(np.ones((2, 2)))        # all ones, shape (2, 2)
print(np.arange(0, 10, 2))    # [0 2 4 6 8]: start, stop (excluded), step
print(np.linspace(0, 1, 5))   # 5 evenly spaced numbers from 0 to 1 inclusive
```

**Expected output:**

```text
[[0. 0. 0.]
 [0. 0. 0.]]
[[1. 1.]
 [1. 1.]]
[0 2 4 6 8]
[0.   0.25 0.5  0.75 1.  ]
```

Note that `np.zeros` and `np.ones` give floating-point numbers by default (the trailing dots, as in `0.`).

### 2.3.3 Indexing and slicing

Indexing starts at **0**. For a 2-D array, `a[row, column]`.

```python
print(a[0, 1])      # row 0, column 1
print(a[:, 1])      # every row, column 1
print(a[1, :])      # row 1, every column
print(a[:, :2])     # every row, first two columns
```

| Expression | Meaning | Result for the array above |
|---|---|---|
| `a[0, 1]` | Single element | `2` |
| `a[:, 1]` | `:` means "all"; the whole second column | `[2 5]` |
| `a[1, :]` | The whole second row | `[4 5 6]` |
| `a[:, :2]` | Columns from the start up to (not including) index 2 | `[[1 2]` and `[4 5]]` |

**Expected output:**

```text
2
[2 5]
[4 5 6]
[[1 2]
 [4 5]]
```

**Boolean masks** select elements by condition:

```python
print(a[a > 3])
```

`a > 3` produces a boolean array of the same shape; indexing with it keeps only the `True` positions, flattened into one dimension.

**Expected output:**

```text
[4 5 6]
```

### 2.3.4 Reshaping

`reshape` rearranges the same elements into a new shape. The total number of elements must stay the same. A single `-1` means "work this dimension out for me".

```python
print(a.reshape(3, 2))
print(a.reshape(-1))       # flatten to 1-D
```

**Expected output:**

```text
[[1 2]
 [3 4]
 [5 6]]
[1 2 3 4 5 6]
```

Elements are read in row-major order (left to right, then next row), then placed into the new shape in the same order.

### 2.3.5 Reductions and the `axis` argument

A **reduction** collapses an array by combining values (sum, mean, max). The `axis` argument says *which dimension disappears*.

```python
print(a.sum())           # everything
print(a.sum(axis=0))     # collapse rows: one result per column
print(a.sum(axis=1))     # collapse columns: one result per row
```

For the array `[[1 2 3], [4 5 6]]`:

- `axis=0` adds **down** the rows: $1+4,\ 2+5,\ 3+6$. Shape `(2, 3)` becomes `(3,)`.
- `axis=1` adds **across** the columns: $1+2+3,\ 4+5+6$. Shape `(2, 3)` becomes `(2,)`.

**Expected output:**

```text
21
[5 7 9]
[ 6 15]
```

**Memory aid:** the axis you name is the one that *vanishes* from the shape.

### 2.3.6 Vectorization in practice

```python
import time

values = np.arange(1_000_000, dtype=np.float64)

# Version 1: plain Python loop
start = time.perf_counter()
total_loop = 0.0
for v in values:
    total_loop += v * v
loop_time = time.perf_counter() - start

# Version 2: vectorized NumPy
start = time.perf_counter()
total_vec = np.sum(values ** 2)
vec_time = time.perf_counter() - start

print(np.isclose(total_loop, total_vec))
print(f"loop: {loop_time:.4f} s, vectorized: {vec_time:.4f} s")
```

| Code | Explanation |
|---|---|
| `np.arange(1_000_000, dtype=np.float64)` | Numbers 0 to 999,999 as 64-bit floats. Underscores in numbers are just for readability |
| `time.perf_counter()` | A high-resolution clock for measuring elapsed time |
| `for v in values:` | Python visits each element one at a time (slow) |
| `values ** 2` | Squares **all** elements in one compiled operation |
| `np.isclose(a, b)` | `True` if two floats are equal within a small tolerance. Use it instead of `==` for floating-point results |

**Expected output:** the first line should be `True`. The timings depend entirely on your machine, but the vectorized version is normally dramatically faster. **The principle to remember: avoid Python loops over data; use array operations.** Neural networks process millions of numbers, and this habit is what makes them feasible.

---

## 2.4 Broadcasting

### 2.4.1 Definition

**Broadcasting** is the set of rules that lets NumPy and PyTorch combine arrays of *different* shapes in one operation, by virtually stretching the smaller one without copying data.

### 2.4.2 Why it exists

Without broadcasting you would write loops or manually repeat data. Adding a bias vector to every row of a batch is a task you will do in nearly every neural network layer, and broadcasting does it in one expression.

### 2.4.3 The rules

To combine two arrays, compare their shapes **from the rightmost dimension to the left**:

1. If one array has fewer dimensions, imagine it padded with 1s on the **left**.
2. Two dimensions are **compatible** if they are equal, or if one of them is 1.
3. A dimension of size 1 is stretched to match the other.
4. If any pair is incompatible, the operation fails with an error.

The result shape takes the larger size in each position.

### 2.4.4 Worked example 1: a column plus a row

```python
col = np.array([[0], [10], [20]])    # shape (3, 1)
row = np.array([1, 2, 3, 4])         # shape (4,)

result = col + row
print(result.shape)
print(result)
```

Step-by-step shape comparison:

| Step | `col` | `row` |
|---|---|---|
| Original shapes | `(3, 1)` | `(4,)` |
| Pad `row` with a leading 1 | `(3, 1)` | `(1, 4)` |
| Compare last dimension | 1 vs 4: compatible, `col` stretches to 4 | |
| Compare first dimension | 3 vs 1: compatible, `row` stretches to 3 | |
| Result shape | `(3, 4)` | |

Each row of the result is the corresponding column value added to `[1, 2, 3, 4]`.

**Expected output:**

```text
(3, 4)
[[ 1  2  3  4]
 [11 12 13 14]
 [21 22 23 24]]
```

### 2.4.5 Worked example 2: a deep-learning use (centring data)

Subtract each feature's average from a dataset. Here `X` has 4 samples (rows) and 3 features (columns).

```python
X = np.array([[1.0, 2.0, 3.0],
              [3.0, 4.0, 5.0],
              [5.0, 6.0, 7.0],
              [7.0, 8.0, 9.0]])      # shape (4, 3)

mean = X.mean(axis=0)                # shape (3,)
centered = X - mean                  # (4, 3) - (3,) -> (4, 3)

print(mean)
print(centered)
```

| Code | Explanation |
|---|---|
| `X.mean(axis=0)` | Average down the rows, giving one mean per feature: $(1+3+5+7)/4 = 4$, then $5$, then $6$ |
| `X - mean` | `mean` (shape `(3,)`) is treated as `(1, 3)` and stretched across all 4 rows |

**Expected output:**

```text
[4. 5. 6.]
[[-3. -3. -3.]
 [-1. -1. -1.]
 [ 1.  1.  1.]
 [ 3.  3.  3.]]
```

### 2.4.6 When broadcasting fails

```python
np.array([1, 2, 3]) + np.array([1, 2, 3, 4])
```

Shapes `(3,)` and `(4,)`: the last dimensions are 3 and 4, neither equal nor 1. NumPy raises a `ValueError` whose message mentions that the operands could not be broadcast together with shapes `(3,)` and `(4,)`.

### 2.4.7 Common misconception: broadcasting is always safe

Broadcasting can **silently** produce a result you did not intend. This is one of the most common bugs in deep-learning code and appears again in Section 2.10.

---

## 2.5 Tensors

### 2.5.1 Definition

A **tensor** is a multi-dimensional array of numbers. The number of dimensions is its **rank** (also called order or ndim).

| Name | Rank (dimensions) | Example | Shape |
|---|---|---|---|
| Scalar | 0 | A single number: `7` | `()` |
| Vector | 1 | A list of numbers: `[1, 2, 3]` | `(3,)` |
| Matrix | 2 | A table: 2 rows, 3 columns | `(2, 3)` |
| 3-D tensor | 3 | A stack of matrices, e.g. one colour image (channels, height, width) | `(3, 32, 32)` |
| 4-D tensor | 4 | A batch of images | `(64, 3, 32, 32)` |

NumPy calls all of these `ndarray`. PyTorch calls them `torch.Tensor`. The formal linear-algebra meaning of these objects is the subject of Chapter 3; here we focus on handling them in code.

### 2.5.2 Why PyTorch tensors, if NumPy already exists?

A PyTorch tensor looks and behaves much like a NumPy array, but adds two abilities that deep learning requires:

1. **Automatic differentiation** (`autograd`): PyTorch can record operations and compute gradients for you (previewed in Section 2.5.8, covered fully in Chapters 4, 8, and 9).
2. **GPU execution:** a tensor can live in GPU memory and be processed with massively parallel hardware (Section 2.7).

| Feature | NumPy `ndarray` | PyTorch `Tensor` |
|---|---|---|
| Multi-dimensional numeric array | Yes | Yes |
| Vectorized operations and broadcasting | Yes | Yes |
| Computes gradients automatically | No | Yes (`requires_grad=True`) |
| Runs on GPU | No (CPU only) | Yes |
| Default float type | `float64` | `float32` |
| Default integer type | `int64` (platform-dependent) | `int64` |
| Typical role | Data preparation, plotting, classical numerics | Model building and training |

### 2.5.3 Creating tensors

```python
import torch

t = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

print(t)
print(t.shape)
print(t.dtype)
print(t.device)
print(t.ndim)
print(t.numel())
```

| Property | Meaning | Value here |
|---|---|---|
| `t.shape` | Size of each dimension (a `torch.Size`, which behaves like a tuple) | `torch.Size([2, 3])` |
| `t.dtype` | Element data type | `torch.float32` |
| `t.device` | Where the tensor lives | `cpu` |
| `t.ndim` | Number of dimensions | `2` |
| `t.numel()` | Total number of elements | `6` |

**Expected output:**

```text
tensor([[1., 2., 3.],
        [4., 5., 6.]])
torch.Size([2, 3])
torch.float32
cpu
2
6
```

**Other creation functions:**

```python
print(torch.zeros(2, 3))          # shape (2, 3), all zeros
print(torch.ones(2, 2))           # shape (2, 2), all ones
print(torch.arange(0, 10, 2))     # tensor([0, 2, 4, 6, 8])
print(torch.linspace(0, 1, 5))    # 5 evenly spaced values from 0 to 1
print(torch.rand(2, 2))           # uniform random numbers in [0, 1)
print(torch.randn(2, 2))          # random numbers from a standard normal distribution
```

| Function | Notes |
|---|---|
| `torch.zeros(2, 3)` | Shape is given as separate arguments (a tuple also works) |
| `torch.rand` | Values are random, so your output will differ from anyone else's unless you set a seed (Section 2.9) |
| `torch.randn` | "n" stands for normal (Gaussian) distribution with mean 0 and standard deviation 1. Chapter 5 explains distributions |

### 2.5.4 Data types (dtypes)

```python
ints = torch.tensor([1, 2, 3])
floats = torch.tensor([1.0, 2.0, 3.0])
print(ints.dtype)
print(floats.dtype)
```

**Expected output:**

```text
torch.int64
torch.float32
```

PyTorch infers the type from what you write: Python integers give `int64`, Python floats give `float32`.

| dtype | Typical use | Bytes per element |
|---|---|---|
| `torch.float32` | **Default for model weights and inputs** | 4 |
| `torch.float64` | High-precision numerics (rarely needed in training) | 8 |
| `torch.float16`, `torch.bfloat16` | Reduced-precision training/inference to save memory (later chapters) | 2 |
| `torch.int64` | Class labels, indices, token IDs | 8 |
| `torch.bool` | Masks | 1 |

**Memory arithmetic.** Memory used by a tensor is $\text{numel} \times \text{bytes per element}$. For one batch of 64 colour images of size $3 \times 224 \times 224$ stored as `float32`:

$$64 \times 3 \times 224 \times 224 = 9{,}633{,}792 \text{ elements}, \quad 9{,}633{,}792 \times 4 = 38{,}535{,}168 \text{ bytes}$$

That is about 38.5 MB for the input batch alone, before the model and intermediate results are counted. This calculation explains many "out of memory" errors (see "Training resource limitations" in later chapters).

**Converting types:**

```python
x = torch.tensor([1, 2, 3])
y = x.to(torch.float32)     # or x.float()
print(y.dtype)
```

**Expected output:**

```text
torch.float32
```

### 2.5.5 Indexing and reductions

Indexing works as in NumPy:

```python
print(t[0, 1])         # tensor(2.)
print(t[:, 1])         # tensor([2., 5.])
print(t[t > 3])        # tensor([4., 5., 6.])
```

Reductions use `dim` (the PyTorch name for NumPy's `axis`):

```python
print(t.sum())                      # tensor(21.)
print(t.sum(dim=0))                 # tensor([5., 7., 9.])
print(t.mean(dim=1))                # tensor([2., 5.])
print(t.mean(dim=1, keepdim=True))  # shape (2, 1) instead of (2,)
print(t[0, 1].item())               # 2.0 as an ordinary Python float
```

| Code | Explanation |
|---|---|
| `dim=0` | Collapse the first dimension (sum down the rows) |
| `keepdim=True` | Keeps the collapsed dimension as size 1. This makes later broadcasting predictable (Section 2.10) |
| `.item()` | Converts a **single-element** tensor into a plain Python number. Use it for printing or logging a loss value |

**Expected output (in order):**

```text
tensor(2.)
tensor([2., 5.])
tensor([4., 5., 6.])
tensor(21.)
tensor([5., 7., 9.])
tensor([2., 5.])
tensor([[2.],
        [5.]])
2.0
```

**Common error.** Mean of an integer tensor fails:

```python
torch.tensor([1, 2, 3]).mean()   # raises an error
```

PyTorch complains that the mean needs a floating-point (or complex) dtype. The fix is `torch.tensor([1, 2, 3]).float().mean()`. The exact wording of the message varies between versions.

### 2.5.6 Arithmetic and matrix multiplication

```python
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0],
                  [7.0, 8.0]])

print(A * B)      # element-wise
print(A @ B)      # matrix multiplication
```

| Operator | Meaning |
|---|---|
| `A + B`, `A - B`, `A * B`, `A / B` | **Element-wise**: operate on matching positions |
| `A @ B` (or `torch.matmul(A, B)`) | **Matrix multiplication** (formal definition in Chapter 3) |

**Hand calculation.**

- Element-wise: $[1 \times 5,\ 2 \times 6;\ 3 \times 7,\ 4 \times 8] = [5, 12; 21, 32]$.
- Matrix product, top-left entry: $1 \times 5 + 2 \times 7 = 19$; top-right: $1 \times 6 + 2 \times 8 = 22$; bottom-left: $3 \times 5 + 4 \times 7 = 43$; bottom-right: $3 \times 6 + 4 \times 8 = 50$.

**Expected output:**

```text
tensor([[ 5., 12.],
        [21., 32.]])
tensor([[19., 22.],
        [43., 50.]])
```

**The shape rule for matrix multiplication:**

$$(m \times n)\ @\ (n \times p) \;\longrightarrow\; (m \times p)$$

| Symbol | Meaning |
|---|---|
| $m$ | Rows of the left matrix (and of the result) |
| $n$ | Columns of the left matrix, which **must equal** rows of the right matrix |
| $p$ | Columns of the right matrix (and of the result) |

The two inner numbers must match and then disappear. This single rule explains most shape errors.

**Preview of a neural-network layer.** A layer computes `X @ W + b`, where `X` holds a batch of inputs, `W` the weights, and `b` the bias:

```python
X = torch.ones(4, 3)                 # 4 samples, 3 features each
W = torch.ones(3, 2)                 # maps 3 features to 2 outputs
b = torch.tensor([0.5, -0.5])        # one bias per output, shape (2,)

out = X @ W + b
print(out.shape)
print(out)
```

| Step | Shapes | Explanation |
|---|---|---|
| `X @ W` | `(4, 3) @ (3, 2)` gives `(4, 2)` | Inner 3s match and vanish. Every entry is $1+1+1 = 3$ |
| `+ b` | `(4, 2) + (2,)` gives `(4, 2)` | Broadcasting adds `b` to every row |

Each row becomes $[3 + 0.5,\ 3 - 0.5] = [3.5,\ 2.5]$.

**Expected output:**

```text
torch.Size([4, 2])
tensor([[3.5000, 2.5000],
        [3.5000, 2.5000],
        [3.5000, 2.5000],
        [3.5000, 2.5000]])
```

You have just computed a complete (linear) neural-network layer. Chapter 7 adds the nonlinear activation that turns it into a useful building block.

### 2.5.7 Changing shape: reshape, view, squeeze, unsqueeze, permute

```python
x = torch.arange(6)                  # tensor([0, 1, 2, 3, 4, 5]), shape (6,)

print(x.reshape(2, 3))               # shape (2, 3)
print(x.reshape(3, -1).shape)        # -1 is inferred as 2 -> (3, 2)

v = torch.tensor([1.0, 2.0, 3.0])    # shape (3,)
print(v.unsqueeze(0).shape)          # (1, 3): new dimension at position 0
print(v.unsqueeze(1).shape)          # (3, 1): new dimension at position 1
print(v.unsqueeze(0).squeeze().shape)  # squeeze removes size-1 dimensions -> (3,)

img = torch.zeros(32, 32, 3)         # height, width, channels (common image layout)
chw = img.permute(2, 0, 1)           # reorder dimensions -> (3, 32, 32)
print(chw.shape)
```

| Operation | What it does | Result |
|---|---|---|
| `reshape(2, 3)` | Same elements, new shape. Total count must match | `(6,)` becomes `(2, 3)` |
| `reshape(3, -1)` | `-1` is computed automatically | `(3, 2)` |
| `unsqueeze(d)` | Inserts a new dimension of size 1 at position `d` | `(3,)` becomes `(1, 3)` or `(3, 1)` |
| `squeeze()` | Removes all dimensions of size 1 | `(1, 3)` becomes `(3,)` |
| `permute(2, 0, 1)` | Reorders dimensions: new dimension 0 is old dimension 2, and so on | `(32, 32, 3)` becomes `(3, 32, 32)` |

**Expected output:**

```text
tensor([[0, 1, 2],
        [3, 4, 5]])
torch.Size([3, 2])
torch.Size([1, 3])
torch.Size([3, 1])
torch.Size([3])
torch.Size([3, 32, 32])
```

**Why `permute` matters for images.** Image files and libraries such as Matplotlib typically use height × width × channels. PyTorch's convolution layers expect **channels first**: a batch is shaped `(N, C, H, W)`, where `N` is batch size, `C` channels, `H` height, `W` width (Chapter 13).

**Comparison: `reshape` vs `view`.**

| | `reshape` | `view` |
|---|---|---|
| Works on any tensor | Yes | Only if memory layout is compatible ("contiguous") |
| May copy data | Sometimes (only if needed) | Never; always shares memory |
| Safe default for beginners | **Yes** | Can raise an error after `permute` or transpose |

If you use `.view()` right after `.permute()`, PyTorch raises an error saying the tensor is not contiguous. Use `.reshape()` instead, or call `.contiguous()` first.

**Common shapes you will meet repeatedly:**

| Data | Typical batch shape | Meaning | Chapter |
|---|---|---|---|
| Tabular | `(N, F)` | `N` samples, `F` features | 7 to 9 |
| Images | `(N, C, H, W)` | batch, channels, height, width | 13 |
| Sequences | `(N, T, D)` | batch, time steps (tokens), features per step. PyTorch recurrent layers use this layout when `batch_first=True` is set | 15 to 17 |

### 2.5.8 A first look at `requires_grad` (preview only)

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)
```

| Code | Explanation |
|---|---|
| `requires_grad=True` | Tells PyTorch to record every operation involving `x` so it can later differentiate |
| `y = x ** 2` | Computes $y = x^2 = 9$ while recording the step |
| `y.backward()` | Asks PyTorch to compute $\frac{dy}{dx}$ automatically |
| `x.grad` | Stores the result |

The slope of $y = x^2$ at $x = 3$ is $2x = 6$. (Derivatives are developed properly in Chapter 4.)

**Expected output:**

```text
tensor(6.)
```

This is the capability that lets PyTorch train networks with millions of parameters without you writing any calculus by hand.

---

## 2.6 Moving Between NumPy and PyTorch

```python
arr = np.array([1.0, 2.0, 3.0])      # NumPy float64

t1 = torch.from_numpy(arr)           # shares memory with arr
t2 = torch.tensor(arr)               # makes an independent copy

arr[0] = 99.0
print(t1)
print(t2)

back = t2.numpy()                    # tensor -> NumPy (CPU tensors only)
print(type(back))
```

| Code | Explanation |
|---|---|
| `torch.from_numpy(arr)` | Builds a tensor **sharing the same memory**. Changing one changes the other |
| `torch.tensor(arr)` | **Copies** the data. The two are independent |
| `arr[0] = 99.0` | Modifies the NumPy array in place |
| `t2.numpy()` | Returns a NumPy array sharing memory with the CPU tensor |

**Expected output:**

```text
tensor([99.,  2.,  3.], dtype=torch.float64)
tensor([1., 2., 3.], dtype=torch.float64)
<class 'numpy.ndarray'>
```

Notice two things:

1. `t1` changed because it shares memory with `arr`; `t2` did not.
2. Both tensors are `float64`, because NumPy's default float is 64-bit. Models usually expect `float32`, so convert with `.float()` before feeding data to a network.

**Rules of thumb:**

| Situation | Use |
|---|---|
| Want an independent tensor | `torch.tensor(arr)` |
| Want zero-copy conversion and understand the sharing | `torch.from_numpy(arr)` |
| Tensor has gradients tracking | `t.detach().numpy()` |
| Tensor is on a GPU | `t.cpu().numpy()` |
| Both of the above | `t.detach().cpu().numpy()` |

Calling `.numpy()` directly on a tensor that requires gradients, or on a GPU tensor, raises an error whose message tells you to use `detach()` or `cpu()` first.

---

## 2.7 CPU vs GPU

### 2.7.1 Definitions

- **CPU (Central Processing Unit):** a general-purpose processor with a small number of powerful cores, good at complex sequential logic.
- **GPU (Graphics Processing Unit):** a processor with thousands of simpler cores designed to perform the same operation on many numbers simultaneously. Originally built for graphics, it is also well suited to the matrix arithmetic in neural networks.

### 2.7.2 Why GPUs help

A matrix multiplication consists of many independent multiply-and-add operations. They do not depend on each other, so they can be done in parallel. A CPU does a limited number at a time; a GPU can do vastly more at once.

**Analogy.** A CPU is a small team of expert engineers who can each solve any hard problem. A GPU is a huge hall of workers who can each do one simple task, all at the same time. If the job is "add up a million pairs of numbers", the hall wins.

### 2.7.3 Comparison

| Aspect | CPU | GPU |
|---|---|---|
| Cores | Few, powerful | Thousands, simpler |
| Best at | Varied logic, small tasks, data loading | Large parallel numeric workloads (matrix and tensor operations) |
| Memory | System RAM (usually large) | Separate GPU memory (VRAM), usually smaller |
| Data transfer | | Moving data between CPU and GPU memory takes time |
| Needed in this book | Always works | Helpful from Chapter 13 onward; optional before that |

### 2.7.4 Misconception: "GPU is always faster"

Not for small workloads. Moving data to the GPU and launching GPU operations has overhead. For tiny tensors, the CPU is often equal or faster. The GPU advantage appears when tensors and models are large.

### 2.7.5 Device-agnostic code

Write code that works on both. Choose the device once, then place tensors there.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

x = torch.randn(1000, 1000)          # created on the CPU
x = x.to(device)                     # move to chosen device
y = torch.randn(1000, 1000, device=device)   # create directly on the device

z = x @ y
print(z.device)

z_numpy = z.cpu().numpy()            # bring back to CPU before converting
print(z_numpy.shape)
```

| Code | Explanation |
|---|---|
| `torch.device("cuda" if ... else "cpu")` | Picks the GPU when available, otherwise the CPU. `"cuda"` is PyTorch's name for NVIDIA GPU devices |
| `x.to(device)` | Returns a tensor on the chosen device. **It is not in-place**: you must assign the result |
| `torch.randn(..., device=device)` | Creates the tensor directly on the target device, avoiding a copy |
| `x @ y` | Matrix multiplication. Both tensors must be on the **same device** |
| `z.cpu().numpy()` | NumPy works only with CPU memory, so move the tensor back first |

**Expected output:** on a CPU-only machine:

```text
Using device: cpu
cpu
(1000, 1000)
```

With a working NVIDIA GPU, the first line becomes `Using device: cuda` and the second shows a CUDA device (for example `cuda:0`).

**Other accelerators.** On Apple Silicon Macs, PyTorch can use the `"mps"` device (check with `torch.backends.mps.is_available()`). The same `.to(device)` pattern applies. This book's examples use the CPU/CUDA pattern above.

### 2.7.6 The device mismatch error

If one tensor is on the CPU and another on the GPU, an operation between them fails with a `RuntimeError` stating that the tensors are expected to be on the same device. **Fix:** move everything (inputs, labels, and later the model) to the same `device`.

### 2.7.7 Measuring speed fairly

GPU operations run **asynchronously**: Python may move on before the GPU finishes. To time them honestly, call `torch.cuda.synchronize()` before reading the clock.

```python
import time

def time_matmul(device, size=2000, repeats=5):
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    _ = a @ b                                # warm-up run (not timed)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        c = a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()             # wait for the GPU to finish
    elapsed = time.perf_counter() - start
    return elapsed / repeats

cpu_time = time_matmul(torch.device("cpu"))
print(f"CPU: {cpu_time:.4f} s per multiplication")

if torch.cuda.is_available():
    gpu_time = time_matmul(torch.device("cuda"))
    print(f"GPU: {gpu_time:.4f} s per multiplication")
```

| Code | Explanation |
|---|---|
| `device.type` | A string, `"cpu"` or `"cuda"` |
| Warm-up run | The first GPU call includes one-time setup costs that would distort timing |
| `torch.cuda.synchronize()` | Blocks until all queued GPU work finishes |
| `elapsed / repeats` | Average time per multiplication |

Output depends entirely on your hardware, so no numbers are shown. On a machine with a capable GPU, you should typically see the GPU time well below the CPU time for this size.

---

## 2.8 Using Google Colab

### 2.8.1 Setup steps

1. Open Google Colab in your browser and sign in with a Google account.
2. Create a new notebook.
3. To request a GPU: open the **Runtime** menu, choose **Change runtime type**, and select a GPU accelerator. Which GPU types are offered, and whether one is available at a given moment, depends on your account and current demand. Menu labels can change over time.
4. Run the verification cell:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

5. To see which GPU you were given, run this in a cell. The leading `!` tells Colab to run a shell command instead of Python:

```text
!nvidia-smi
```

6. Colab notebooks usually come with PyTorch and NumPy pre-installed. To add other packages:

```text
!pip install package_name
```

### 2.8.2 Things to know about Colab

| Point | Practical consequence |
|---|---|
| Sessions are temporary | Files saved in the session's local storage are lost when it ends. Download important files or save them to Google Drive |
| GPU access is not guaranteed | Code should still run on CPU (use the device-agnostic pattern from Section 2.7.5) |
| Idle or long sessions can be disconnected | Save checkpoints regularly (Chapter 9 shows how) |
| Libraries may update | Print versions at the top of every notebook for reproducibility |

To save files to Google Drive from a Colab notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
```

This asks for permission and then makes your Drive appear under `/content/drive`. This import works **only inside Colab**.

### 2.8.3 Adapting any example from this book for GPU or Colab

Every foundational example in this book runs on a CPU. To use a GPU, change only these places:

1. Add `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` near the top.
2. Move each tensor with `.to(device)`.
3. From Chapter 9 onward, also move the model: `model = model.to(device)`.
4. Before using NumPy or Matplotlib on a result, call `.cpu()` (and `.detach()` if it tracks gradients).

---

## 2.9 Reproducibility

### 2.9.1 The problem

Neural networks involve randomness: initial weights, shuffling of training data, dropout, and data augmentation. If you run the same code twice, you may get different results. For learning, debugging, and comparing experiments, you want to control this.

### 2.9.2 Random seeds

A **seed** is the starting value for a pseudo-random number generator. The same seed produces the same sequence of "random" numbers. Python, NumPy, and PyTorch each have their own generator, so seed all of them.

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)                 # Python's built-in random module
    np.random.seed(seed)              # NumPy's legacy global generator
    torch.manual_seed(seed)           # PyTorch CPU generator (and the default CUDA generators)
    torch.cuda.manual_seed_all(seed)  # all GPUs; ignored if CUDA is unavailable

set_seed(42)
a = torch.rand(3)

set_seed(42)
b = torch.rand(3)

print(torch.equal(a, b))
```

| Code | Explanation |
|---|---|
| `random.seed(seed)` | Seeds Python's standard library generator |
| `np.random.seed(seed)` | Seeds NumPy's legacy global generator. (Newer NumPy code may use `np.random.default_rng(seed)`, which is a separate generator you pass around) |
| `torch.manual_seed(seed)` | Seeds PyTorch's generator |
| `torch.cuda.manual_seed_all(seed)` | Seeds GPU generators for every GPU |
| `torch.equal(a, b)` | `True` if two tensors have the same shape and identical values |

Calling `set_seed(42)` twice resets the generator to the same starting point, so `a` and `b` contain the same numbers.

**Expected output:**

```text
True
```

### 2.9.3 Where randomness enters a training run

| Source | Chapter |
|---|---|
| Weight initialization | 10 |
| Shuffling data each epoch (`DataLoader`) | 9 |
| Dropout | 10 |
| Data augmentation | 13 and 14 |
| Some GPU operations (non-deterministic by design for speed) | this chapter |

### 2.9.4 Stricter determinism and its limits

PyTorch offers `torch.use_deterministic_algorithms(True)`, which makes PyTorch use deterministic algorithms where available and raise an error for operations that have no deterministic version. Some GPU setups also need an environment variable (`CUBLAS_WORKSPACE_CONFIG`) for full determinism. Deterministic mode can be slower.

### 2.9.5 Misconceptions

| Misconception | Correction |
|---|---|
| "Setting a seed makes results identical everywhere." | It gives repeatable results on the **same** setup. Different hardware, drivers, or library versions can still change numerical results slightly |
| "Reproducibility means getting the best result." | A seed makes a run repeatable, not good. Report results across several seeds when comparing methods |
| "One seed call at the top is enough forever." | Random state advances as you use it. Re-seed (or use dedicated generators) at the start of each experiment you want to compare |

### 2.9.6 Reproducibility checklist

- Set seeds at the start.
- Print library versions (`torch.__version__`, `np.__version__`).
- Record hyperparameters, data splits, and hardware.
- Save the code that produced each result.

---

## 2.10 Debugging Shapes, Types, and Devices

Most beginner errors in Deep Learning are not mathematical; they are **shape, dtype, or device** errors. The habit that prevents most of them is printing shapes.

### 2.10.1 Silent broadcasting bug (the most important example)

```python
y_pred = torch.tensor([[1.0], [2.0], [3.0]])   # shape (3, 1)
y_true = torch.tensor([1.0, 2.0, 3.0])         # shape (3,)

diff = y_pred - y_true
print(diff.shape)
print(diff)
```

Shapes `(3, 1)` and `(3,)`: the second is padded to `(1, 3)`, and both are stretched to `(3, 3)`. Instead of 3 element-wise differences, you get **9** (every prediction minus every target). No error is raised.

**Expected output:**

```text
torch.Size([3, 3])
tensor([[ 0., -1., -2.],
        [ 1.,  0., -1.],
        [ 2.,  1.,  0.]])
```

If you then average the squares of this wrong result as an MSE, you get a wrong loss and your model trains poorly or not at all.

**Fix:** make shapes agree explicitly.

```python
diff_ok = y_pred.squeeze(1) - y_true      # (3,) - (3,) -> (3,)
print(diff_ok)
```

**Expected output:**

```text
tensor([0., 0., 0.])
```

### 2.10.2 Error catalogue

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError` saying two matrices cannot be multiplied (for example shapes `2x3` and `2x3`) | Inner dimensions do not match in `A @ B` | Print both shapes. Transpose with `.T` or fix a layer size so the inner numbers agree |
| `ValueError`/`RuntimeError` about shapes that cannot be broadcast | Dimensions neither equal nor 1 | Use `reshape`, `unsqueeze`, or `squeeze` to align dimensions |
| Operation works but the result shape is larger than expected | Silent broadcasting (Section 2.10.1) | Assert expected shapes; use `keepdim=True` or `squeeze` deliberately |
| Error about mismatched dtypes in a matrix multiplication or layer | One tensor is `float64` (or an integer type) and another `float32` | Convert with `.float()` or `.to(torch.float32)` |
| Error that mean/average needs a floating-point type | Tensor holds integers | `.float()` before `.mean()` |
| Error that tensors must be on the same device | One tensor on CPU, one on GPU | Move all with `.to(device)` |
| Error when calling `.numpy()` on a tensor that requires grad or sits on a GPU | NumPy cannot hold gradient history or GPU memory | `t.detach().cpu().numpy()` |
| Error that a tensor is not contiguous when using `.view()` | `.view()` after `permute` or transpose | Use `.reshape()` or `.contiguous()` |
| "Why is my result different every run?" | Unseeded randomness | Use `set_seed` (Section 2.9) |

### 2.10.3 A shape-printing habit

```python
def describe(name, t):
    print(f"{name:10s} shape={tuple(t.shape)}  dtype={t.dtype}  device={t.device}")

describe("X", X := torch.ones(4, 3))
describe("W", W := torch.ones(3, 2))
describe("X @ W", X @ W)
```

| Code | Explanation |
|---|---|
| `f"{name:10s}"` | Pads the name to width 10 so columns line up |
| `tuple(t.shape)` | Shows the shape as a plain tuple |
| `X := torch.ones(4, 3)` | The "walrus" operator assigns and passes the value along in one step (Python 3.8 or newer) |

**Expected output:**

```text
X          shape=(4, 3)  dtype=torch.float32  device=cpu
W          shape=(3, 2)  dtype=torch.float32  device=cpu
X @ W      shape=(4, 2)  dtype=torch.float32  device=cpu
```

---

## 2.11 Comparison Summary

| Question | NumPy | PyTorch |
|---|---|---|
| Create from a list | `np.array([...])` | `torch.tensor([...])` |
| Shape | `a.shape` | `t.shape` |
| Reshape | `a.reshape(...)` | `t.reshape(...)` |
| Sum along a dimension | `a.sum(axis=0)` | `t.sum(dim=0)` |
| Matrix multiply | `a @ b` | `t @ u` |
| Add a dimension | `a[:, None]` or `np.expand_dims` | `t.unsqueeze(1)` |
| Reorder dimensions | `a.transpose(...)` | `t.permute(...)` |
| Convert type | `a.astype(np.float32)` | `t.to(torch.float32)` or `t.float()` |
| Run on GPU | Not supported | `t.to("cuda")` |
| Automatic gradients | No | `requires_grad=True` |

---

## Exercises

1. **Shapes by hand.** For each, give the result shape or say it fails: (a) `(5, 1) + (1, 4)`, (b) `(5, 4) + (4,)`, (c) `(5, 4) + (5,)`, (d) `(2, 3, 4) + (3, 1)`.
2. **Axis practice.** For `a = np.array([[1, 2, 3], [4, 5, 6]])`, predict `a.mean(axis=0)` and `a.mean(axis=1)` by hand, then check in code.
3. **Matrix shapes.** A batch `X` has shape `(32, 10)`. Which shapes of `W` make `X @ W` valid, and what is the result shape if `W` is `(10, 5)`?
4. **Memory.** How many megabytes does a `float32` batch of shape `(128, 3, 64, 64)` occupy? (Use bytes $=$ elements $\times 4$ and $1\text{ MB} = 10^6$ bytes.)
5. **The silent bug.** Reproduce the bug in Section 2.10.1, then compute the (wrong) mean of squared differences and the (correct) one. Compare the numbers.
6. **Copy or share?** Create a NumPy array, build one tensor with `torch.from_numpy` and another with `torch.tensor`, modify the original array, and explain what you observe.
7. **Class practice.** Write a class `Layer` with `__init__(self, weight, bias)` and `__call__` that applies $wx + b$, then make `ReLULayer` inherit from it and apply `max(0, ...)` to the output.
8. **Reproducibility.** Generate `torch.randn(3)` three times with `set_seed(0)` called before each, then once without re-seeding between two calls. Explain the difference.
