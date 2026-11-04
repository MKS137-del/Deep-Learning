# Chapter 5: Math III: Probability and Statistics

**Level 1: Essential Foundations | Tag: Essential**

## Chapter Roadmap

Chapters 3 and 4 gave us the machinery of a network (matrices) and the machinery of learning (gradients). This chapter explains **what the loss functions mean**. A network's output is best understood as a statement about uncertainty, and the standard losses (mean squared error and cross-entropy) fall out of probability theory rather than being arbitrary choices.

By the end of this chapter you will be able to:

- Describe a probability distribution and read basic probability notation.
- Compute expectation, variance, and standard deviation, and explain the difference between population and sample versions.
- Recognise the Bernoulli, categorical, uniform, and normal distributions and where each appears in Deep Learning.
- Explain likelihood and maximum likelihood estimation, and why we minimise the **negative log-likelihood**.
- Compute entropy, cross-entropy, and KL divergence on small examples.
- Explain why mean squared error corresponds to a Gaussian assumption and cross-entropy to a categorical one.
- Standardise data correctly without leaking test-set information.

**Scope.** Only the probability and statistics needed to understand training and evaluation. Bayes' theorem, hypothesis testing, and confidence intervals are not needed in the early chapters and are skipped. Ideas such as KL divergence reappear when we meet VAEs (Chapter 20).

> **Honesty note.** The code in this chapter has not been executed by me. Every "Expected output" was derived by hand. Where a block involves random numbers, the text states what you should see only approximately. Run each block and compare.

---

## 5.1 Why Probability?

Real data is noisy and incomplete. Two houses with identical features sell for different prices; the same blurry photo could be a cat or a fox. A model that outputs a single "certain" answer hides this uncertainty. Probability gives us a language to describe it.

| Where probability appears | Role |
|---|---|
| Classifier output | A probability for each class, for example $[0.7, 0.2, 0.1]$ |
| Loss functions | Derived from the likelihood of the data under the model |
| Weight initialisation | Weights are drawn from a random distribution (Chapter 10) |
| Dropout | Randomly zeroes activations using Bernoulli draws (Chapter 10) |
| Generative models | Learn a distribution over data and sample from it (Chapters 20 to 22) |
| Language models | Output a probability for every possible next token (Chapters 17 and 23) |
| Evaluation | Statistics of errors across a test set (Chapter 12) |

---

## 5.2 Probability Basics

### 5.2.1 Formal definitions

- An **experiment** is a process with an uncertain outcome (rolling a die).
- The **sample space** $\Omega$ is the set of all possible outcomes ($\{1,2,3,4,5,6\}$).
- An **event** is a subset of outcomes ("the roll is even" is $\{2,4,6\}$).
- A **probability** $P(A)$ assigns each event a number satisfying:
  1. $0 \le P(A) \le 1$.
  2. $P(\Omega) = 1$: something must happen.
  3. For events that cannot occur together, probabilities add: $P(A \text{ or } B) = P(A) + P(B)$.

For equally likely outcomes, $P(A) = \dfrac{\text{number of outcomes in } A}{\text{total number of outcomes}}$. For the die, $P(\text{even}) = 3/6 = 0.5$.

### 5.2.2 Random variables and distributions

A **random variable** $X$ is a quantity whose value is determined by a random outcome (the number shown on a die). Its **probability distribution** says how likely each value is.

| Kind | Values | Described by | Rule |
|---|---|---|---|
| **Discrete** | Countable (die faces, class labels) | **Probability mass function (PMF)** $p(x) = P(X = x)$ | $\sum_x p(x) = 1$ |
| **Continuous** | Any real number in a range (height, a weight value) | **Probability density function (PDF)** $p(x)$ | $\int p(x)\,dx = 1$ |

**Important difference for continuous variables.** A PDF value is a *density*, not a probability. For a continuous variable, $P(X = x) = 0$ for any single exact value; probabilities come from areas under the curve over an interval. A density can therefore exceed 1 (a very narrow distribution has a tall peak).

### 5.2.3 Joint, conditional, and independent

- **Joint probability** $P(A, B)$: both $A$ and $B$ happen.
- **Conditional probability** $P(A \mid B)$: the probability of $A$ *given that* $B$ is known to have happened:

$$P(A \mid B) = \frac{P(A, B)}{P(B)}$$

| Symbol | Meaning |
|---|---|
| $P(A \mid B)$ | Probability of $A$ given $B$. The vertical bar reads "given" |
| $P(A, B)$ | Probability that both happen |
| $P(B)$ | Probability of $B$; must be greater than 0 |

Rearranged, this is the **product rule**: $P(A, B) = P(A \mid B)\,P(B)$.

- $A$ and $B$ are **independent** if knowing one tells you nothing about the other: $P(A \mid B) = P(A)$, equivalently $P(A, B) = P(A)P(B)$.

**Worked example with two dice.** Let $A$ be "the sum is at least 10" and $B$ be "the first die shows 6".

- Sums of at least 10 come from $(4,6), (5,5), (5,6), (6,4), (6,5), (6,6)$: $6$ outcomes out of $36$, so $P(A) = 6/36 \approx 0.1667$.
- Given the first die is 6, the second die must be at least 4 (that is 4, 5, or 6): $P(A \mid B) = 3/6 = 0.5$.

Since $0.5 \neq 0.1667$, the events are **dependent**: learning the first die changes the odds.

```python
from itertools import product

outcomes = list(product(range(1, 7), repeat=2))   # all 36 (die1, die2) pairs

sum_at_least_10 = [o for o in outcomes if o[0] + o[1] >= 10]
p_a = len(sum_at_least_10) / len(outcomes)

first_is_6 = [o for o in outcomes if o[0] == 6]
both = [o for o in first_is_6 if o[0] + o[1] >= 10]
p_a_given_b = len(both) / len(first_is_6)

print(round(p_a, 4))
print(round(p_a_given_b, 4))
```

| Code | Explanation |
|---|---|
| `product(range(1, 7), repeat=2)` | Generates every pair $(d_1, d_2)$ with each die from 1 to 6 |
| `[o for o in outcomes if ...]` | A list comprehension (Chapter 2) that keeps only matching pairs |
| `len(...) / len(...)` | Counting favourable outcomes over total outcomes |
| `p_a_given_b` | Restricts attention to the outcomes where $B$ holds, then counts again |

**Expected output:**

```text
0.1667
0.5
```

### 5.2.4 Why conditional probability is central to Deep Learning

A classifier is a model of the **conditional distribution** $P(y \mid \mathbf{x};\ \theta)$: the probability of each label $y$ given an input $\mathbf{x}$, under parameters $\theta$. When a network outputs $[0.7, 0.2, 0.1]$ for an image, it is claiming $P(\text{class 1} \mid \text{image}) = 0.7$.

---

## 5.3 Expectation, Variance, and Standard Deviation

### 5.3.1 Expectation (the mean)

**Definition.** The **expectation** (expected value) of a discrete random variable is the probability-weighted average of its values:

$$\mathbb{E}[X] = \sum_x x \, p(x)$$

For a continuous variable the sum becomes an integral: $\mathbb{E}[X] = \int x\,p(x)\,dx$.

| Symbol | Meaning |
|---|---|
| $\mathbb{E}[X]$ | Expected value of $X$; often written $\mu$ |
| $x$ | A possible value of $X$ |
| $p(x)$ | Its probability |
| $\sum_x$ | Sum over all possible values |

**Intuition.** The long-run average if you repeat the experiment many times. **Analogy:** the balance point of a seesaw, where each value sits at position $x$ with weight $p(x)$.

**Worked example (fair die).**

$$\mathbb{E}[X] = 1\cdot\tfrac16 + 2\cdot\tfrac16 + 3\cdot\tfrac16 + 4\cdot\tfrac16 + 5\cdot\tfrac16 + 6\cdot\tfrac16 = \frac{21}{6} = 3.5$$

Note that 3.5 is not a value the die can show; an expectation need not be a possible outcome.

**Linearity of expectation.** For constants $a, b$: $\ \mathbb{E}[aX + b] = a\,\mathbb{E}[X] + b$. (And $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$ always, even when $X$ and $Y$ are dependent.)

**Expectation of a function.** The expected loss over a dataset or distribution is $\mathbb{E}[\ell(X)] = \sum_x \ell(x)\,p(x)$. Training minimises an expected loss, estimated by averaging over data.

### 5.3.2 Variance and standard deviation

**Definition.** The **variance** measures how spread out a variable is around its mean:

$$\mathrm{Var}(X) = \mathbb{E}\big[(X - \mu)^2\big] = \mathbb{E}[X^2] - \mu^2$$

The **standard deviation** is its square root, which has the same units as $X$:

$$\sigma = \sqrt{\mathrm{Var}(X)}$$

| Symbol | Meaning |
|---|---|
| $\mu = \mathbb{E}[X]$ | The mean |
| $(X - \mu)^2$ | Squared distance from the mean (squaring removes sign and emphasises large deviations) |
| $\sigma$ | Standard deviation; "typical distance from the mean" |

**Scaling rule:** $\mathrm{Var}(aX + b) = a^2\,\mathrm{Var}(X)$. Adding a constant shifts the distribution but does not change its spread; multiplying by $a$ scales the standard deviation by $|a|$.

**Worked example (fair die).**

$$\mathbb{E}[X^2] = \frac{1 + 4 + 9 + 16 + 25 + 36}{6} = \frac{91}{6} \approx 15.1667$$

$$\mathrm{Var}(X) = \frac{91}{6} - (3.5)^2 = 15.1667 - 12.25 = 2.9167, \qquad \sigma = \sqrt{2.9167} \approx 1.7078$$

```python
import numpy as np
import torch

values = np.arange(1, 7)             # die faces 1..6, shape (6,)
probs = np.full(6, 1 / 6)            # each face has probability 1/6, shape (6,)

mean = np.sum(values * probs)
variance = np.sum(probs * (values - mean) ** 2)
std = np.sqrt(variance)

print(round(mean, 4))
print(round(variance, 4))
print(round(std, 4))

# Simulation: the average of many rolls approaches the expectation
torch.manual_seed(0)
rolls = torch.randint(low=1, high=7, size=(100_000,))   # integers 1..6
print(round(rolls.float().mean().item(), 1))
```

| Code | Explanation |
|---|---|
| `np.full(6, 1 / 6)` | An array of six copies of $1/6$ |
| `values * probs` | Element-wise product $x \cdot p(x)$; summing gives $\mathbb{E}[X]$ |
| `(values - mean) ** 2` | Squared deviations from the mean |
| `torch.randint(low=1, high=7, size=...)` | Random integers from 1 up to **but not including** 7 |
| `.float().mean()` | `mean` needs a floating-point tensor (Chapter 2), so convert first |

**Expected output** (the last number is a simulation, so it will differ very slightly in the digits you do not see, but should round to 3.5):

```text
3.5
2.9167
1.7078
3.5
```

The simulation illustrates the **law of large numbers**: the average of many independent trials converges to the expectation.

### 5.3.3 Sample statistics: estimating from data

In practice we do not know the true distribution; we have a **sample** $x_1, \dots, x_n$ and compute estimates:

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i \qquad\qquad s^2 = \frac{1}{n - 1}\sum_{i=1}^{n} (x_i - \bar{x})^2$$

| Symbol | Meaning |
|---|---|
| $\bar{x}$ | Sample mean |
| $s^2$ | **Sample** variance (divides by $n-1$, called Bessel's correction, which removes a systematic underestimate when the true mean is unknown) |
| $\sigma^2$ computed with $\frac{1}{n}$ | **Population** variance (divides by $n$) |

**This is a frequent source of silent discrepancies between libraries:**

| Library call | Divides by | Default |
|---|---|---|
| `np.var(x)`, `np.std(x)` | $n$ | `ddof=0` (population). Use `ddof=1` for the sample version |
| `torch.var(x)`, `torch.std(x)` | $n - 1$ | Sample version by default. Use `correction=0` (or the older `unbiased=False`) for population |

**Worked example.** Data: $[2, 4, 4, 4, 5, 5, 7, 9]$, with $n = 8$.

- Mean: $(2+4+4+4+5+5+7+9)/8 = 40/8 = 5$.
- Deviations: $-3, -1, -1, -1, 0, 0, 2, 4$. Squares: $9, 1, 1, 1, 0, 0, 4, 16$, summing to $32$.
- Population variance: $32/8 = 4$, so $\sigma = 2$.
- Sample variance: $32/7 \approx 4.5714$, so $s \approx 2.1381$.

```python
data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])

print(data.mean())
print(data.var())
print(data.std())
print(round(data.var(ddof=1), 4))

t = torch.tensor(data)
print(round(t.var().item(), 4))
print(round(t.var(correction=0).item(), 4))
```

**Expected output:**

```text
5.0
4.0
2.0
4.5714
4.5714
4.0
```

(`correction=` requires a reasonably recent PyTorch version; if your version rejects it, use `unbiased=False` instead.)

### 5.3.4 Standardisation (z-scores)

Neural networks train better when input features are on comparable scales (Chapter 10 explains why). The standard fix is the **z-score**:

$$z = \frac{x - \mu}{\sigma}$$

After this transformation the data has mean 0 and standard deviation 1.

```python
z = (data - data.mean()) / data.std()
print(z)
print(round(z.mean(), 4), round(z.std(), 4))
```

For our data, $(2 - 5)/2 = -1.5$, $(4 - 5)/2 = -0.5$, $(5 - 5)/2 = 0$, $(7 - 5)/2 = 1$, $(9 - 5)/2 = 2$.

**Expected output:**

```text
[-1.5 -0.5 -0.5 -0.5  0.   0.   1.   2. ]
0.0 1.0
```

(The mean may display as `-0.0`; that is just a floating-point sign on zero.)

**A critical rule: avoid data leakage.** Compute $\mu$ and $\sigma$ **from the training set only**, then apply those same numbers to the validation and test sets. Computing them on the whole dataset lets information about the test data influence training, which inflates results. (Chapter 12 returns to this.)

---

## 5.4 Common Distributions

### 5.4.1 Bernoulli distribution (one yes/no event)

A variable that is $1$ with probability $p$ and $0$ with probability $1 - p$ (a biased coin flip):

$$P(X = 1) = p, \qquad P(X = 0) = 1 - p, \qquad \mathbb{E}[X] = p, \qquad \mathrm{Var}(X) = p(1 - p)$$

**Where it appears:** binary classification (spam or not spam), and dropout masks.

### 5.4.2 Categorical distribution (one of $K$ classes)

A variable taking one of $K$ values, with probabilities $p_1, \dots, p_K$ that are non-negative and sum to 1. This is the distribution a multi-class classifier outputs, after the softmax function (Chapter 7). Choosing the next token in a language model is a draw from a categorical distribution (Chapter 23).

### 5.4.3 Uniform distribution

All values in an interval $[a, b]$ are equally likely. `torch.rand` (Chapter 2) draws from the uniform distribution on $[0, 1)$. Some weight-initialisation schemes use uniform ranges (Chapter 10).

### 5.4.4 Normal (Gaussian) distribution

The bell curve. A normal distribution with mean $\mu$ and variance $\sigma^2$ has density:

$$\mathcal{N}(x;\ \mu, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

| Symbol | Meaning |
|---|---|
| $\mu$ | Mean: the centre of the bell |
| $\sigma$ | Standard deviation: its width |
| $\dfrac{1}{\sigma\sqrt{2\pi}}$ | Normalising constant so that the total area under the curve is 1 |
| $\exp(\cdot)$ | The exponential function $e^{(\cdot)}$ |
| $-\dfrac{(x-\mu)^2}{2\sigma^2}$ | Penalises distance from $\mu$, measured in units of $\sigma$ |

The **standard normal** has $\mu = 0$ and $\sigma = 1$. If $z$ is standard normal, then $\mu + \sigma z$ has mean $\mu$ and standard deviation $\sigma$ (the scaling rule from Section 5.3.2).

**The 68-95-99.7 rule:** approximately 68% of the probability lies within $1\sigma$ of the mean, 95% within $2\sigma$, and 99.7% within $3\sigma$.

**Why it is everywhere.** The **central limit theorem** (informally) says that adding up many small independent random effects tends to produce a bell-shaped result, whatever the individual effects look like. Measurement noise and many natural quantities behave this way. In Deep Learning the normal distribution is used for random weight initialisation (Chapter 10), for the noise in VAEs and diffusion models (Chapters 20 and 22), and in the Gaussian assumption behind mean squared error (Section 5.6.4).

```python
import math
from torch.distributions import Normal

# Density of the standard normal at x = 0
print(round(1 / math.sqrt(2 * math.pi), 4))

dist = Normal(loc=0.0, scale=1.0)
print(dist.log_prob(torch.tensor(0.0)))

# Sampling: draw standard normal numbers, then shift and scale
torch.manual_seed(0)
z = torch.randn(100_000)
samples = 5.0 + 2.0 * z                      # mean 5, standard deviation 2
print(round(samples.mean().item(), 1))
print(round(samples.std().item(), 1))

within_one_sigma = ((samples - 5.0).abs() < 2.0).float().mean()
print(round(within_one_sigma.item(), 2))
```

| Code | Explanation |
|---|---|
| `1 / math.sqrt(2 * math.pi)` | The peak density $\frac{1}{\sqrt{2\pi}}$ of the standard normal at $x=0$ |
| `Normal(loc=0.0, scale=1.0)` | A PyTorch distribution object: `loc` is $\mu$, `scale` is $\sigma$ (not the variance) |
| `dist.log_prob(x)` | The natural logarithm of the density at `x`. We use logs for numerical reasons (Section 5.6) |
| `5.0 + 2.0 * z` | Shifts and scales standard normal samples |
| `(samples - 5.0).abs() < 2.0` | A boolean tensor marking samples within one standard deviation |
| `.float().mean()` | The fraction of samples satisfying the condition |

**Expected output** (the last three lines are simulation results and should be approximately these values):

```text
0.3989
tensor(-0.9189)
5.0
2.0
0.68
```

Check: $\ln(0.39894) \approx -0.9189$, consistent with the second line.

### 5.4.5 Comparison of the distributions

| Distribution | Values | Parameters | Typical Deep Learning use |
|---|---|---|---|
| Bernoulli | 0 or 1 | $p$ | Binary labels; dropout masks |
| Categorical | One of $K$ classes | $p_1, \dots, p_K$ | Multi-class classifier output; next-token prediction |
| Uniform | Any value in $[a, b]$ | $a, b$ | Random initialisation; random sampling |
| Normal | Any real number | $\mu, \sigma$ | Weight initialisation; noise; assumption behind MSE |

---

## 5.5 Probability Is Not Likelihood

These two words describe the same formula viewed from opposite directions.

| | Probability | Likelihood |
|---|---|---|
| Written as | $P(\text{data} \mid \theta)$ | $\mathcal{L}(\theta \mid \text{data}) = P(\text{data} \mid \theta)$ |
| What is fixed | The parameters $\theta$ | The observed data |
| What varies | Possible data | The parameters $\theta$ |
| Question answered | "If the coin has $p = 0.5$, how likely is this outcome?" | "Given that I saw this outcome, which value of $p$ explains it best?" |
| Sums or integrates to 1? | Yes, over all possible data | **No**, not over $\theta$ |

Training a model is exactly the second question: the data is fixed, and we search for the parameters that make it most plausible.

---

## 5.6 Maximum Likelihood Estimation

### 5.6.1 Formal definition

**Maximum likelihood estimation (MLE)** chooses the parameters under which the observed data has the highest probability:

$$\hat{\theta}_{\text{MLE}} = \arg\max_{\theta}\ \mathcal{L}(\theta)$$

For a dataset of independent examples $(\mathbf{x}_i, y_i)$, $i = 1, \dots, N$, independence lets us multiply (the product rule from Section 5.2.3):

$$\mathcal{L}(\theta) = \prod_{i=1}^{N} P(y_i \mid \mathbf{x}_i;\ \theta)$$

| Symbol | Meaning |
|---|---|
| $\arg\max_\theta$ | "The value of $\theta$ that makes the following quantity largest" |
| $\prod_{i=1}^{N}$ | Multiply the terms for $i = 1$ to $N$ (like $\sum$ but with multiplication) |
| $P(y_i \mid \mathbf{x}_i; \theta)$ | The probability the model assigns to the correct answer for example $i$ |

### 5.6.2 Worked example: estimating a coin's bias

You flip a coin four times and see: Heads, Heads, Tails, Heads (3 heads, 1 tail). For a coin with $P(\text{heads}) = p$:

$$\mathcal{L}(p) = p \cdot p \cdot (1 - p) \cdot p = p^3(1 - p)$$

| $p$ | $\mathcal{L}(p) = p^3(1-p)$ |
|---|---|
| 0.5 | $0.125 \times 0.5 = 0.0625$ |
| 0.75 | $0.421875 \times 0.25 \approx 0.1055$ |
| 0.9 | $0.729 \times 0.1 = 0.0729$ |

The likelihood is highest near $p = 0.75$. To find the exact maximum, work with the logarithm (justified below):

$$\ln \mathcal{L}(p) = 3\ln p + \ln(1 - p), \qquad \frac{d}{dp}\ln\mathcal{L} = \frac{3}{p} - \frac{1}{1-p} = 0 \ \Rightarrow\ 3(1 - p) = p \ \Rightarrow\ p = \frac{3}{4}$$

The maximum-likelihood estimate is the observed fraction of heads, $3/4$, exactly the intuitive answer. (This used the calculus from Chapter 4: set the derivative to zero.)

```python
def likelihood(p, heads=3, tails=1):
    return p ** heads * (1 - p) ** tails

print(round(likelihood(0.5), 4), round(likelihood(0.75), 4), round(likelihood(0.9), 4))

p_grid = np.linspace(0.01, 0.99, 99)            # candidate values of p
best_p = p_grid[np.argmax(likelihood(p_grid))]
print(round(float(best_p), 2))
```

| Code | Explanation |
|---|---|
| `likelihood(p, heads, tails)` | Computes $p^{\text{heads}}(1-p)^{\text{tails}}$. Works for a single number or for a whole NumPy array |
| `np.linspace(0.01, 0.99, 99)` | 99 evenly spaced values from 0.01 to 0.99 (steps of 0.01) |
| `np.argmax(...)` | The **index** of the largest value |
| `p_grid[...]` | Looks up the candidate $p$ at that index |

**Expected output:**

```text
0.0625 0.1055 0.0729
0.75
```

This is the brute-force search from Chapter 1, now with a meaningful objective.

### 5.6.3 Why we use the log, and why we minimise the *negative*

Three practical reasons for taking logarithms:

1. **Products become sums.** $\ln \prod_i P_i = \sum_i \ln P_i$. Sums are easier to differentiate and to average over batches.
2. **Numerical stability.** Multiplying many probabilities (each below 1) shrinks toward zero until the computer rounds it to exactly 0 (called **underflow**).
3. **Same answer.** The logarithm is increasing, so the $\theta$ that maximises $\mathcal{L}$ also maximises $\ln\mathcal{L}$.

```python
probs = np.full(1100, 0.5)               # 1100 examples, each assigned probability 0.5
print(np.prod(probs))
print(round(np.sum(np.log(probs)), 4))
```

The true product is $0.5^{1100} \approx 10^{-331}$, smaller than the smallest positive number a 64-bit float can hold, so it prints `0.0`. The log version stays perfectly usable: $1100 \times \ln 0.5 \approx -762.4619$.

**Expected output:**

```text
0.0
-762.4619
```

Optimisers are written to **minimise** a loss (Chapter 4), so we flip the sign. The **negative log-likelihood (NLL)**, averaged over the data, is the standard training objective:

$$\text{NLL}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \ln P(y_i \mid \mathbf{x}_i;\ \theta)$$

**Minimising NLL is the same as maximising likelihood.** Every common loss function can be read as an NLL under some assumption about the output's distribution.

### 5.6.4 Why mean squared error comes from a Gaussian assumption

Suppose the target is the model's prediction plus Gaussian noise: $y = \hat{y} + \varepsilon$, with $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ and $\sigma$ fixed. Then $y$ is normally distributed around $\hat{y}$, and its negative log-density (from the formula in Section 5.4.4) is:

$$-\ln \mathcal{N}(y;\ \hat{y}, \sigma^2) = \frac{(y - \hat{y})^2}{2\sigma^2} + \ln\!\big(\sigma\sqrt{2\pi}\big)$$

The second term does not depend on the model. The first is the squared error scaled by a constant. So minimising the NLL under a Gaussian assumption is exactly **minimising mean squared error**. This explains why MSE is the natural loss for regression, and also its weakness: it assumes noise that is roughly Gaussian with constant variance.

### 5.6.5 Which distribution gives which loss

| Output type | Assumed distribution | Resulting loss | Chapter |
|---|---|---|---|
| Real number (regression) | Gaussian | Mean squared error | 7 |
| Yes/no (binary classification) | Bernoulli | Binary cross-entropy | 7 |
| One of $K$ classes | Categorical | Cross-entropy | 7 |

The remainder of this chapter develops cross-entropy through information theory, which gives it a second, complementary interpretation.

---

## 5.7 Entropy, Cross-Entropy, and KL Divergence

### 5.7.1 Information and entropy

The **information content** (surprise) of an event with probability $p$ is $-\ln p$. Rare events are very surprising (large value); certain events carry no surprise ($-\ln 1 = 0$).

**Entropy** is the average surprise of a distribution $p$:

$$H(p) = -\sum_{k} p_k \ln p_k$$

| Symbol | Meaning |
|---|---|
| $p_k$ | Probability of outcome $k$ |
| $-\ln p_k$ | Surprise of outcome $k$ |
| $H(p)$ | Expected surprise: how unpredictable the distribution is |

(By convention $0 \ln 0 = 0$.) With the natural logarithm the unit is the **nat**; with $\log_2$ it is the **bit**. PyTorch and most Deep Learning code use the natural log.

**Examples:**

- Fair coin: $H = -(0.5\ln 0.5 + 0.5\ln 0.5) = \ln 2 \approx 0.6931$ nats (1 bit). Maximum unpredictability for two outcomes.
- Biased coin, $p = 0.9$: $H = -(0.9\ln 0.9 + 0.1\ln 0.1) \approx 0.0948 + 0.2303 = 0.3251$ nats. More predictable, so lower entropy.
- Certain outcome: $H = 0$.

### 5.7.2 Cross-entropy

**Cross-entropy** measures the average surprise when the world follows distribution $p$ but you predicted using distribution $q$:

$$H(p, q) = -\sum_{k} p_k \ln q_k$$

| Symbol | Meaning |
|---|---|
| $p$ | The **true** distribution (for a labelled example, a one-hot vector) |
| $q$ | The **model's predicted** distribution |
| $\ln q_k$ | Log of the probability the model gave to outcome $k$ |

**Intuition.** If the model puts high probability on what actually happened, there is little surprise, and cross-entropy is low. If it puts tiny probability on the truth, the loss is large.

**Important special case.** For a labelled training example the true distribution is one-hot: all probability on the correct class $c$. Every term in the sum vanishes except one:

$$H(p, q) = -\ln q_c$$

This is precisely the **negative log-likelihood** of the correct class (Section 5.6.3). **Cross-entropy loss for classification is the negative log-likelihood under a categorical distribution.** The two interpretations (probabilistic and information-theoretic) are the same quantity.

**Worked example.** Three classes, true class is class 1 (first position), so $p = [1, 0, 0]$.

| Model output $q$ | Loss $= -\ln q_1$ | Comment |
|---|---|---|
| $[0.7, 0.2, 0.1]$ | $-\ln 0.7 \approx 0.3567$ | Fairly confident and right |
| $[0.1, 0.8, 0.1]$ | $-\ln 0.1 \approx 2.3026$ | Confident and **wrong**: large penalty |
| $[1.0, 0.0, 0.0]$ | $-\ln 1 = 0$ | Perfect |
| $[0.0001, 0.9998, 0.0001]$ | $-\ln 0.0001 \approx 9.2103$ | Extremely wrong: very large loss |

The loss grows without bound as the model's probability for the truth approaches zero, which strongly discourages confident mistakes.

### 5.7.3 KL divergence

The **Kullback-Leibler divergence** measures how much worse it is to use $q$ when the truth is $p$:

$$D_{\mathrm{KL}}(p \,\Vert\, q) = \sum_k p_k \ln\frac{p_k}{q_k} = H(p, q) - H(p)$$

Properties: it is always $\ge 0$, equals $0$ only when $p = q$, and is **not symmetric** ($D_{\mathrm{KL}}(p\Vert q) \ne D_{\mathrm{KL}}(q\Vert p)$ in general), so it is not a true distance.

Since $H(p)$ depends only on the data, not on the model, **minimising cross-entropy with respect to $q$ is the same as minimising KL divergence** between the data distribution and the model. For one-hot labels, $H(p) = 0$ and the two are identical. KL divergence reappears prominently in VAEs (Chapter 20).

**Worked example.** $p = [0.5, 0.5]$, $q = [0.9, 0.1]$:

- $H(p) = \ln 2 \approx 0.6931$
- $H(p, q) = -(0.5\ln 0.9 + 0.5\ln 0.1) = 0.0527 + 1.1513 \approx 1.2040$
- $D_{\mathrm{KL}}(p \Vert q) = 1.2040 - 0.6931 \approx 0.5108$

### 5.7.4 Practical Python: from scratch

```python
import numpy as np

def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]                          # skip zeros (convention 0 * ln 0 = 0)
    return -np.sum(p * np.log(p))

def cross_entropy(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0                          # only terms where p_k > 0 contribute
    return -np.sum(p[mask] * np.log(q[mask]))

def kl_divergence(p, q):
    return cross_entropy(p, q) - entropy(p)

print(f"{entropy([0.5, 0.5]):.4f}")
print(f"{entropy([0.9, 0.1]):.4f}")
print(f"{cross_entropy([0.5, 0.5], [0.9, 0.1]):.4f}")
print(f"{kl_divergence([0.5, 0.5], [0.9, 0.1]):.4f}")
print(f"{cross_entropy([1, 0, 0], [0.7, 0.2, 0.1]):.4f}")
print(f"{cross_entropy([1, 0, 0], [0.1, 0.8, 0.1]):.4f}")
```

| Code | Explanation |
|---|---|
| `np.asarray(p, dtype=float)` | Converts a list to a float array (no copy if it already is one) |
| `p[p > 0]` | Boolean-mask indexing (Chapter 2): drops zero-probability entries so `np.log(0)` never occurs |
| `-np.sum(p * np.log(p))` | The entropy formula, with element-wise products and a sum |
| `mask = p > 0` | Reused so that `p` and `q` are filtered at the same positions |
| `kl_divergence` | Uses $D_{\mathrm{KL}} = H(p,q) - H(p)$ |
| `f"{value:.4f}"` | Formats to 4 decimal places |

**Expected output:**

```text
0.6931
0.3251
1.2040
0.5108
0.3567
2.3026
```

### 5.7.5 Cross-entropy in PyTorch, with softmax previewed

A network's raw outputs are called **logits**: any real numbers, not yet probabilities. The **softmax** function converts logits $z_1, \dots, z_K$ into a categorical distribution:

$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

(Chapter 7 treats softmax properly.) For logits $\mathbf{z} = [2, 1, 0]$: the exponentials are $e^2 \approx 7.389$, $e^1 \approx 2.718$, $e^0 = 1$, summing to $11.107$. Dividing gives $[0.6652,\ 0.2447,\ 0.0900]$. If the true class is the first one, the loss is $-\ln 0.6652 \approx 0.4076$.

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 1.0, 0.0]])        # shape (1, 3): one sample, three classes
target = torch.tensor([0])                      # shape (1,): index of the true class

probs = torch.softmax(logits, dim=1)
print(probs)

manual_loss = -torch.log(probs[0, target[0]])
builtin_loss = F.cross_entropy(logits, target)

print(round(manual_loss.item(), 4))
print(round(builtin_loss.item(), 4))

# Binary cross-entropy: true label 1, predicted probability 0.8
bce = F.binary_cross_entropy(torch.tensor([0.8]), torch.tensor([1.0]))
print(round(bce.item(), 4))
```

| Code | Explanation |
|---|---|
| `logits`, shape `(1, 3)` | Raw scores: batch of 1 sample, 3 classes |
| `target`, shape `(1,)` | The correct class **index** (an integer, dtype `int64`), not a one-hot vector |
| `torch.softmax(logits, dim=1)` | Normalises across the class dimension (`dim=1`) so each row sums to 1 |
| `probs[0, target[0]]` | Probability the model assigned to the true class |
| `-torch.log(...)` | Negative log-likelihood of the true class |
| `F.cross_entropy(logits, target)` | PyTorch's built-in version. It **takes raw logits** and applies a (numerically stable) log-softmax internally |
| `F.binary_cross_entropy(prob, label)` | For a single yes/no output already converted to a probability: $-[y\ln p + (1-y)\ln(1-p)]$, which for $y = 1, p = 0.8$ is $-\ln 0.8$ |

**Expected output:**

```text
tensor([[0.6652, 0.2447, 0.0900]])
0.4076
0.4076
0.2231
```

**A very common mistake:** applying `softmax` yourself and then passing the result to `F.cross_entropy`. The function expects raw logits and applies the softmax internally, so doing it twice gives wrong losses without raising an error. Likewise, `target` must hold class indices (integers), not probabilities, for this standard usage.

### 5.7.6 Why cross-entropy rather than accuracy?

Accuracy counts correct predictions; it is **piecewise constant**, so its gradient is zero almost everywhere and tells the optimiser nothing about which way to move. Cross-entropy changes smoothly as probabilities change, giving the gradients that Chapter 4's descent needs. Accuracy remains the right number to *report*; cross-entropy is the right number to *train* with.

### 5.7.7 Comparison

| Quantity | Formula | Measures | Role |
|---|---|---|---|
| Entropy $H(p)$ | $-\sum p_k \ln p_k$ | Unpredictability of one distribution | Reference level; zero for one-hot labels |
| Cross-entropy $H(p, q)$ | $-\sum p_k \ln q_k$ | Average surprise using $q$ when truth is $p$ | **Classification loss** |
| KL divergence $D_{\mathrm{KL}}(p\Vert q)$ | $H(p,q) - H(p)$ | Extra surprise caused by using $q$ instead of $p$ | VAEs, distillation, regularisers |
| MSE | $\frac{1}{N}\sum (y - \hat{y})^2$ | Squared prediction error | **Regression loss** (Gaussian NLL) |

---

## 5.8 Randomness in Deep Learning

| Source of randomness | Distribution | Chapter |
|---|---|---|
| Initial weights | Normal or uniform, with scale chosen carefully | 10 |
| Shuffling training data each epoch | Uniform random permutation | 9 |
| Dropout | Bernoulli mask on activations | 10 |
| Data augmentation | Random crops, flips, noise | 13, 14 |
| Mini-batch selection (hence "stochastic" gradient descent) | Random subset of the data | 8 |
| Noise in generative models | Standard normal | 20, 22 |
| Text generation (sampling the next token) | Categorical over the vocabulary | 23 |

Controlling the seed for these draws is what makes experiments reproducible (Chapter 2).

---

## 5.9 Distributions, Generalisation, and Their Assumptions

Most of what we do assumes that training and test examples are **independent and identically distributed (i.i.d.)**: each one is an independent draw from the same underlying distribution. This assumption is what allows the dataset likelihood to be written as a product and what lets test performance predict real-world performance.

When reality violates it, for example a model trained on daytime photos deployed at night, we have **distribution shift**, and test-set numbers no longer predict real behaviour. This connects to the limitations discussed in Chapter 1 and the evaluation topics of Chapter 12.

---

## 5.10 Advantages, Limitations, and Misconceptions

### 5.10.1 Advantages of the probabilistic view

- **Principled losses:** loss functions are derived, not guessed.
- **Uncertainty:** outputs can express confidence, not only a label.
- **Unified view:** regression, classification, and generation share one framework (maximum likelihood).

### 5.10.2 Limitations

- Probabilities from a network are only as trustworthy as the model: networks can be **over-confident**, assigning $0.99$ to wrong answers. A softmax output is not automatically a calibrated probability.
- The Gaussian and categorical assumptions may not match reality (for example, heavy-tailed noise makes MSE sensitive to outliers).
- The i.i.d. assumption often fails in real deployments (Section 5.9).

### 5.10.3 Common misconceptions and mistakes

| Misconception or mistake | Correction |
|---|---|
| "Likelihood and probability are the same." | Same formula, different thing held fixed (Section 5.5) |
| "A density value is a probability." | A PDF value can exceed 1; probabilities are areas under it |
| "The softmax output is the true probability of being correct." | It is the model's score normalised to sum to 1. It can be badly miscalibrated |
| "Variance and standard deviation are interchangeable." | $\sigma$ is the square root of variance and has the original units |
| Using `np.var` and `torch.var` and expecting identical numbers | Their defaults differ ($n$ vs $n-1$); Section 5.3.3 |
| Applying softmax before `F.cross_entropy` | The loss applies it internally; pass raw logits |
| Computing normalisation statistics on all the data | Compute on the training set only, to avoid leakage |
| "Lower cross-entropy always means higher accuracy." | Usually they move together, but not always; they measure different things |
| "Cross-entropy and MSE are interchangeable for classification." | MSE implies a Gaussian assumption that does not suit class labels, and it gives weaker gradients for confident wrong predictions |

---

## 5.11 Summary of Connections

| Concept from this chapter | Where it is used later |
|---|---|
| Conditional distribution $P(y \mid \mathbf{x};\theta)$ | Every classifier (Chapters 7, 9, 13, 18) |
| Expectation | Training objective as an expected loss; batch averaging (Chapter 8) |
| Variance, standardisation | Input normalisation; initialisation scale (Chapter 10) |
| Normal distribution | Weight initialisation, noise, VAEs, diffusion (Chapters 10, 20, 22) |
| Bernoulli | Dropout (Chapter 10) |
| Categorical and softmax | Multi-class output, next-token prediction (Chapters 7, 18, 23) |
| NLL, MLE | The meaning behind every standard loss (Chapter 7) |
| Cross-entropy | Classification loss (Chapters 7 to 9) |
| KL divergence | VAEs, knowledge distillation (Chapter 20 onward) |
| i.i.d. and distribution shift | Evaluation and deployment (Chapters 12, 24) |

---

## Exercises

1. **Expectation and variance.** A game pays $0$ with probability $0.5$, $10$ with probability $0.3$, and $50$ with probability $0.2$. Compute $\mathbb{E}[X]$, $\mathrm{Var}(X)$, and $\sigma$ by hand, then check with NumPy.
2. **Conditional probability.** Roll two dice. Find $P(\text{sum} = 8)$ and $P(\text{sum} = 8 \mid \text{first die is even})$. Are the events independent? Verify by enumeration.
3. **Population vs sample.** For the data $[1, 3, 5, 7]$, compute the population variance and the sample variance by hand. Then compute both with `np.var` and `torch.var` and explain which call gives which.
4. **Standardisation.** Standardise $[10, 20, 30, 40]$ using the population standard deviation. Show that the result has mean $0$ and standard deviation $1$. Then explain why you would compute these statistics from the training set only.
5. **Likelihood.** You observe 7 heads in 10 flips. Write the likelihood $\mathcal{L}(p)$, evaluate it at $p = 0.5$, $0.7$, and $0.9$, and derive the MLE by setting the derivative of the log-likelihood to zero.
6. **Entropy.** Compute the entropy (in nats) of the distribution $[0.25, 0.25, 0.25, 0.25]$ and of $[0.7, 0.1, 0.1, 0.1]$. Which is larger, and why?
7. **Cross-entropy loss.** The true class is the second of three classes. Compute the loss for model outputs $[0.2, 0.5, 0.3]$ and $[0.05, 0.9, 0.05]$. Then verify one of them using `F.cross_entropy` with logits $\ln q$ (the log of the probabilities works as logits, since softmax of $\ln q$ returns $q$ when $q$ sums to 1).
8. **Softmax by hand.** Compute softmax for logits $[1, 2, 3]$ and the cross-entropy loss if the true class is the last one. Confirm in PyTorch.
9. **MSE and Gaussians.** In your own words, explain why minimising MSE is equivalent to maximum likelihood under Gaussian noise, and describe one situation where that assumption would be a poor fit.
10. **Underflow.** Modify the code in Section 5.6.3 to find the smallest number of examples (each with probability $0.5$) at which `np.prod` first returns exactly `0.0`. Compare with the log-sum value at that point.
