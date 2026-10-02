# Week 4: Confidence intervals and tests

The PDF moves from estimation to decisions:
known-$\sigma$ intervals, margin of error,
sample size, interpretation, unknown $\sigma$,
Student's $t$, proportions, hypotheses,
error types, tails, p-values, critical values,
power, one-sample and two-sample $t$ tests,
paired tests, and A/B testing.

## Confidence intervals with known σ

### Intuition

A sample mean is a noisy snapshot.
If the population standard deviation is known,
the Central Limit Theorem makes the noise
quantifiable:
$\bar X$ is centered at $\mu$ and has
standard error $\sigma/\sqrt n$.
So an interval is not a magic guarantee.
It is a calibrated recipe.

### Definitions and formulas

For confidence level $1-\alpha$,

$$
\bar x \pm z_{\alpha/2}
\frac{\sigma}{\sqrt n}.
$$

The value $z_{\alpha/2}$ leaves area
$\alpha/2$ in each normal tail.
Common values are:
90% $\to 1.645$,
95% $\to 1.96$,
99% $\to 2.576$.

### Fully worked example

The slides use Statistopia heights.
Suppose $n=49$, $\bar x=170$ cm,
and the known population standard deviation is
$\sigma=25$ cm.
For 95% confidence, use $z=1.96$.

$$
SE=\frac{25}{\sqrt{49}}=\frac{25}{7}=3.571.
$$

$$
MOE=1.96(3.571)=7.00.
$$

Thus the interval is

$$
170\pm 7=(163,177).
$$

We estimate the average adult height
in Statistopia to be between 163 and 177 cm.

### Python snippet

```python
import math

xbar = 170
sigma = 25
n = 49
z = 1.96
me = z * sigma / math.sqrt(n)
print(xbar - me, xbar + me)
```

> ⚠️ Common mistake:
> using $s$ and a $t$ critical value when the
> problem explicitly says $\sigma$ is known.

### Summary

- Known $\sigma$ leads to a z interval.
- Width is controlled by $z$, $\sigma$, and $n$.
- The estimate is $\bar x$; the parameter is $\mu$.

## Margin of error, sample size, and confidence level

### Intuition

The margin of error is half the interval width.
It is the price paid for uncertainty.
A stricter confidence level requires
a larger critical value.
A larger sample size makes the average
less variable, but only at a square-root rate.

### Definitions and formulas

For a known-$\sigma$ mean interval,

$$
MOE=z_{\alpha/2}\frac{\sigma}{\sqrt n}.
$$

To plan a target margin $m$:

$$
n\ge
\left(\frac{z_{\alpha/2}\sigma}{m}\right)^2.
$$

Always round the result upward.

### Fully worked example

The initial Statistopia interval had margin 7 cm.
Now require margin at most 3 cm,
still with $\sigma=25$ and 95% confidence.

$$
n\ge
\left(\frac{1.96\cdot25}{3}\right)^2
=266.78.
$$

The smallest integer sample size is 267.
Rounding down to 266 would miss the target.

### Python snippet

```python
import math

sigma = 25
z = 1.96
target = 3
n = math.ceil((z * sigma / target) ** 2)
print(n)
```

> ⚠️ Common mistake:
> doubling $n$ does not halve the margin.
> To halve margin, multiply $n$ by about four.

### Summary

- Higher confidence widens intervals.
- Larger $n$ narrows intervals.
- Sample-size formulas must be rounded up.

## Interpreting confidence intervals

### Intuition

Confidence describes a procedure,
not the chance that a fixed number moved.
After the data are observed,
the interval endpoints are fixed.
The true mean is also fixed.
The interval either covers it or misses it.

### Definitions and formulas

A 95% confidence procedure means:
if we repeated sampling many times,
about 95% of the intervals would contain $\mu$.

It does not mean:
95% of people lie inside the interval,
or $P(\mu\in[L,U])=0.95$ after seeing data.

### Fully worked example

Imagine building 100 intervals for the same
population mean, each from a new random sample.
If the method is a valid 95% method,
we expect about 95 intervals to cover $\mu$.
The actual number could be 93 or 98.
Randomness affects the long-run count.

For one reported interval, say $(163,177)$,
we should say:
"This method has 95% long-run coverage,
and this realized interval is our estimate."

### Python snippet

```python
import numpy as np

rng = np.random.default_rng(7)
mu, sigma, n = 10, 2, 40
hits = 0
for _ in range(2000):
    sample = rng.normal(mu, sigma, n)
    xbar = sample.mean()
    me = 1.96 * sigma / np.sqrt(n)
    hits += (xbar - me <= mu <= xbar + me)
print(hits / 2000)
```

> ⚠️ Common mistake:
> treating a confidence interval for a mean
> as a prediction interval for individuals.

### Summary

- Confidence is long-run coverage.
- A fixed interval has no moving probability.
- The parameter, not individual observations,
  is the target.

## t confidence intervals and proportions

### Intuition

When $\sigma$ is unknown, the sample standard
deviation $s$ adds extra uncertainty.
Student's $t$ distribution has heavier tails
to account for that estimation.
For binary outcomes, the parameter is a
proportion, so the variance comes from
Bernoulli trials.

### Definitions and formulas

Mean with unknown $\sigma$:

$$
\bar x\pm t_{\alpha/2,n-1}\frac{s}{\sqrt n}.
$$

Single proportion:

$$
\hat p\pm z_{\alpha/2}
\sqrt{\frac{\hat p(1-\hat p)}{n}}.
$$

### Fully worked example

For the height sample, take
$n=10$, $\bar x=68.442$, $s=3.113$.
With $df=9$, $t^*=2.262$ for 95%.

$$
SE=3.113/\sqrt{10}=0.984.
$$

$$
MOE=2.262(0.984)=2.227.
$$

The t interval is $(66.215,70.669)$.

For a proportion, if 24 of 30 people prefer
a design, $\hat p=0.8$.
The 95% margin is
$1.96\sqrt{0.8(0.2)/30}=0.143$,
so the interval is about $(0.657,0.943)$.

### Python snippet

```python
import math
from scipy import stats

xbar, s, n = 68.442, 3.113, 10
tcrit = stats.t.ppf(0.975, df=n-1)
print(xbar - tcrit*s/math.sqrt(n))
print(xbar + tcrit*s/math.sqrt(n))
```

> ⚠️ Common mistake:
> using the same formula for means and
> proportions just because both are intervals.

### Summary

- Unknown $\sigma$ uses $t$ and $df=n-1$.
- Proportions use $\hat p(1-\hat p)$.
- Heavy tails make small-sample t intervals wider.

## Hypotheses, Type I/II errors, and significance

### Intuition

A hypothesis test starts with a default story.
The null hypothesis $H_0$ is the baseline.
The alternative $H_1$ is the claim that needs
evidence.  In the spam example, regular mail
is the default; spam requires evidence.

### Definitions and formulas

Type I error:
reject a true $H_0$.
Type II error:
do not reject a false $H_0$.

The significance level is

$$
\alpha=\max P(\text{reject }H_0
\mid H_0\text{ true}).
$$

### Fully worked example

Toss a coin 10 times and see 8 heads.
Let $H_0:p=0.5$ and $H_1:p>0.5$.
Under the null,

$$
P(X\ge 8)=
\frac{\binom{10}{8}+\binom{10}{9}
+\binom{10}{10}}{2^{10}}
=0.0547.
$$

At $\alpha=0.05$, this is not quite small
enough to reject.
With 80 heads in 100 tosses,
the same null gives a tiny tail probability,
so rejection is clear.

### Python snippet

```python
from math import comb

p = sum(comb(10, k) for k in range(8, 11)) / 2**10
print(p)
```

> ⚠️ Common mistake:
> saying "accept $H_0$" after a large p-value.
> The safer phrase is "fail to reject $H_0$".

### Summary

- $H_0$ is the default model.
- $\alpha$ limits false positives.
- Type II error is a missed real effect.

## One/two-tailed tests, p-values, and critical values

### Intuition

The alternative decides what counts as extreme.
If $H_1:\mu>\mu_0$, only the right tail
supports the claim.
If $H_1:\mu\ne\mu_0$, both tails matter.
The p-value measures how far into the
relevant tail the observed statistic lies.

### Definitions and formulas

Known-$\sigma$ mean test:

$$
z=\frac{\bar x-\mu_0}{\sigma/\sqrt n}.
$$

Right tail:
$p=P(Z\ge z_{obs})$.
Left tail:
$p=P(Z\le z_{obs})$.
Two tails:
$p=2P(Z\ge |z_{obs}|)$.

### Fully worked example

Use the height example with
$\bar x=68.442$, $\mu_0=66.7$,
$\sigma=3$, and $n=10$.

$$
z=\frac{68.442-66.7}{3/\sqrt{10}}=1.837.
$$

For $H_1:\mu>66.7$,
the p-value is about 0.0407,
so a 5% test rejects.
For $H_1:\mu\ne66.7$,
the p-value doubles to about 0.0814,
so the same data do not reject.

### Python snippet

```python
import math
from scipy import stats

z = (68.442 - 66.7) / (3 / math.sqrt(10))
print(stats.norm.sf(z))
print(2 * stats.norm.sf(abs(z)))
```

> ⚠️ Common mistake:
> choosing the tail after looking at whether
> the observed mean went up or down.

### Summary

- The alternative fixes the tail.
- p-value and critical-value rules agree.
- Two-tailed tests need stronger evidence.

## Power and interpreting test results

### Intuition

Power asks whether the test can detect
an effect that truly exists.
A tiny p-value is not the only important
number.  A non-significant result is also
hard to read if the study was underpowered.

### Definitions and formulas

For a true alternative value $\theta$,

$$
\beta(\theta)=P(\text{fail to reject }H_0
\mid \theta),
$$

and

$$
\text{power}(\theta)=1-\beta(\theta).
$$

### Fully worked example

Suppose a right-tailed height test rejects
when $\bar X>68.26$.
Let $\sigma=3$ and $n=10$.
If the true mean is actually 70, then
$\bar X$ has standard error $3/\sqrt{10}$.

$$
z=\frac{68.26-70}{3/\sqrt{10}}=-1.834.
$$

Thus
$\beta=P(\bar X\le 68.26\mid\mu=70)=0.0333$.
Power is $1-0.0333=0.9667$.

### Python snippet

```python
import math
from scipy import stats

crit = 68.26
mu_true = 70
se = 3 / math.sqrt(10)
beta = stats.norm.cdf((crit - mu_true) / se)
print(beta, 1 - beta)
```

> ⚠️ Common mistake:
> reporting "no effect" from a large p-value
> without checking the detectable effect size.

### Summary

- Power is sensitivity under the alternative.
- Larger $n$ and smaller noise increase power.
- Lower $\alpha$ usually lowers power.

## t-tests, proportion tests, and A/B testing

### Intuition

Different experimental designs need
different statistics.
One sample compares a mean to a baseline.
Two independent samples compare groups.
Paired samples compare within-unit changes.
A/B testing uses these ideas for product
metrics such as spend or conversion.

### Definitions and formulas

One-sample t statistic:

$$
t=\frac{\bar x-\mu_0}{s/\sqrt n}.
$$

Welch two-sample statistic:

$$
t=\frac{\bar x-\bar y-\Delta_0}
{\sqrt{s_x^2/n_x+s_y^2/n_y}}.
$$

Two-proportion z statistic under
$H_0:p_A=p_B$:

$$
z=\frac{\hat p_A-\hat p_B}
{\sqrt{\hat p(1-\hat p)(1/n_A+1/n_B)}}.
$$

### Fully worked example

For conversion A/B testing,
design A has $x_A=20$ conversions
from $n_A=80$ users.
Design B has $x_B=8$ from $n_B=20$.
Then $\hat p_A=0.25$ and $\hat p_B=0.40$.
Under $H_0:p_A=p_B$,
the pooled estimate is $28/100=0.28$.

$$
SE=\sqrt{0.28(0.72)(1/80+1/20)}=0.1123.
$$

$$
z=\frac{0.25-0.40}{0.1123}=-1.336.
$$

For $H_1:p_A-p_B<0$,
the p-value is about 0.091.
At 5%, there is not enough evidence
that B converts better than A.

### Python snippet

```python
import math
from scipy import stats

xA, nA, xB, nB = 20, 80, 8, 20
pA, pB = xA/nA, xB/nB
pool = (xA + xB) / (nA + nB)
se = math.sqrt(pool*(1-pool)*(1/nA + 1/nB))
z = (pA - pB) / se
print(z, stats.norm.cdf(z))
```

> ⚠️ Common mistake:
> treating paired before/after observations
> as independent groups and losing the pairing.

### Summary

- Match the test to the data design.
- Welch handles unequal variances better.
- A/B tests need both statistical and
  practical significance.
