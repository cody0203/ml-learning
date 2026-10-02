## What is probability?

### Intuition
Probability starts with a repeatable experiment: pick a child, flip coins, or roll dice. The slides define an event as the part of the sample space we care about. If 3 of 10 children play soccer, the event has 3 favorable cases out of 10 possible children, so the probability is $0.3$. This is the same habit used later in ML: first say what is random, then ask how often the target event occurs.

### Definitions and formulas
The sample space $S$ contains all possible outcomes, and an event $A\subseteq S$ is a set of outcomes. When outcomes are equally likely,
$$P(A)=\frac{|A|}{|S|}.$$
The answer must be between 0 and 1. A value near 0 means rare in this setup; a value near 1 means common, not guaranteed in every short run.

### Worked example
For one fair die, $S=\{1,2,3,4,5,6\}$. The event "obtain 6" has one outcome, so $P(6)=1/6$. For two dice, $(6,6)$ is one ordered pair among 36, so $P((6,6))=1/36$.

> ⚠️ Common mistake: changing the sample space mid-solution. One die has 6 outcomes; two dice have 36 ordered outcomes.

### NumPy / SciPy snippet
```python
import numpy as np
rng = np.random.default_rng(1)
rolls = rng.integers(1, 7, size=20000)
print(np.mean(rolls == 6))
```

### Summary
- Name the experiment before computing.
- Count favorable and total outcomes.
- Use ordered pairs for two dice.

## Complement rule

### Intuition
Sometimes the event you want is messy, while the opposite event is simple. The slides ask for not getting three heads. Instead of listing seven outcomes, compute the probability of $HHH$ and subtract from 1. Complements are useful because an event and its complement partition the whole sample space.

### Definitions and formulas
The complement of $A$ is $A^c$, the outcomes where $A$ does not happen. Since exactly one of $A$ or $A^c$ occurs,
$$P(A^c)=1-P(A).$$
This works for discrete and continuous distributions.

### Worked example
Three fair coin tosses have $2^3=8$ ordered outcomes. The event $HHH$ has probability $1/8$. Therefore the probability of not landing heads three times is
$$1-\frac18=\frac78=0.875.$$

> ⚠️ Common mistake: interpreting "not $HHH$" as "all tails". It means every outcome except $HHH$.

### NumPy / SciPy snippet
```python
import numpy as np
p_hhh = 0.5 ** 3
p_not_hhh = 1 - p_hhh
print(p_not_hhh)
```

### Summary
- Complements are exhaustive and non-overlapping.
- Use them for "at least one" and "not all" questions.
- The complement of a rare event is often easy to compute.

## Sum rules for disjoint and joint events

### Intuition
"Or" means union. If soccer and basketball are mutually exclusive, counts simply add. If students can play both sports, adding raw percentages double-counts the overlap. The slides show both cases: a school where kids can play only one sport, and another where the soccer and basketball circles overlap.

### Definitions and formulas
For any two events,
$$P(A\cup B)=P(A)+P(B)-P(A\cap B).$$
When $A$ and $B$ are disjoint, $P(A\cap B)=0$, so the formula becomes $P(A\cup B)=P(A)+P(B)$.

### Worked example
In a group of 10 children, 6 play soccer, 5 play basketball, and 3 play both. The union has $6+5-3=8$ children, so $P(S\cup B)=8/10=0.8$. If the sports were exclusive with probabilities 0.3 and 0.4, the union would be 0.7.

> ⚠️ Common mistake: using the disjoint shortcut when the Venn circles overlap.

### NumPy / SciPy snippet
```python
soccer = {1, 2, 3, 4, 5, 6}
basket = {4, 5, 6, 7, 8}
print(len(soccer | basket) / 10)
```

### Summary
- Union counts outcomes in either event.
- Subtract the intersection once.
- Disjoint means the intersection is empty.

## Independence and product rule

### Intuition
Independence means learning that one event happened does not change the probability of the other. In the room-and-soccer slide, if 40% of all children play soccer and room assignment is independent, then 40% of each room should play soccer. The product rule then fills the joint cell.

### Definitions and formulas
Events $A$ and $B$ are independent when $P(B\mid A)=P(B)$, equivalently
$$P(A\cap B)=P(A)P(B).$$
The formula is not a definition of "and" in every problem; it is a test and shortcut only under independence.

### Worked example
Suppose $P(S)=0.4$ for soccer and $P(R_1)=0.3$ for room 1. If the events are independent, then
$$P(S\cap R_1)=0.4\cdot0.3=0.12.$$
In 100 children, that predicts 12 soccer players in room 1.

> ⚠️ Common mistake: assuming all pairs of events are independent because the word "and" appears.

### NumPy / SciPy snippet
```python
p_soccer = 0.4
p_room1 = 0.3
print(p_soccer * p_room1)
```

### Summary
- Independence is about unchanged conditional probability.
- Product rule needs independence.
- Dependent events need the general product rule.

## Conditional probability

### Intuition
Conditional probability shrinks the sample space to the cases where the evidence is true. The slides first ask $P(\text{sum}=10)$ for two dice, then ask the same question given the first die is 6. The second question has only six possible outcomes, not 36.

### Definitions and formulas
For $P(A)>0$,
$$P(B\mid A)=\frac{P(A\cap B)}{P(A)}.$$
The vertical bar means "given"; it changes the denominator to the probability of the condition.

### Worked example
For two dice, the event $B=\{\text{sum}=10\}$ has 3 outcomes. Let $A=\{\text{first die}=6\}$. The intersection is only $(6,4)$. Hence
$$P(B\mid A)=\frac{1/36}{6/36}=\frac16.$$
If the first die were 1, the answer would be 0.

> ⚠️ Common mistake: confusing $P(B\mid A)$ with $P(A\mid B)$.

### NumPy / SciPy snippet
```python
p_joint = 1 / 36
p_first6 = 6 / 36
print(p_joint / p_first6)
```

### Summary
- Conditioning changes the reference population.
- The numerator is the overlap.
- The denominator is the condition.

## Bayes' theorem

### Intuition
Bayes' theorem reverses a conditional probability. A medical test may be 99% effective, but if the disease is extremely rare, most positive tests can still be false positives. The slides demonstrate this with a million people: the base rate matters as much as the test accuracy.

### Definitions and formulas
For a hypothesis $A$ and evidence $B$,
$$P(A\mid B)=\frac{P(A)P(B\mid A)}{P(A)P(B\mid A)+P(A^c)P(B\mid A^c)}.$$
The numerator is the true-positive path; the denominator is all ways to see the evidence.

### Worked example
Let $P(\text{sick})=0.0001$, $P(+\mid \text{sick})=0.99$, and $P(+\mid \text{healthy})=0.01$. Then
$$P(\text{sick}\mid +)=\frac{0.0001\cdot0.99}{0.0001\cdot0.99+0.9999\cdot0.01}\approx0.0098.$$
The posterior is under 1%, even with a strong test.

> ⚠️ Common mistake: answering 99% because the test is 99% sensitive.

### NumPy / SciPy snippet
```python
prior = 0.0001
posterior = prior * 0.99 / (prior * 0.99 + (1 - prior) * 0.01)
print(posterior)
```

### Summary
- Prior is belief before evidence.
- Likelihood says how evidence behaves under a class.
- Posterior is the updated probability.

## Naive Bayes

### Intuition
Naive Bayes extends Bayes' theorem to many features by making a simplifying assumption: words are conditionally independent given the class. The spam slides use "lottery" and "winning" to show how multiplying word likelihoods can make a spam posterior much larger than using either word alone.

### Definitions and formulas
For class $C$ and words $w_i$,
$$P(C\mid w_1,\ldots,w_n)\propto P(C)\prod_i P(w_i\mid C).$$
The denominator normalizes the scores across classes.

### Worked example
Use $P(spam)=0.2$, $P(lottery\mid spam)=0.7$, $P(winning\mid spam)=0.75$, $P(lottery\mid ham)=0.125$, and $P(winning\mid ham)=0.1$. The spam score is $0.105$ and the ham score is $0.01$, so the posterior is $0.105/(0.115)\approx0.913$.

> ⚠️ Common mistake: believing the naive assumption is literally true. It is a practical approximation.

### NumPy / SciPy snippet
```python
spam_score = 0.2 * 0.7 * 0.75
ham_score = 0.8 * 0.125 * 0.1
print(spam_score / (spam_score + ham_score))
```

### Summary
- Multiply likelihoods within each class.
- Normalize class scores to get probabilities.
- The model is simple but often effective.

## Probability in machine learning

### Intuition
ML models often output probabilities rather than hard facts. An image classifier might report $P(cat\mid pixels)=0.9$, a patient model might estimate $P(healthy\mid symptoms)$, and sentiment analysis estimates $P(happy\mid words)$. These are conditional probabilities learned from data.

### Definitions and formulas
Supervised classification often seeks
$$P(y\mid x),$$
the probability of a label $y$ given features $x$. Generative models can also ask whether generated pixels have high probability under a desired class.

### Worked example
If a cat detector returns probabilities 0.9 for cat and 0.1 for not-cat, the predicted class is cat. The number 0.9 should be read as model confidence under its training assumptions, not as a proof that the image contains a cat.

> ⚠️ Common mistake: treating a probability score as calibrated truth without validation.

### NumPy / SciPy snippet
```python
probs = {"cat": 0.9, "not cat": 0.1}
print(max(probs, key=probs.get))
```

### Summary
- Classification uses conditional probabilities.
- Calibration matters for decision thresholds.
- Probability also drives sampling in generative models.

## Random variables

### Intuition
A random variable converts messy outcomes into numbers. Instead of storing every sequence of coin flips, the slides define $X$ as the number of heads. Then $X=0,1,\ldots,10$ summarizes a 10-toss experiment.

### Definitions and formulas
A random variable is a function from outcomes to real values:
$$X:S\to\mathbb R.$$
Discrete random variables take countable values; continuous ones range over intervals.

### Worked example
For one coin toss, define $X=1$ for heads and $X=0$ for tails. Then $P(X=1)=0.5$ and $P(X=0)=0.5$. For ten tosses, $X$ can be any integer from 0 to 10.

> ⚠️ Common mistake: thinking the random variable is the outcome itself. It is a numeric summary of the outcome.

### NumPy / SciPy snippet
```python
rng = np.random.default_rng(2)
heads = rng.binomial(n=10, p=0.5, size=6)
print(heads)
```

### Summary
- Random variables map outcomes to numbers.
- They let one formula describe many outcomes.
- Counts are usually discrete.

## Discrete distributions and PMFs

### Intuition
A PMF assigns probability mass to each possible value of a discrete random variable. The slides build the PMF for the number of heads in three, four, and five coin tosses. The bars add to 1 because one of the possible counts must happen.

### Definitions and formulas
For a discrete variable,
$$p_X(x)=P(X=x),\qquad \sum_x p_X(x)=1.$$
Each mass must be non-negative.

### Worked example
For three fair tosses, the counts of heads have probabilities $1/8,3/8,3/8,1/8$ for $0,1,2,3$. Thus $P(X\ge2)=3/8+1/8=1/2$.

> ⚠️ Common mistake: drawing PMF bars like a continuous curve and reading area instead of bar height.

### NumPy / SciPy snippet
```python
from math import comb
pmf = [comb(3, k) / 8 for k in range(4)]
print(pmf, sum(pmf))
```

### Summary
- PMF bars are probabilities.
- All bars must sum to 1.
- Count variables are natural PMF examples.

## Bernoulli and binomial distributions

### Intuition
Bernoulli is one success/failure trial. Binomial counts successes across $n$ independent Bernoulli trials with the same success probability $p$. The slides turn "rolling a one" into a biased coin with $p=1/6$.

### Definitions and formulas
If $X\sim Bernoulli(p)$, then $P(X=1)=p$ and $P(X=0)=1-p$. If $X\sim Binomial(n,p)$,
$$P(X=k)=\binom nk p^k(1-p)^{n-k}.$$

### Worked example
For five fair coin tosses, the probability of exactly two heads is
$$\binom52(0.5)^2(0.5)^3=10/32=0.3125.$$
For five dice rolls, exactly three ones uses $p=1/6$.

> ⚠️ Common mistake: forgetting the binomial coefficient and counting only one order.

### NumPy / SciPy snippet
```python
from math import comb
print(comb(5, 2) * 0.5**5)
```

### Summary
- Bernoulli models one trial.
- Binomial counts successes across trials.
- The coefficient counts all orders.

## Continuous variables and PDFs

### Intuition
Waiting time can be 1.01, 2.43, or infinitely many other values. For a continuous variable, the probability of exactly one point is zero. Probabilities come from intervals, drawn as area under a density curve.

### Definitions and formulas
A PDF $f_X$ satisfies $f_X(x)\ge0$ and total area 1. For an interval,
$$P(a<X<b)=\int_a^b f_X(x)\,dx.$$
The height $f_X(x)$ is a density, not a probability.

### Worked example
If waiting time is uniform from 0 to 5 minutes, the density is $1/5$. The probability of waiting between 2 and 3 minutes is the rectangle area $1\cdot(1/5)=0.2$.

> ⚠️ Common mistake: saying $P(X=2)=f_X(2)$. For continuous $X$, point probability is 0.

### NumPy / SciPy snippet
```python
x = np.linspace(0, 5, 501)
area = np.trapz(np.full_like(x, 0.2), x)
print(area)
```

### Summary
- Continuous probability is area.
- Exact point probabilities are zero.
- A PDF can be greater than 1 on narrow intervals.

## Cumulative distribution functions

### Intuition
The CDF accumulates probability from the far left up to a chosen value. The slides show a curve that starts at 0, never decreases, and eventually reaches 1. CDFs work for both discrete and continuous variables.

### Definitions and formulas
The cumulative distribution function is
$$F_X(x)=P(X\le x).$$
It satisfies $0\le F_X(x)\le1$ and is non-decreasing.

### Worked example
For $X\sim Uniform(0,4)$, the CDF at 2 is $(2-0)/(4-0)=0.5$. The probability $P(1<X\le3)$ is $F(3)-F(1)=3/4-1/4=1/2$.

> ⚠️ Common mistake: accepting a decreasing graph as a CDF.

### NumPy / SciPy snippet
```python
def uniform_cdf(x):
    return np.clip(x / 4, 0, 1)
print(uniform_cdf(2), uniform_cdf(3) - uniform_cdf(1))
```

### Summary
- CDF means probability up to $x$.
- Differences of CDF values give interval probabilities.
- A valid CDF never goes down.

## Uniform distribution

### Intuition
Uniform means every equal-length interval inside the support has the same probability. The tech-support example assumes a response can arrive any time between 0 and 15 minutes with no preferred subinterval.

### Definitions and formulas
For $X\sim Uniform(a,b)$,
$$f_X(x)=\frac1{b-a}\quad a<x<b,$$
and the CDF rises linearly from 0 to 1 across the interval.

### Worked example
If $T\sim Uniform(0,15)$, the density height is $1/15$. The probability of waiting between 6 and 9 minutes is $(9-6)/15=0.2$.

> ⚠️ Common mistake: using the endpoint value as a probability. Only interval length matters.

### NumPy / SciPy snippet
```python
a, b = 0, 15
print((9 - 6) / (b - a))
```

### Summary
- Constant density over a finite interval.
- Probability equals interval length divided by total length.
- Outside the interval the density is zero.

## Normal distribution

### Intuition
The normal distribution is the bell curve seen in heights, noise, IQ-like scores, and sums of many small independent effects. The slides show how $\mu$ shifts the center and $\sigma$ changes spread.

### Definitions and formulas
If $X\sim\mathcal N(\mu,\sigma^2)$,
$$f_X(x)=\frac1{\sigma\sqrt{2\pi}}\exp\left[-\frac12\left(\frac{x-\mu}{\sigma}\right)^2\right].$$
Standardization uses $Z=(X-\mu)/\sigma$.

### Worked example
For $X\sim\mathcal N(10,2^2)$, the value $x=14$ has $z=(14-10)/2=2$. It is two standard deviations above the mean; software or tables are used for exact areas.

> ⚠️ Common mistake: reading normal probabilities directly from the PDF height instead of integrating or using a CDF.

### NumPy / SciPy snippet
```python
from math import erf, sqrt
z = (14 - 10) / 2
Phi = 0.5 * (1 + erf(z / sqrt(2)))
print(Phi)
```

### Summary
- $\mu$ controls center.
- $\sigma$ controls spread.
- Areas require CDFs or numerical tools.

## Chi-squared distribution

### Intuition
The chi-squared distribution appears when squared standard-normal noise terms are added. The slides motivate it with communication-channel noise power: if $Z$ is standard normal noise, then power is $Z^2$, always non-negative.

### Definitions and formulas
If $Z_1,\ldots,Z_k$ are independent standard normal variables, then
$$W=\sum_{i=1}^k Z_i^2\sim \chi^2_k.$$
The parameter $k$ is the degrees of freedom.

### Worked example
If two transmissions have standardized noise values $1$ and $-2$, accumulated power is $1^2+(-2)^2=5$. With random standard normal terms, that sum follows a chi-squared distribution with 2 degrees of freedom.

> ⚠️ Common mistake: expecting negative chi-squared values. Squares make the support start at 0.

### NumPy / SciPy snippet
```python
rng = np.random.default_rng(4)
z = rng.normal(size=(10000, 2))
w = np.sum(z**2, axis=1)
print(w.mean())
```

### Summary
- Chi-squared values are sums of squares.
- Degrees of freedom count squared normal terms.
- It models accumulated noise power.

## Sampling from a distribution

### Intuition
Sampling turns probabilities into simulated outcomes. The slides divide the interval $[0,1]$ into cumulative probability ranges: green gets $[0,0.3)$, blue gets $[0.3,0.8)$, and orange gets $[0.8,1]$.

### Definitions and formulas
For a discrete PMF, cumulative probabilities define intervals whose lengths equal the probabilities. Draw $u\sim Uniform(0,1)$ and return the category whose interval contains $u$.

### Worked example
With probabilities green 0.3, blue 0.5, orange 0.2, a random number $u=0.62$ falls in $[0.3,0.8)$, so the sampled color is blue. A value $u=0.91$ would select orange.

> ⚠️ Common mistake: using unequal intervals for categories that should have specified probabilities.

### NumPy / SciPy snippet
```python
probs = np.array([0.3, 0.5, 0.2])
u = 0.62
print(np.searchsorted(np.cumsum(probs), u, side="right"))
```

### Summary
- Random numbers become outcomes through intervals.
- CDFs make sampling systematic.
- Simulations should match PMFs in large samples.
