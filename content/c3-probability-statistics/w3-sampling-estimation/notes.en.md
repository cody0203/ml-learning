# Week 3: Sampling and point estimates

This week moves from sampling to parameter estimation: population/sample, sample statistics, LLN, CLT, point estimation, MLE, Bayesian/MAP, and regularization.

## Population, samples, and random sampling

### Intuition
Statistics starts by separating the **population** we care about from the **sample** we can actually observe. In the slides, Statistopia has 10,000 people; asking everyone gives a population mean, while asking a subset gives an estimate. In machine learning, every training set is a sample from a larger world of future examples, so sampling quality affects generalization.

### Definitions and formulas
A population is the entire group of individuals or elements that share the behavior of interest. A sample is the subset used to draw conclusions about the population. Population size is usually $N$; sample size is $n$. A random sample tries to avoid systematic preference. Independent observations mean one selected value does not determine another. Identically distributed observations mean each draw follows the same distribution.

For i.i.d. data $X_1,\dots,X_n$, each $X_i$ is independent and has the same distribution as $X$. This condition is the default assumption behind the law of large numbers, CLT, MLE derivations, and many ML train/test stories.

### Worked examples
If the study asks, “What is the price of avocados sold in the United States?”, the population is all avocados sold in the U.S. If you record avocados sold in four chosen stores, those avocados are the sample. A sample from only luxury stores may be random within those stores but not representative of the target population.

A useful diagnostic is to ask: ?If I repeated the sampling process tomorrow, would each unit of the target population have a defensible chance to appear?? If the answer is no, the sample may still be useful, but the conclusion should be limited. For example, a model trained only on daytime road images should not be evaluated as if it represented night, rain, and snow equally.

Independence is also a modeling statement. Sampling without replacement from a tiny population creates mild dependence, while drawing from a huge population is often close enough to independent. Identically distributed means the mechanism is stable: if half the sample is collected from one country and half from another with different behavior, the pooled data may violate this assumption.

### NumPy snippet
```python
import numpy as np
rng = np.random.default_rng(7)
population = np.arange(151, 171, 2)
sample = rng.choice(population, size=4, replace=False)
print(sample, sample.mean(), population.mean())
```

> ⚠️ Common mistakes: treating the training set as the population; using a convenient sample and calling it random; or forgetting that “independent” and “identically distributed” are two different requirements.

### Summary
- Population = full target group; sample = observed subset.
- Good sampling is about representation, independence, and identical distribution.
- ML datasets are samples, so sample bias becomes model bias.

## Sample mean, sample proportion, and sample variance

### Intuition
Because we usually cannot measure every unit, we compute sample summaries that estimate population quantities. The slides compare the population mean $\mu$ with sample means $\bar x$, the population proportion $p$ with sample proportion $\hat p$, and the population variance $\sigma^2$ with sample variance $s^2$.

### Definitions and formulas
The sample mean is
$$\bar x=\frac1n\sum_{i=1}^n x_i.$$
The sample proportion for a binary characteristic is
$$\hat p=\frac{x}{n},$$
where $x$ is the number of sample items with the characteristic. The sample variance used to estimate population variance is
$$s^2=\frac{1}{n-1}\sum_{i=1}^n (x_i-\bar x)^2.$$
The denominator $n-1$ is Bessel's correction: because $\bar x$ is fitted from the same data, the deviations have one lost degree of freedom.

### Worked examples
For Statistopia heights $151,153,\dots,169$, the population mean is $160$. A sample $153,155,159,163,165,169$ has $\bar x=160.667$, while $151,153,155,157,159,161$ has $\bar x=156$. For data $2,3,4,5,6$, $\bar x=4$, squared deviations sum to $10$, so the sample variance is $10/(5-1)=2.5$.

The three summaries answer different questions. The mean estimates a center for numerical values, the proportion estimates a success probability for binary values, and the variance estimates spread. The same dataset can have a reasonable mean but a misleading variance if outliers or subgroups are ignored.

The $n-1$ denominator is not a cosmetic detail. After computing $\bar x$, the deviations must sum to zero, so only $n-1$ deviations are free to vary. Dividing by $n$ tends to underestimate population variance for small samples; this is exactly what the slide tables demonstrate by averaging all possible size-2 samples.

### NumPy snippet
```python
import numpy as np
x = np.array([2, 3, 4, 5, 6])
print(x.mean(), x.var(ddof=1), x.var(ddof=0))
```

> ⚠️ Common mistakes: dividing sample variance by $n$ when estimating population variance; confusing $\hat p$ with a probability known exactly; or comparing sample means without considering sample size and sampling design.

### Summary
- $\bar x$ estimates $\mu$; $\hat p$ estimates $p$.
- Sample variance for estimation divides by $n-1$.
- Different samples produce different point estimates.

## Law of large numbers

### Intuition
The law of large numbers explains why averages stabilize. In the slides, rolling a four-sided die many times makes the running average move toward the population mean $2.5$. Individual rolls remain noisy, but their average becomes reliable.

### Definitions and formulas
For i.i.d. observations with finite mean $\mu=\mathbb E[X]$,
$$\bar X_n=\frac1n\sum_{i=1}^n X_i \to \mu$$
as $n$ becomes large. The slides state the practical conditions: randomly drawn sample, sufficiently large sample size, and independent observations.

### Worked examples
For a fair four-sided die with outcomes $1,2,3,4$, the mean is $(1+2+3+4)/4=2.5$. A two-roll average might be $4$ or $1$, but over thousands of rolls the average is usually close to $2.5$.

LLN is about repeated averaging, not about certainty. If the true mean is $2.5$, a sample average of $2.7$ after 100 rolls is not a contradiction; it is a finite-sample fluctuation. What changes with $n$ is that large deviations become less common.

For ML, LLN is the reason empirical training metrics can approximate population metrics when examples are representative and numerous. It is also a warning: more data from the wrong distribution only stabilizes the wrong answer.

### NumPy snippet
```python
import numpy as np
rng = np.random.default_rng(3)
rolls = rng.integers(1, 5, size=5000)
running = np.cumsum(rolls) / np.arange(1, len(rolls)+1)
print(running[-1])
```

> ⚠️ Common mistakes: thinking the next few observations must compensate for earlier ones (gambler's fallacy); expecting exact equality at finite $n$; or applying LLN to strongly dependent samples without checking assumptions.

### Summary
- LLN is about convergence of sample averages.
- Larger $n$ reduces random fluctuation of the average.
- It does not make individual observations less random.

## Central limit theorem

### Intuition
The central limit theorem describes the **shape** of sums or averages. The slides show two versions: counts of heads in many coin flips become bell-shaped, and averages of Uniform$(0,15)$ waiting times become approximately normal even though one wait time is uniform.

### Definitions and formulas
If $X_1,\dots,X_n$ are i.i.d. with mean $\mu$ and variance $\sigma^2$, then for large $n$,
$$\bar X_n \approx \mathcal N\left(\mu,\frac{\sigma^2}{n}\right),$$
or standardized,
$$\frac{\bar X_n-\mu}{\sigma/\sqrt n}\Rightarrow \mathcal N(0,1).$$
For $X\sim\text{Binomial}(n,p)$, $\mu=np$ and $\sigma^2=np(1-p)$. For $X\sim U(0,15)$, $\mu=7.5$ and $\sigma^2=18.75$, so the average of $n$ waits has variance $18.75/n$.

### Worked examples
For 10 fair coin flips, the number of heads has mean $5$ and variance $2.5$, and the distribution is already more bell-shaped than a single flip. For averages of 25 uniform wait times, $\mathbb E[\bar X]=7.5$ and $\operatorname{sd}(\bar X)=\sqrt{18.75/25}\approx0.866$.

The CLT adds a shape approximation on top of averaging. LLN says the mean moves toward $\mu$; CLT says the scaled error $\bar X_n-\mu$ has an approximately normal shape. This is why standard errors and z-scores appear throughout statistical inference.

For bounded variables such as Uniform$(0,15)$, the approximation often becomes good quickly. For skewed or heavy-tailed variables, ?large enough? may be much larger. Always identify whether you are approximating a count, a sum, or an average, because their means and variances differ by factors of $n$.

### NumPy snippet
```python
import numpy as np
rng = np.random.default_rng(4)
means = rng.uniform(0, 15, size=(4000, 25)).mean(axis=1)
print(means.mean(), means.std())
```

> ⚠️ Common mistakes: saying “the data become normal” instead of “the sampling distribution of sums/means becomes approximately normal”; forgetting the standard error $\sigma/\sqrt n$; or using CLT with tiny $n$ and extreme heavy tails.

### Summary
- CLT is about sums/averages, not raw data.
- The average keeps mean $\mu$ and has variance $\sigma^2/n$.
- It justifies normal approximations for many estimators.

## Point estimation

### Intuition
A point estimate compresses sample data into one number used to approximate an unknown population or model parameter. The slides use $\bar x$ for a population mean, $s^2$ for a variance, $7/10$ for a coin's head probability, and $(\beta_0,\beta_1)$ for a regression line.

### Definitions and formulas
A parameter is a fixed but unknown quantity in the population or model, such as $\mu$, $\sigma^2$, $p$, or regression coefficients. A statistic is computed from data. A point estimator is the rule; a point estimate is the realized number. Examples:
$$\hat\mu=\bar X,\qquad \hat p=\frac1n\sum_i X_i,\qquad s^2=\frac1{n-1}\sum_i(X_i-\bar X)^2.$$

### Worked examples
If 10 tosses produce 7 heads, the point estimate of $P(H)$ is $\hat p=0.7$. If the heights are the 10 values from the slides, the estimated Gaussian mean is their average, $68.442$.

A point estimate is intentionally incomplete: it gives a best single guess under a chosen rule. Two estimators can target the same parameter but behave differently. One may be unbiased but noisy; another may be slightly biased but lower variance. This tradeoff appears later in regularization.

When reporting a point estimate, include the sampling story. The number $0.43$ from a survey is not just a property of users; it is also a property of the survey design, non-response pattern, and sample size.

### NumPy snippet
```python
import numpy as np
x = np.array([66.75,70.24,67.19,67.09,63.65,64.64,69.81,69.79,73.52,71.74])
print(x.mean(), x.var(ddof=1), x.var(ddof=0))
```

> ⚠️ Common mistakes: treating one point estimate as the true parameter; not reporting uncertainty; or mixing up estimator (random rule) and estimate (observed value).

### Summary
- Point estimates are single-number summaries for unknown parameters.
- They vary from sample to sample.
- Good estimators are judged by bias, variance, consistency, and fit to the modeling goal.

## MLE for Bernoulli

### Intuition
Maximum likelihood estimation asks: which parameter value would make the observed data most likely? In the slides, popcorn on the floor is more likely after a movie than a nap, and a coin with $p=0.7$ explains 8 heads and 2 tails better than a coin with $p=0.3$.

### Definitions and formulas
For i.i.d. Bernoulli data $x_i\in\{0,1\}$,
$$L(p;x)=\prod_{i=1}^n p^{x_i}(1-p)^{1-x_i}=p^{k}(1-p)^{n-k},$$
where $k=\sum_i x_i$. The log-likelihood is
$$\ell(p)=k\log p+(n-k)\log(1-p).$$
Setting the derivative to zero gives
$$\hat p_{\text{MLE}}=\frac{k}{n}=\bar x.$$

### Worked examples
For 8 heads and 2 tails, $L(p)=p^8(1-p)^2$. Candidate likelihoods: $0.7^8 0.3^2\approx0.00519$, $0.5^{10}\approx0.00098$, and $0.3^8 0.7^2\approx0.000032$. The continuous maximizer is $\hat p=8/10=0.8$.

A likelihood curve is not required to integrate to one over $p$. It is a scoring function for candidate parameters after the data are fixed. This is why multiplying the likelihood by a positive constant does not change the MLE.

The log-likelihood has the same maximizer because log is increasing. For Bernoulli data, log turns $p^k(1-p)^{n-k}$ into a sum of two terms, making the derivative transparent. At the boundaries, if all observations are heads, the MLE is $p=1$; if all are tails, it is $p=0$.

### NumPy snippet
```python
import numpy as np
x = np.array([1,1,1,1,1,1,1,1,0,0])
p_hat = x.mean()
loglik = x.sum()*np.log(p_hat) + (len(x)-x.sum())*np.log(1-p_hat)
print(p_hat, loglik)
```

> ⚠️ Common mistakes: maximizing $P(p\mid\text{data})$ while calling it MLE; confusing likelihood as a function of $p$ with probability of the parameter; or multiplying tiny probabilities without using logs.

### Summary
- Likelihood is $P(\text{data}\mid\text{parameter})$ viewed as a function of the parameter.
- Bernoulli MLE is the sample proportion.
- Log-likelihood is easier and numerically safer.

## MLE for Gaussian models and linear regression

### Intuition
For Gaussian data, MLE chooses the curve that puts high density near the observed points. The slides compare Gaussians with different means and variances, then connect linear regression to likelihood: if vertical residuals are Gaussian, maximizing likelihood is the same as minimizing squared error.

### Definitions and formulas
For $x_i\sim\mathcal N(\mu,\sigma^2)$,
$$L(\mu,\sigma)=\prod_i \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right).$$
With $\sigma$ fixed, maximizing likelihood gives $\hat\mu=\bar x$. If variance is also estimated by MLE,
$$\hat\sigma^2_{\text{MLE}}=\frac1n\sum_i(x_i-\bar x)^2.$$
For linear regression with $y_i=mx_i+b+\epsilon_i$ and $\epsilon_i\sim\mathcal N(0,\sigma^2)$, MLE minimizes $\sum_i (y_i-(mx_i+b))^2$.

### Worked examples
The slide heights have mean $68.442$. With the Gaussian MLE convention, the variance divides by $10$, not $9$. For regression residuals $d_1,\dots,d_5$, likelihood contains $\exp[-\frac12\sum d_i^2]$, so maximizing it is equivalent to minimizing least squares.

The Gaussian variance formula is a common place where context matters. The MLE divides by $n$ because it maximizes the Gaussian likelihood. The unbiased sample variance divides by $n-1$ because it targets population variance without systematic underestimation. Both formulas are useful; they answer different questions.

For regression, the probabilistic story is: the line gives the expected $y$, and Gaussian noise scatters points vertically around it. Lines with smaller squared residuals assign larger density to the observed points, so maximum likelihood selects the least-squares line.

### NumPy snippet
```python
import numpy as np
x = np.array([66.75,70.24,67.19,67.09,63.65,64.64,69.81,69.79,73.52,71.74])
mu = x.mean()
sigma2_mle = np.mean((x-mu)**2)
print(mu, sigma2_mle)
```

> ⚠️ Common mistakes: using $n-1$ inside Gaussian MLE variance without noticing the objective changed; forgetting residuals are vertical distances in standard regression; or comparing raw likelihoods that underflow instead of log-likelihoods.

### Summary
- Gaussian MLE mean is the sample mean.
- Gaussian MLE variance divides by $n$.
- Least squares is MLE under Gaussian residual noise.

## Bayes, MAP, and regularization

### Intuition
Bayesian statistics treats unknown parameters as uncertain quantities with prior beliefs. The slides contrast frequentists (probabilities as long-run frequencies, MLE as a point) with Bayesians (probabilities as degrees of belief, updated by data). MAP estimation maximizes posterior belief. Regularization appears when priors penalize complex model parameters.

### Definitions and formulas
Bayes' rule for parameters is
$$p(\theta\mid x)=\frac{p(x\mid\theta)p(\theta)}{p(x)}.$$
The posterior is proportional to likelihood times prior:
$$p(\theta\mid x)\propto p(x\mid\theta)p(\theta).$$
MAP chooses
$$\hat\theta_{\text{MAP}}=\arg\max_\theta p(x\mid\theta)p(\theta).$$
For a Bernoulli parameter with a Beta$(\alpha,\beta)$ prior and $k$ heads in $n$ tosses, the posterior is Beta$(\alpha+k,\beta+n-k)$ and the interior MAP is
$$\frac{\alpha+k-1}{\alpha+\beta+n-2}.$$
If coefficients have Gaussian priors centered at zero, the negative log posterior adds an $L_2$ penalty, which is regularization.

### Worked examples
With a Uniform$(0,1)$ prior for coin bias, MAP equals MLE for 8 heads and 2 tails, giving $0.8$. With a Beta$(2,2)$ prior, the posterior is Beta$(10,4)$ and MAP is $(10-1)/(10+4-2)=0.75$, slightly pulled toward fairness. In polynomial regression, a high-degree model may fit the data but gets a larger prior penalty if its coefficients are large.

Bayesian updating separates evidence from prior belief. The likelihood says what the data prefer; the prior says what was plausible before the data. With little data, the prior can matter a lot. With abundant data, a reasonable prior is often overwhelmed by the likelihood.

Regularization is MAP in optimization clothing. A Gaussian prior centered at zero says very large coefficients are unlikely before seeing data. Taking negative logs converts that prior into an $L_2$ penalty, so the fitted model must buy any extra complexity with enough improvement in data fit.

### NumPy snippet
```python
alpha, beta, k, n = 2, 2, 8, 10
map_est = (alpha+k-1)/(alpha+beta+n-2)
print(map_est)
```

> ⚠️ Common mistakes: ignoring the normalizing constant when you need probabilities rather than an argmax; assuming MAP always equals MLE; or viewing regularization as an arbitrary trick rather than a prior preference for simpler parameters.

### Summary
- Posterior $\propto$ likelihood $\times$ prior.
- MAP is a point estimate from the posterior.
- Non-informative priors can make MAP match MLE; informative priors shrink estimates.
- $L_2$ regularization corresponds to Gaussian priors on coefficients.
