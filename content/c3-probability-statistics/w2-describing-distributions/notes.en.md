# Week 2: Describing distributions and multivariable distributions

Week 2 is about reading a distribution rather than only computing one probability. The first lesson studies center, spread, tail shape, quantiles, and plots. The second lesson moves to joint distributions, marginal and conditional distributions, covariance, correlation, covariance matrices, and multivariate Gaussians.

## Measures of central tendency

### Intuition

Mean, median, and mode answer different questions. The mean is the balance point, the median is the middle ordered value, and the mode is the most frequent value. The Michael Jordan salary example shows why the median can describe a typical person better when one observation is huge.

### Definitions and formulas

For data $x_1,\ldots,x_n$,
$$
\bar x=\frac{1}{n}\sum_{i=1}^n x_i.
$$
The median is the 50% quantile. The mode is the value with largest frequency. For a PMF, the mean is the probability-weighted average.

### Fully worked example

For $0,0,0,1,1,2,2,2,2,3$, the sum is $13$ and $n=10$, so the mean is $1.3$. The two middle ordered entries are $1$ and $2$, giving median $1.5$. The mode is $2$ because it appears four times.

### NumPy snippet

```python
import numpy as np
x = np.array([0,0,0,1,1,2,2,2,2,3])
np.mean(x), np.median(x)
```

> ⚠️ A large outlier can dominate the mean.
> Compare the median before saying "typical."

### Summary

- Mean uses all magnitudes.
- Median is order based.
- Mode locates the highest pile of mass.

## Expected value

### Intuition

Expected value is the long-run average over many repetitions. It need not be a possible outcome. A fair coin game that pays $10$ on heads and $0$ on tails has expectation $5$, although one play never pays exactly $5$.

### Definitions and formulas

For a discrete random variable,
$$
E[X]=\sum_x x\,p(x).
$$
For a continuous variable,
$$
E[X]=\int x f_X(x)\,dx.
$$
A uniform variable on $[a,b]$ has expectation $(a+b)/2$.

### Fully worked example

The coin payoff $X$ has $P(X=10)=0.5$ and $P(X=0)=0.5$. Thus
$$
E[X]=0.5\cdot10+0.5\cdot0=5.
$$
A $6$ dollar entry fee is unfavorable by $1$ dollar on average, while a $4$ dollar fee is favorable by $1$ dollar.

### NumPy snippet

```python
import numpy as np
xs = np.array([0, 10])
ps = np.array([0.5, 0.5])
np.sum(xs * ps)
```

> ⚠️ The most likely value is not automatically the expectation.
> Use every probability in the PMF.

### Summary

- Expectation is weighted averaging.
- It can fall between possible outcomes.
- A fair price equals expected payoff.

## Expected value of a function

### Intuition

ML often studies a transformation of a random variable: squared loss, log-loss, or a payoff rule. Transform each outcome first, then average. This is why $E[X^2]$ differs from $(E[X])^2$.

### Definitions and formulas

For a discrete variable,
$$
E[g(X)] = \sum_x g(x)p(x).
$$
Usually $E[g(X)]\ne g(E[X])$. For a linear function $g(x)=ax+b$, however, $E[aX+b]=aE[X]+b$.

### Fully worked example

For a fair die,
$$
E[X^2]=\frac{1^2+2^2+3^2+4^2+5^2+6^2}{6}
=\frac{91}{6}.
$$
But $(E[X])^2=3.5^2=12.25$. Squaring is nonlinear, so the operations cannot be swapped.

### NumPy snippet

```python
import numpy as np
xs = np.arange(1, 7)
np.mean(xs**2)
```

> ⚠️ Do not square the mean when the target is mean square.
> This mistake also breaks variance calculations.

### Summary

- Apply $g$ before averaging.
- Linear functions behave nicely.
- Nonlinear functions generally do not commute with $E$.

## Sum of expectations

### Intuition

Linearity of expectation says expected totals can be decomposed into expected parts. Independence is not required. This makes indicator-variable arguments powerful, such as the random name-matching example in the slides.

### Definitions and formulas

For any variables with finite expectations,
$$
E[X_1+\cdots+X_n]=E[X_1]+\cdots+E[X_n].
$$
Independence is needed for some product and variance shortcuts, but not for this sum rule.

### Fully worked example

A game pays $1$ for a coin head and then pays the result of a fair die. The coin part has expectation $0.5$. The die part has expectation $3.5$. The total expected payoff is $4$ without enumerating all 12 combined outcomes.

### NumPy snippet

```python
E_coin = 0.5
E_die = (1 + 2 + 3 + 4 + 5 + 6) / 6
E_total = E_coin + E_die
```

> ⚠️ Do not add an independence assumption for a sum of expectations.
> It is stronger than what this formula needs.

### Summary

- Expectations add term by term.
- Indicators turn counts into sums.
- Variance needs more care.

## Variance

### Intuition

The mean does not describe risk. A game paying $\pm1$ and a game paying $\pm100$ can both have mean $0$, but their spreads are very different. Variance measures squared distance from the mean.

### Definitions and formulas

If $\mu=E[X]$,
$$
\mathrm{Var}(X)=E[(X-\mu)^2].
$$
The shortcut is
$$
\mathrm{Var}(X)=E[X^2]-E[X]^2.
$$
Also, $\mathrm{Var}(aX+b)=a^2\mathrm{Var}(X)$.

### Fully worked example

For $X=\pm1$ equally likely, mean is $0$ and variance is $1$. For $Y=\pm100$ equally likely, mean is still $0$ but variance is $10000$. Same center does not mean same risk.

### NumPy snippet

```python
import numpy as np
x = np.array([-1, 1])
np.mean((x - x.mean())**2)
```

> ⚠️ $E[X^2]$ alone is not variance unless $E[X]=0$.
> Check the center first.

### Summary

- Variance averages squared deviations.
- Shifts do not change variance.
- Scaling by $a$ multiplies variance by $a^2$.

## Standard deviation and standardizing

### Intuition

Variance uses squared units, so the standard deviation returns spread to the original units. Standardizing converts observations to z-scores, making features on different scales comparable.

### Definitions and formulas

$$
\sigma=\sqrt{\mathrm{Var}(X)},\qquad
Z=\frac{X-\mu}{\sigma}.
$$
If $\sigma>0$, the standardized variable has mean $0$ and standard deviation $1$.

### Fully worked example

If $x=74$, $\mu=70$, and $\sigma=8$, then
$$
z=\frac{74-70}{8}=0.5.
$$
The observation is half a standard deviation above the mean.

### NumPy snippet

```python
import numpy as np
x = np.array([66, 70, 74, 78])
z = (x - x.mean()) / x.std(ddof=0)
```

> ⚠️ Subtract before dividing.
> Reversing the order changes the scale.

### Summary

- Standard deviation is in original units.
- Z-scores count standard deviations from the mean.
- Standardization is common before ML models.

## Skewness and kurtosis

### Intuition

Skewness describes tail direction. Kurtosis describes tail heaviness after standardization. The lottery and insurance examples show that distributions can share mean and variance but differ in tail behavior.

### Definitions and formulas

$$
\mathrm{skew}=E\left[\left(\frac{X-\mu}{\sigma}\right)^3\right],
$$
and
$$
\mathrm{kurt}=E\left[\left(\frac{X-\mu}{\sigma}\right)^4\right].
$$
The third power keeps sign; the fourth power emphasizes far tails.

### Fully worked example

A lottery payoff loses $1$ with probability $0.99$ and wins $99$ with probability $0.01$. Its mean is $0$, but the rare large positive value creates positive skew. Reversing the payoff creates negative skew.

### NumPy snippet

```python
import numpy as np
z = (x - x.mean()) / x.std(ddof=0)
skew = np.mean(z**3)
kurt = np.mean(z**4)
```

> ⚠️ Raw moments are scale dependent.
> Standardize before comparing shapes.

### Summary

- Positive skew means a stronger right tail.
- Negative skew means a stronger left tail.
- High kurtosis warns about extreme tails.

## Quantiles, box plots, KDE, violin plots, QQ plots

### Intuition

Quantiles describe ordered cut points. Box plots compactly display quartiles and possible outliers. KDE and violin plots show smooth distribution shape. QQ plots compare sample quantiles with Gaussian quantiles.

### Definitions and formulas

$q_p$ leaves proportion $p$ of mass to the left. Common examples are $Q_1=q_{0.25}$, $Q_2=q_{0.5}$, and $Q_3=q_{0.75}$. The interquartile range is $IQR=Q_3-Q_1$.

### Fully worked example

For the sorted ad data $8.7,14.2,18.3,18.4,23.2,25.9,29.7,35.2,51.2,54.7,65.9,75$, the median is $(25.9+29.7)/2=27.8$. Also $Q_1=18.35$, $Q_3=52.95$, so $IQR=34.6$.

### NumPy snippet

```python
import numpy as np
np.quantile(x, [0.25, 0.5, 0.75])
```

> ⚠️ Quartiles can vary by interpolation rule.
> State the method when precision matters.

### Summary

- Quantiles depend on order.
- IQR describes the middle half.
- QQ plots diagnose normality visually.

## Joint distributions: discrete

### Intuition

A discrete joint distribution assigns probabilities to pairs. In the age-height table, each cell is the probability of a child having a particular age and height. The whole table, not each row, sums to $1$.

### Definitions and formulas

$$
p_{XY}(x,y)=P(X=x,Y=y).
$$
A valid joint PMF satisfies
$$
p_{XY}(x,y)\ge0,\qquad
\sum_x\sum_y p_{XY}(x,y)=1.
$$
For independent variables, $p_{XY}(x,y)=p_X(x)p_Y(y)$.

### Fully worked example

If 3 of 10 children are age 9 and height 49 inches, then $p_{XY}(9,49)=3/10$. For two independent fair dice, every ordered pair has probability $1/36$.

### NumPy snippet

```python
import numpy as np
P = np.array([[0.1, 0.2], [0.3, 0.4]])
P.sum()
```

> ⚠️ A row need not sum to $1$ in a joint table.
> The total mass of all cells must be $1$.

### Summary

- Joint PMFs describe pairs.
- All cells together sum to one.
- Independence factors a joint PMF.

## Joint distributions: continuous

### Intuition

For continuous variables, density height is not probability. Probability comes from area or volume under the density over a region, such as a rectangle in the waiting-time and satisfaction plane.

### Definitions and formulas

A joint PDF satisfies $f_{XY}(x,y)\ge0$ and
$$
\iint f_{XY}(x,y)\,dx\,dy=1.
$$
For a region $A$,
$$
P((X,Y)\in A)=\iint_A f_{XY}(x,y)\,dx\,dy.
$$

### Fully worked example

For a uniform density on $[0,2]\times[0,3]$, the rectangle area is $6$, so the density height is $1/6$. A subregion with area $1.5$ has probability $1.5/6=0.25$.

### NumPy snippet

```python
import numpy as np
prob = np.sum(density[mask] * dx * dy)
```

> ⚠️ $f_{XY}(4,7)$ is not a point probability.
> Integrate over a nonzero region.

### Summary

- Continuous probability uses integrals.
- Point probabilities are usually zero.
- Density can exceed one on small regions.

## Marginal and conditional distributions

### Intuition

Marginal distributions ignore one variable by summing or integrating it out. Conditional distributions renormalize a slice after information is given. In the age-height table, knowing age equals 9 means looking only at that row.

### Definitions and formulas

For a discrete joint PMF,
$$
p_X(x)=\sum_y p_{XY}(x,y).
$$
When $p_X(x)>0$,
$$
p_{Y|X=x}(y)=\frac{p_{XY}(x,y)}{p_X(x)}.
$$

### Fully worked example

The slide has $P(X=9)=4/10$ and $P(X=9,Y=49)=3/10$. Therefore
$$
P(Y=49\mid X=9)=\frac{3/10}{4/10}=3/4.
$$
The row is divided by its own row total.

### NumPy snippet

```python
import numpy as np
px = P.sum(axis=1)
cond_y_given_x0 = P[0] / px[0]
```

> ⚠️ The conditional denominator is the marginal of the condition.
> Do not divide by one again if the table is already a PMF.

### Summary

- Marginals sum out variables.
- Conditionals normalize slices.
- The conditioning event must have positive probability.

## Covariance, covariance matrix, and correlation

### Intuition

Covariance measures whether two centered variables tend to move together. Positive covariance means paired deviations often share sign. Negative covariance means opposite signs dominate. Correlation rescales covariance to compare strength.

### Definitions and formulas

$$
\mathrm{Cov}(X,Y)=E[(X-\mu_X)(Y-\mu_Y)]
=E[XY]-E[X]E[Y].
$$
The correlation coefficient is
$$
\rho=\frac{\mathrm{Cov}(X,Y)}{\sigma_X\sigma_Y}.
$$

### Fully worked example

The slides give positive age-height covariance $17$, negative age-naps covariance $-7.45$, and near-zero age-grades covariance. If $\mathrm{Var}(X)=9.17$ and $\mathrm{Var}(Y)=7.57$, then
$$
\rho=\frac{-7.45}{\sqrt{9.17}\sqrt{7.57}}\approx -0.894.
$$

### NumPy snippet

```python
import numpy as np
X = np.c_[x, y]
np.cov(X, rowvar=False, bias=True)
np.corrcoef(x, y)[0, 1]
```

> ⚠️ Zero covariance does not prove independence.
> It only removes linear co-movement.

### Summary

- Covariance gives direction of co-movement.
- Correlation is unitless.
- Covariance matrices store all pairwise covariances.

## Multivariate Gaussian distribution

### Intuition

A univariate Gaussian uses a mean and a variance. A multivariate Gaussian uses a mean vector and covariance matrix. The covariance matrix controls widths and the tilt of elliptical contours.

### Definitions and formulas

For $x\in\mathbb{R}^n$,
$$
f(x)=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}
\exp\left(-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)\right).
$$
Diagonal $\Sigma$ corresponds to independent coordinates in the Gaussian case.

### Fully worked example

If height and weight were independent with variances $4$ and $9$, then
$$
\Sigma=\begin{bmatrix}4&0\\0&9\end{bmatrix}.
$$
With covariance $3$, the off-diagonal entries become $3$, tilting the density ellipses.

### NumPy snippet

```python
import numpy as np
d = x - mu
score = d @ np.linalg.inv(Sigma) @ d
```

> ⚠️ The off-diagonal entries are not decoration.
> They encode dependence and contour tilt.

### Summary

- $\mu$ locates the center.
- $\Sigma$ determines spread and dependence.
- The exponent uses a Mahalanobis distance.

## Week 2 conclusion

### Intuition

The week builds a distribution-reading toolkit. One-variable summaries describe center, spread, and shape. Two-variable tools describe joint probability and dependence. Gaussian models connect these ideas to ML assumptions.

### Definitions and formulas

Mean and median address center. Variance and standard deviation address spread. Skewness and kurtosis address tails. Joint, marginal, and conditional distributions address multiple variables. Covariance and correlation address linear relationships.

### Fully worked example

For customer-service data, report mean waiting time, median if outliers exist, IQR for middle spread, KDE for shape, and correlation between waiting time and satisfaction to see whether longer waits align with lower ratings.

### NumPy snippet

```python
import numpy as np
center = np.mean(x)
spread = np.std(x, ddof=0)
relation = np.corrcoef(x, y)[0, 1]
```

> ⚠️ Do not choose a statistic just because it is familiar.
> Choose it for the question being asked.

### Summary

- One variable needs center, spread, and shape.
- Two variables need joint and conditional views.
- Dependence needs covariance or correlation.
