## Derivative foundations

This group introduces why derivatives matter in ML.
It connects loss, average slope, point slope,
tangent lines, and notation into one foundation.

### Machine learning motivation

#### Intuition
Training is guided trial and correction.
The model predicts a house price or a label.
The loss measures how wrong that guess is.
A derivative says how the loss moves.
If $J'(w)>0$, raising $w$ raises the loss.
Gradient descent therefore tries a smaller $w$.

#### Definitions and formulas
For one parameter, the training signal is
$\frac{d}{dw}J(w)$.
For many parameters it becomes a gradient vector.
The sign gives direction.
The size gives sensitivity.

#### Fully worked example
Suppose $J(w)=(w-3)^2$.
Then $J'(w)=2(w-3)$.
At $w=1$, the slope is $-4$.
With learning rate $0.1$,
$w_{\text{new}}=1-0.1(-4)=1.4$.
The update moves toward $3$.

#### NumPy / Python snippet
```python
w = 1.0
lr = 0.1
grad = 2*(w - 3)
w = w - lr*grad
```

> ⚠️ Common mistake: using the prediction error alone
> as the update without differentiating the loss.

#### Summary
- Derivatives connect loss to parameter updates.
- A positive slope suggests moving left.
- A negative slope suggests moving right.

### Introduction to derivatives

#### Intuition
Average speed over five seconds is useful.
Instant speed at one time needs a limit.
The slides shrink the time interval.
The secant slope becomes a point slope.

#### Definitions and formulas
The average slope from $t=a$ to $t=b$ is
$\frac{x(b)-x(a)}{b-a}$.
The derivative is the limiting value.
Use $x'(t)$ for velocity.

#### Fully worked example
From the table, $x(10)=122$ and $x(15)=202$.
Average speed on $[10,15]$ is
$\frac{202-122}{15-10}=16$ m/s.
Using $x(12)=155$ and $x(13)=170$
gives a closer estimate near $12.5$:
$15$ m/s.

#### NumPy / Python snippet
```python
t = [12, 13]
x = [155, 170]
speed = (x[1]-x[0])/(t[1]-t[0])
```

> ⚠️ Common mistake: saying the car has one speed
> for the whole trip when table intervals differ.

#### Summary
- A derivative refines average rate.
- Smaller intervals give local information.
- Units matter: meters per second here.

### Derivatives and tangents

#### Intuition
A tangent line just touches the curve locally.
Its slope is the derivative at that point.
Near the point, the curve and tangent agree
to first order.

#### Definitions and formulas
At $x=a$, the tangent line is
$y=f(a)+f'(a)(x-a)$.
This is also the local linear approximation.

#### Fully worked example
Let $f(x)=x^2$ and choose $a=1$.
Then $f(1)=1$ and $f'(1)=2$.
The tangent line is $y=1+2(x-1)$.
At $x=1.1$, it predicts $1.2$.
The true value is $1.21$.

#### NumPy / Python snippet
```python
def tangent_at_one(x):
    return 1 + 2*(x-1)
```

> ⚠️ Common mistake: drawing a line through two
> faraway points and calling it a tangent.

#### Summary
- Tangent slope equals derivative.
- Tangents are local, not global.
- Linearization is useful near the base point.

## Extrema and zero slope

This group focuses on what zero slope can mean.
It explains why horizontal tangents are useful
but still need classification and boundary checks.

### Slopes, maxima, and minima

#### Intuition
At a smooth hilltop or valley bottom,
the curve momentarily stops rising or falling.
That is why the tangent becomes horizontal.
Zero slope marks a candidate, not a verdict.

#### Definitions and formulas
For an interior smooth optimum,
$f'(a)=0$.
Check nearby values, signs, or second derivative.
Endpoints can be optimal with nonzero slope.

#### Fully worked example
For $f(x)=(x-4)^2+7$,
$f'(x)=2(x-4)$.
Solving $f'(x)=0$ gives $x=4$.
Values around it are larger:
$f(3)=8$, $f(4)=7$, $f(5)=8$.
So $x=4$ is a minimum.

#### NumPy / Python snippet
```python
xs = [3, 4, 5]
vals = [(x-4)**2 + 7 for x in xs]
```

> ⚠️ Common mistake: every zero slope is not a
> minimum; $x^3$ has zero slope at $0$.

#### Summary
- Zero slope gives a critical point.
- Classify it with context.
- Boundaries need separate checking.

### Derivative notation

#### Intuition
Different notations emphasize different ideas.
$f'(x)$ names a new function.
$\frac{dy}{dx}$ highlights change in output
per change in input.

#### Definitions and formulas
If $y=f(x)$, then
$f'(x)=\frac{d}{dx}f(x)=\frac{dy}{dx}$.
At a point, write $f'(2)$ or
$\left.\frac{dy}{dx}\right|_{x=2}$.

#### Fully worked example
Let $y=3x^2-1$.
Lagrange notation gives $f'(x)=6x$.
Leibniz notation gives $\frac{dy}{dx}=6x$.
At $x=2$, both give $12$.

#### NumPy / Python snippet
```python
def dy_dx(x):
    return 6*x
```

> ⚠️ Common mistake: treating $dy$ and $dx$
> as ordinary finite changes without context.

#### Summary
- Prime notation is compact.
- Leibniz notation shows variables.
- Both describe the same derivative.

## Derivatives of common functions

This group builds the first derivative table.
It covers constants, lines, quadratics,
the power rule, and reciprocal functions.

### Some common derivatives: Lines

#### Intuition
A constant function is flat.
A line rises by the same amount everywhere.
So its derivative does not depend on $x$.

#### Definitions and formulas
$\frac{d}{dx}c=0$.
For $f(x)=ax+b$, $f'(x)=a$.
The intercept $b$ shifts the line vertically
but does not change its slope.

#### Fully worked example
For $f(x)=-3x+7$,
the slope is always $-3$.
At $x=0$, $x=5$, or $x=-2$,
the derivative remains $-3$.

#### NumPy / Python snippet
```python
import numpy as np
x = np.array([-2, 0, 5])
derivative = np.full_like(x, -3)
```

> ⚠️ Common mistake: differentiating $b$ as $b$
> instead of $0$.

#### Summary
- Constants have zero derivative.
- The derivative of $ax+b$ is $a$.
- Lines have constant rate of change.

### Some common derivatives: Quadratics

#### Intuition
For $x^2$, the graph gets steeper
as $|x|$ grows.
The slope is negative on the left,
zero at the vertex,
and positive on the right.

#### Definitions and formulas
$\frac{d}{dx}x^2=2x$.
More generally,
$\frac{d}{dx}(ax^2+bx+c)=2ax+b$.

#### Fully worked example
Let $f(x)=2x^2-4x+1$.
Then $f'(x)=4x-4$.
At $x=3$, the slope is $8$.
At $x=1$, the tangent is horizontal.

#### NumPy / Python snippet
```python
def quad_grad(x):
    return 4*x - 4
```

> ⚠️ Common mistake: forgetting the coefficient
> in front of $x^2$.

#### Summary
- Quadratic slopes are linear.
- The vertex occurs where the slope is zero.
- Coefficients scale the derivative.

### Some common derivatives: Higher degree polynomials

#### Intuition
Polynomials are built from powers.
The power rule handles each term separately.
That makes complicated curves manageable.

#### Definitions and formulas
For integer $n$,
$\frac{d}{dx}x^n=nx^{n-1}$.
Apply the rule term by term.
Constants vanish.

#### Fully worked example
For $p(x)=4x^5-2x^3+x-9$,
$p'(x)=20x^4-6x^2+1$.
At $x=1$, the slope is
$20-6+1=15$.

#### NumPy / Python snippet
```python
def pprime(x):
    return 20*x**4 - 6*x**2 + 1
```

> ⚠️ Common mistake: lowering the power
> but forgetting to multiply by the old power.

#### Summary
- Differentiate each monomial.
- Add the resulting derivatives.
- Polynomial derivatives stay polynomial.

### Some common derivatives: Other power functions

#### Intuition
Negative powers describe reciprocal curves.
The same power rule still works,
but the domain may exclude zero.

#### Definitions and formulas
$x^{-1}=1/x$.
Using the power rule,
$\frac{d}{dx}x^{-1}=-x^{-2}$.
So $\frac{d}{dx}(1/x)=-1/x^2$.

#### Fully worked example
At $x=2$,
the derivative of $1/x$ is
$-1/2^2=-1/4$.
The graph is decreasing there,
so the negative sign makes sense.

#### NumPy / Python snippet
```python
def reciprocal_grad(x):
    return -1/(x*x)
```

> ⚠️ Common mistake: writing $1/x^2$
> and losing the negative sign.

#### Summary
- Negative exponents are allowed.
- Watch the domain near zero.
- Reciprocal slopes are negative for $x>0$.

## Inverse, trigonometric, exponential, and log functions

This group extends derivatives beyond polynomials.
It links inverse-function slopes, trig waves,
the natural base $e$, exponentials, and logarithms.

### The inverse function and its derivative

#### Intuition
An inverse swaps input and output.
Geometrically, its graph reflects across $y=x$.
Reflection turns steep slopes into shallow ones.

#### Definitions and formulas
If $y=f(x)$ and $g=f^{-1}$,
then $g'(y)=1/f'(x)$.
This requires $f'(x)\ne0$ locally.

#### Fully worked example
Take $f(x)=x^2$ on $x>0$.
Its inverse is $g(y)=\sqrt y$.
At $x=2$, $y=4$ and $f'(2)=4$.
Thus $g'(4)=1/4$.
Indeed, $(\sqrt y)'=1/(2\sqrt y)$.

#### NumPy / Python snippet
```python
y = 4.0
gprime = 1/(2*y**0.5)
```

> ⚠️ Common mistake: using the inverse rule
> where the original function is not one-to-one.

#### Summary
- Inverse slopes are reciprocal slopes.
- Use matching points $(x,y)$.
- Local invertibility matters.

### Derivative of trigonometric functions

#### Intuition
The sine wave is steepest at zero crossings.
It is flat at peaks and troughs.
That pattern is exactly cosine.

#### Definitions and formulas
$\frac{d}{dx}\sin x=\cos x$.
$\frac{d}{dx}\cos x=-\sin x$.
Angles are measured in radians.

#### Fully worked example
For $f(x)=3\sin x-2\cos x$,
$f'(x)=3\cos x+2\sin x$.
At $x=0$, $f'(0)=3$.

#### NumPy / Python snippet
```python
import numpy as np
def trig_grad(x):
    return 3*np.cos(x) + 2*np.sin(x)
```

> ⚠️ Common mistake: using degrees inside
> calculus formulas without conversion.

#### Summary
- Sine differentiates to cosine.
- Cosine differentiates to negative sine.
- Radians are the calculus unit.

### Meaning of the exponential (e)

#### Intuition
The number $e$ comes from compounding growth
more and more often.
The limit is not arbitrary;
it is the natural base for continuous change.

#### Definitions and formulas
$e=\lim_{n\to\infty}(1+1/n)^n$.
The slides compare yearly, semiannual,
and more frequent compounding.

#### Fully worked example
With $n=10$,
$(1+1/10)^{10}\approx2.594$.
With $n=1000$,
$(1+1/1000)^{1000}\approx2.717$.
These approach $2.71828\ldots$.

#### NumPy / Python snippet
```python
for n in [10, 1000]:
    print((1 + 1/n)**n)
```

> ⚠️ Common mistake: thinking $e$ is chosen
> only for convenience.

#### Summary
- $e$ is a growth limit.
- It appears in continuous compounding.
- It makes exponential calculus simple.

### The derivative of ex

#### Intuition
$e^x$ is its own rate of change.
The height of the curve equals its slope.
This is why it models natural growth.

#### Definitions and formulas
$\frac{d}{dx}e^x=e^x$.
For $a e^x$, the derivative is $a e^x$.

#### Fully worked example
Let $f(x)=5e^x$.
Then $f'(x)=5e^x$.
At $x=0$, the slope is $5$.
At $x=2$, the slope is $5e^2$.

#### NumPy / Python snippet
```python
import numpy as np
def exp_grad(x):
    return 5*np.exp(x)
```

> ⚠️ Common mistake: treating $e^x$
> like a power $x^e$.

#### Summary
- The natural exponential is self-derivative.
- Scaling the function scales the slope.
- Growth rate rises with the function value.

### The derivative of log(x)

#### Intuition
Logarithm is the inverse of exponential.
As $x$ grows, equal absolute changes
become less important.
So the slope $1/x$ gets smaller.

#### Definitions and formulas
For natural log,
$\frac{d}{dx}\log x=1/x$.
The domain is $x>0$.
This follows from the inverse derivative rule.

#### Fully worked example
For $f(x)=\log x$,
$f'(4)=1/4$.
For $g(x)=3\log x$,
$g'(4)=3/4$.

#### NumPy / Python snippet
```python
def log_grad(x):
    return 1/x
```

> ⚠️ Common mistake: applying $\log x$
> or its derivative at nonpositive $x$.

#### Summary
- Natural log differentiates to $1/x$.
- The slope decreases as $x$ increases.
- The domain restriction is essential.

## Differentiability

This group is about when derivatives do not exist.
It highlights corners, one-sided slopes,
and the difference between continuity and smoothness.

### Existence of the derivative

#### Intuition
A derivative exists only when left and right
local slopes agree and stay finite.
Corners, jumps, and vertical tangents break this.

#### Definitions and formulas
A function is differentiable at $a$
if the limit defining $f'(a)$ exists.
For $|x|$ at zero,
the left slope is $-1$ and right slope is $1$.

#### Fully worked example
For $f(x)=|x|$,
$\frac{|0+h|-|0|}{h}=1$ when $h>0$.
The same quotient is $-1$ when $h<0$.
The two-sided limit does not exist.

#### NumPy / Python snippet
```python
def right_abs_slope(h):
    return abs(h)/h
```

> ⚠️ Common mistake: averaging left and right
> slopes and calling that the derivative.

#### Summary
- Differentiability needs a two-sided limit.
- Corners can be continuous but not smooth.
- Always inspect suspicious points.

## Derivative rules

This group collects the rules used to combine
derivatives.  Scalar multiples, sums, products,
and compositions each have a different pattern.

### Properties of the derivative: Multiplication by scalars

#### Intuition
Stretching a graph vertically stretches
each rise by the same factor.
The run is unchanged,
so every slope is scaled too.

#### Definitions and formulas
For constant $c$,
$(cf)'=c f'$.
The constant must not depend on $x$.

#### Fully worked example
If $f(x)=x^2$ and $g(x)=7f(x)$,
then $g(x)=7x^2$.
So $g'(x)=14x$,
which is $7$ times $2x$.

#### NumPy / Python snippet
```python
def scaled_grad(x):
    return 7*(2*x)
```

> ⚠️ Common mistake: using this rule
> when the multiplier is another function.

#### Summary
- Constant factors pass through derivatives.
- Vertical scaling scales slope.
- Variable factors need the product rule.

### Properties of the derivative: The sum rule

#### Intuition
If two effects add,
their small changes add too.
This is why derivatives distribute over sums.

#### Definitions and formulas
$(g+h)'=g'+h'$.
The same idea extends to several terms.

#### Fully worked example
Let $f(x)=x^2+\sin x$.
Then $f'(x)=2x+\cos x$.
At $x=0$, the slope is $1$.
The quadratic part contributes $0$;
the sine part contributes $1$.

#### NumPy / Python snippet
```python
import numpy as np
def sum_grad(x):
    return 2*x + np.cos(x)
```

> ⚠️ Common mistake: differentiating only
> the first term of a sum.

#### Summary
- Differentiate each added term.
- Then add the derivative pieces.
- This powers polynomial differentiation.

### Properties of the derivative: The product rule

#### Intuition
When two changing quantities multiply,
both can cause the product to change.
One term freezes the second factor;
the other freezes the first.

#### Definitions and formulas
$(gh)'=g'h+gh'$.
Do not use $g'h'$.

#### Fully worked example
For $f(x)=x^2\sin x$,
let $g=x^2$ and $h=\sin x$.
Then $g'=2x$ and $h'=\cos x$.
So $f'(x)=2x\sin x+x^2\cos x$.

#### NumPy / Python snippet
```python
import numpy as np
def product_grad(x):
    return 2*x*np.sin(x) + x*x*np.cos(x)
```

> ⚠️ Common mistake: multiplying
> the two derivatives together.

#### Summary
- Products need two terms.
- Each term changes one factor.
- Missing a term changes the answer.

### Properties of the derivative: The chain rule

#### Intuition
A composite changes in layers.
The outside changes with respect
to the inside value.
The inside changes with respect to $x$.
Multiply those rates.

#### Definitions and formulas
$\frac{d}{dx}g(h(x))=g'(h(x))h'(x)$.
This is central in neural networks.

#### Fully worked example
For $f(x)=(3x+1)^2$,
outside derivative is $2u$.
Inside derivative is $3$.
Thus $f'(x)=2(3x+1)\cdot3$.
At $x=1$, the slope is $24$.

#### NumPy / Python snippet
```python
def chain_grad(x):
    return 6*(3*x + 1)
```

> ⚠️ Common mistake: differentiating
> the outside and forgetting $h'(x)$.

#### Summary
- Identify outside and inside functions.
- Differentiate both layers.
- Multiply the rates.

## Optimization and ML losses

This group shows why the earlier rules matter.
Derivatives guide search, squared loss leads to
means, and log-loss connects probability to training.

### Introduction to optimization

#### Intuition
Optimization searches for the best input
under a chosen objective.
In the slides, the coolest point
or cheapest connection point is the target.

#### Definitions and formulas
For gradient descent in one dimension,
$x_{\text{new}}=x-\alpha f'(x)$.
The learning rate $\alpha$ controls step size.

#### Fully worked example
Let $f(x)=(x-6)^2$.
At $x=2$, $f'(x)=2(x-6)=-8$.
With $\alpha=0.25$,
$x_{\text{new}}=2-0.25(-8)=4$.
The move heads toward $6$.

#### NumPy / Python snippet
```python
x = 2.0
x = x - 0.25*2*(x - 6)
```

> ⚠️ Common mistake: stepping in the direction
> of the gradient when minimizing.

#### Summary
- Optimization needs an objective.
- Derivatives provide search direction.
- Step size changes the behavior.

### Optimization of squared loss: The one powerline problem

#### Intuition
With one powerline, the best house position
is directly at that line.
Squared distance is zero there
and positive everywhere else.

#### Definitions and formulas
If the line is at $a$,
$J(x)=(x-a)^2$.
The derivative is $J'(x)=2(x-a)$.

#### Fully worked example
For $a=8$,
$J(x)=(x-8)^2$.
Setting $J'(x)=2(x-8)=0$
gives $x=8$.
The minimum cost is $0$.

#### NumPy / Python snippet
```python
a = 8
best_x = a
```

> ⚠️ Common mistake: minimizing distance sign
> instead of squared distance.

#### Summary
- One squared distance is minimized at its center.
- The derivative points back to the line.
- Cost cannot go below zero.

### Optimization of squared loss: The two powerline problem

#### Intuition
With two powerlines, the best point balances
the squared distances to both.
The optimum is their midpoint.

#### Definitions and formulas
$J(x)=(x-a)^2+(x-b)^2$.
Then $J'(x)=2(x-a)+2(x-b)$.
Solving gives $x=(a+b)/2$.

#### Fully worked example
Let $a=2$ and $b=10$.
$J'(x)=2(x-2)+2(x-10)=4x-24$.
Set $4x-24=0$.
The best point is $x=6$.

#### NumPy / Python snippet
```python
a, b = 2, 10
x_star = (a + b)/2
```

> ⚠️ Common mistake: choosing the point
> with smaller individual distance only.

#### Summary
- Two squared losses balance at the midpoint.
- The derivative equation is linear.
- Both lines influence the solution.

### Optimization of squared loss: The three powerline problem

#### Intuition
For several powerlines,
the squared-loss optimum is the average
of their positions.
This foreshadows least squares.

#### Definitions and formulas
$J(x)=\sum_i(x-a_i)^2$.
Then $J'(x)=2\sum_i(x-a_i)$.
Setting it to zero gives
$x=\frac{1}{n}\sum_i a_i$.

#### Fully worked example
For positions $1,4,10$,
the mean is $(1+4+10)/3=5$.
The derivative is
$2(x-1)+2(x-4)+2(x-10)=6x-30$.
Setting zero gives $x=5$.

#### NumPy / Python snippet
```python
import numpy as np
x_star = np.mean([1, 4, 10])
```

> ⚠️ Common mistake: using the median
> for squared loss; that fits absolute loss.

#### Summary
- Squared loss leads to the mean.
- More points add more derivative terms.
- The optimum balances total pull.

### Optimization of log-loss Part 1

#### Intuition
For coin data, the likelihood multiplies
probabilities assigned to observed outcomes.
Many wrong confident probabilities
make the product tiny.

#### Definitions and formulas
For $H$ heads and $T$ tails,
$L(p)=p^H(1-p)^T$.
Maximizing likelihood chooses the probability
that best explains the sequence.

#### Fully worked example
With $7$ heads and $3$ tails,
coin $p=0.7$ gives $0.7^7 0.3^3$.
Coin $p=0.5$ gives $0.5^{10}$.
The first is larger,
so it better matches the data.

#### NumPy / Python snippet
```python
p = 0.7
likelihood = p**7 * (1-p)**3
```

> ⚠️ Common mistake: comparing only the
> most recent toss instead of all outcomes.

#### Summary
- Likelihood multiplies observation chances.
- Better probabilities give larger products.
- Products can become very small.

### Optimization of log-loss Part 2

#### Intuition
Logs turn products into sums.
That makes differentiation easier.
Negative log-likelihood becomes a loss
to minimize.

#### Definitions and formulas
Binary log-loss is
$-[y\log p+(1-y)\log(1-p)]$.
For many data points, average the terms.

#### Fully worked example
If $y=1$ and $p=0.9$,
loss is $-\log(0.9)\approx0.105$.
If $y=1$ and $p=0.1$,
loss is $-\log(0.1)\approx2.303$.
Confident wrong predictions are expensive.

#### NumPy / Python snippet
```python
import numpy as np
loss = -np.log(0.9)
```

> ⚠️ Common mistake: minimizing likelihood
> instead of minimizing negative log-likelihood.

#### Summary
- Log changes products into sums.
- Negative log-likelihood is log-loss.
- Bad confident probabilities get large loss.

### Conclusion

#### Intuition
Week 1 builds one chain:
local slope becomes derivative rules,
and derivative rules power optimization.
That chain is a foundation for ML training.

#### Definitions and formulas
Remember three layers:
definition by a limit,
rules for fast differentiation,
and updates such as
$x_{\text{new}}=x-\alpha f'(x)$.

#### Fully worked example
For $J(w)=(w-5)^2$,
the derivative is $2(w-5)$.
Starting at $w=9$ with $\alpha=0.25$,
the next value is
$9-0.25\cdot8=7$.
The parameter moves toward the minimizer.

#### NumPy / Python snippet
```python
w = 9
w -= 0.25*2*(w - 5)
```

> ⚠️ Common mistake: memorizing derivative
> tables without connecting them to loss.

#### Summary
- Derivatives measure local change.
- Rules make derivatives practical.
- Optimization uses slopes to improve models.
