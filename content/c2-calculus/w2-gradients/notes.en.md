# Week 2: Gradients and Gradient Descent

The Week 2 slides start with a surface such as
$f(x,y)=x^2+y^2$, then build toward optimization.
The important change from Week 1 is that a derivative is no
longer just a number on a line.  In several variables, we need
one slope per coordinate, collect those slopes into a gradient,
and use that vector to move parameters of a model.

## Tangent planes

### Intuition
A two-variable function is a height field.  If you freeze
$y=4$, the surface becomes the curve $f(x,4)$.  If you freeze
$x=2$, it becomes $f(2,y)$.  At $(2,4)$ the tangent plane is
the flat sheet that matches both slice tangents at once.

### Definitions and formulas
For a differentiable surface, the local linear model at
$(a,b)$ is
$$L(x,y)=f(a,b)+f_x(a,b)(x-a)+f_y(a,b)(y-b).$$
This is the equation of the tangent plane written as height
$z=L(x,y)$.

### Fully worked example
Take the slide function $f(x,y)=x^2+y^2$ at $(2,4)$.
First compute the height: $f(2,4)=4+16=20$.
The $x$-slice slope is $f_x=2x$, so $f_x(2,4)=4$.
The $y$-slice slope is $f_y=2y$, so $f_y(2,4)=8$.
Therefore
$$z=20+4(x-2)+8(y-4).$$
At $(2.1,3.9)$ this plane predicts
$20+0.4-0.8=19.6$, a nearby approximation.

> ⚠️ A tangent plane is local.  It is not claiming that the
> paraboloid is flat far away from $(2,4)$.

### NumPy snippet
```python
def f(x, y):
    return x*x + y*y
a, b = 2.0, 4.0
fx, fy = 2*a, 2*b
def plane(x, y):
    return f(a, b) + fx*(x-a) + fy*(y-b)
print(plane(2.1, 3.9))
```

### Summary
- Slice the surface in two coordinate directions.
- Use the two slice slopes in one local plane.
- The plane is a first-order approximation.

## Partial derivatives

### Intuition
A partial derivative asks: "what is the slope if only this
coordinate is allowed to move?"  The slides show this by
treating one variable as a constant while differentiating the
other variable normally.

### Definitions and formulas
The notation $f_x$ or $\partial f/\partial x$ means that
$y$ is held fixed.  Similarly, $f_y$ holds $x$ fixed.
For a product such as $3x^2y^3$, the factor not involving the
active variable behaves like a constant coefficient.

### Fully worked example
Let $f(x,y)=3x^2y^3$.  For $f_x$, keep $y^3$ unchanged:
$$f_x=3y^3\cdot 2x=6xy^3.$$
For $f_y$, keep $3x^2$ unchanged:
$$f_y=3x^2\cdot 3y^2=9x^2y^2.$$
At $(1,2)$, these become $6\cdot1\cdot8=48$ and
$9\cdot1\cdot4=36$.

> ⚠️ "Treat as constant" does not mean "replace by 1" or
> "replace by 0"; it means keep the symbol while differentiating.

### NumPy snippet
```python
def f(x, y):
    return 3*x*x*y**3
h = 1e-5
x, y = 1.0, 2.0
dfdx = (f(x+h, y) - f(x-h, y)) / (2*h)
dfdy = (f(x, y+h) - f(x, y-h)) / (2*h)
print(round(dfdx), round(dfdy))
```

### Summary
- A partial derivative is a slope of a slice.
- The inactive variable remains fixed, not erased.
- Finite differences can check symbolic work.

## Gradients

### Intuition
The gradient packages all partial derivatives into one vector.
It tells you the direction in input space where the function
increases fastest.  The negative gradient points down the hill.

### Definitions and formulas
For two variables,
$$\nabla f(x,y)=\begin{bmatrix}f_x(x,y)\\ f_y(x,y)\end{bmatrix}.$$
The directional derivative in a unit direction $u$ is
$\nabla f\cdot u$.  This dot product is largest when $u$ points
the same way as the gradient.

### Fully worked example
For $f(x,y)=x^2+y^2$, the partials are $2x$ and $2y$.
At $(2,3)$,
$$\nabla f(2,3)=\begin{bmatrix}4\\6\end{bmatrix}.$$
Moving a small amount toward $(4,6)$ raises the function most
quickly; moving toward $(-4,-6)$ lowers it most quickly.

> ⚠️ The gradient lives in the $(x,y)$ plane.  It is not the
> vertical height $z$ on the surface.

### NumPy snippet
```python
import numpy as np
def grad_xy(point):
    x, y = point
    return np.array([2*x, 2*y])
g = grad_xy(np.array([2.0, 3.0]))
print(g, -g)
```

### Summary
- The gradient is a vector of coordinate slopes.
- It points to steepest local ascent.
- Descent uses the opposite direction.

## Gradients and maxima/minima

### Intuition
At a smooth interior peak or valley, there is no first-order
uphill direction left.  That is why the slides set both partial
derivatives equal to zero when searching for extrema.

### Definitions and formulas
A stationary point satisfies $\nabla f=0$.  This condition
gives candidates only.  To decide minimum, maximum, or saddle,
you must use context, boundaries, second derivatives, or direct
comparison of values.

### Fully worked example
For $f(x,y)=x^2+y^2$, solve $2x=0$ and $2y=0$.
The only candidate is $(0,0)$.  Because squares are never
negative, $f(0,0)=0$ is a global minimum.
For the slide heat formula, the same first step produces a list
of candidates; then points outside the room are discarded and
remaining temperatures are compared.

> ⚠️ A zero gradient does not guarantee a minimum.  The function
> $x^2-y^2$ has $\nabla f(0,0)=0$, but the origin is a saddle.

### NumPy snippet
```python
candidates = [(0, 0), (4, 4), (6, 6)]
def simple_heat(x, y):
    return 85 - (x*x*(x-6)*y*y*(y-6))/90
for p in candidates:
    print(p, round(simple_heat(*p), 3))
```

### Summary
- Set every partial derivative to zero.
- Filter impossible or boundary-irrelevant points.
- Classify candidates before naming an optimum.

## Analytical optimization

### Intuition
Analytical optimization solves the gradient equations exactly.
It is satisfying for a small quadratic, but the slides motivate
gradient descent because many real cost functions resist closed
forms.

### Definitions and formulas
For a cost $E(m,b)$, the analytical method solves
$$E_m(m,b)=0,\qquad E_b(m,b)=0.$$
When $E$ is quadratic, these equations are often linear in the
unknown parameters.

### Fully worked example
The slide cost is
$$E=14m^2+3b^2+38+12mb-42m-20b.$$
Differentiate:
$$E_m=28m+12b-42,\qquad E_b=12m+6b-20.$$
Set both to zero.  Doubling the second equation gives
$24m+12b-40=0$.  Subtract from the first:
$4m-2=0$, so $m=1/2$.  Substitute into $E_b=0$:
$6+6b-20=0$, hence $b=7/3$.

> ⚠️ Solving only $E_m=0$ leaves a whole line of candidates,
> not a single optimal pair.

### NumPy snippet
```python
import numpy as np
A = np.array([[28, 12], [12, 6]], dtype=float)
r = np.array([42, 20], dtype=float)
print(np.linalg.solve(A, r))
```

### Summary
- Write all first-order equations.
- Solve the system jointly.
- Closed forms are useful but not always available.

## Linear regression motivation

### Intuition
Linear regression turns fitting a line into minimizing a cost.
The data points in the slides are $(1,2)$, $(2,5)$, and
$(3,3)$.  The model predicts $\hat y=mx+b$, so $m$ and $b$ are
the variables being optimized.

### Definitions and formulas
The residual for point $i$ is $mx_i+b-y_i$.  The slide uses the
sum of squared residuals:
$$E(m,b)=\sum_i (mx_i+b-y_i)^2.$$
Squaring makes positive and negative errors both costly.

### Fully worked example
For the three slide points, residuals are
$m+b-2$, $2m+b-5$, and $3m+b-3$.
Expanding and collecting terms gives
$$E=14m^2+3b^2+12mb-42m-20b+38.$$
The analytical solution from the previous section is
$m=1/2$, $b=7/3$, so the fitted line is
$y=\tfrac12x+\tfrac73$.

> ⚠️ The plotted data coordinate $x$ is not the same kind of
> variable as the parameter $m$.  The cost surface lives over
> $(m,b)$.

### NumPy snippet
```python
import numpy as np
x = np.array([1., 2., 3.])
y = np.array([2., 5., 3.])
m, b = 0.5, 7/3
err = m*x + b - y
print(np.sum(err**2))
```

### Summary
- Choose parameters that minimize squared residuals.
- The cost is a surface over parameter space.
- Squared error penalizes large misses strongly.

## Gradient descent in one variable

### Intuition
Gradient descent replaces exact solving with repeated local
movement.  If the derivative is negative, the function decreases
by stepping to the right; if the derivative is positive, step
left.

### Definitions and formulas
For a one-dimensional function,
$$x_{k+1}=x_k-\alpha f'(x_k).$$
The learning rate $\alpha$ controls how much of the slope is
used as a step.

### Fully worked example
Let $f(x)=(x-4)^2$, so $f'(x)=2(x-4)$.
Start at $x_0=1$ with $\alpha=0.25$.
Then $x_1=1-0.25(-6)=2.5$.
Next $x_2=2.5-0.25(-3)=3.25$.
Then $x_3=3.25-0.25(-1.5)=3.625$.
The iterates approach the minimum at $x=4$.

> ⚠️ For minimization, subtract the derivative term.  Adding it
> performs gradient ascent.

### NumPy snippet
```python
x = 1.0
alpha = 0.25
for step in range(3):
    x = x - alpha * 2*(x-4)
    print(step + 1, x)
```

### Summary
- Iterate instead of solving exactly.
- The derivative sign determines left or right.
- The step size depends on both slope and learning rate.

## Learning rate and local minima

### Intuition
The learning rate is the knob that turns slope into motion.
Too small makes progress crawl; too large can bounce across the
valley.  A non-convex curve can also trap descent in a local
minimum that is not the best point overall.

### Definitions and formulas
For $f(x)=x^2$, gradient descent gives
$x_{k+1}=x_k-2\alpha x_k=(1-2\alpha)x_k$.
The factor $1-2\alpha$ explains slow convergence, oscillation,
and divergence in one line.

### Fully worked example
Start with $x_0=1$ and $\alpha=1.1$ for $f(x)=x^2$.
Then $x_1=(1-2.2)\cdot1=-1.2$.
The next point is $x_2=(-1.2)(-1.2)=1.44$.
The signs alternate and the magnitude grows, so this learning
rate is too large.

> ⚠️ A few decreasing steps do not prove a global minimum.
> Different starting points are useful on wavy objectives.

### NumPy snippet
```python
def run(alpha):
    x = 1.0
    history = []
    for _ in range(5):
        x = x - alpha * 2*x
        history.append(round(x, 3))
    return history
print(run(0.1))
print(run(1.1))
```

### Summary
- Small rates are safe but slow.
- Large rates may oscillate or diverge.
- Local minima motivate multiple restarts.

## Gradient descent in two variables

### Intuition
In two variables, the current position is a vector.  The slides'
heat example starts at a point in the room, computes a gradient,
and takes a small step in the opposite direction to find a
cooler point.

### Definitions and formulas
For $p_k=[x_k,y_k]^T$,
$$p_{k+1}=p_k-\alpha\nabla f(p_k).$$
Both coordinates must be updated from the same old gradient.

### Fully worked example
Let $f(x,y)=(x-1)^2+2(y+1)^2$.
At $(3,1)$, the gradient is
$[2(3-1),4(1+1)]=[4,8]$.
With $\alpha=0.1$,
$$[x_1,y_1]=[3,1]-0.1[4,8]=[2.6,0.2].$$
The new point is closer to the minimizer $(1,-1)$.

> ⚠️ Do not update $x$ and then use the new $x$ to compute the
> old-step gradient for $y$.

### NumPy snippet
```python
import numpy as np
p = np.array([3.0, 1.0])
g = np.array([2*(p[0]-1), 4*(p[1]+1)])
p_next = p - 0.1*g
print(p_next)
```

### Summary
- The point and gradient are vectors.
- Negative gradient is steepest local descent.
- Coordinate updates belong to the same iteration.

## Gradient descent for least squares

### Intuition
Least-squares gradient descent applies the same vector update to
model parameters.  For a line, the parameters are $m$ and $b$.
Each step uses all residuals to nudge the line.

### Definitions and formulas
For
$$J(m,b)=\frac1{2n}\sum_i (mx_i+b-y_i)^2,$$
the gradients are
$$J_m=\frac1n\sum_i (mx_i+b-y_i)x_i,$$
$$J_b=\frac1n\sum_i (mx_i+b-y_i).$$
Then update $m\leftarrow m-\alpha J_m$ and
$b\leftarrow b-\alpha J_b$.

### Fully worked example
Use tiny data $(0,1),(1,3),(2,5)$, which lies on $y=2x+1$.
At $m=0,b=0$, predictions are $0,0,0$ and residuals are
$-1,-3,-5$.  Thus
$J_m=(-1\cdot0-3\cdot1-5\cdot2)/3=-13/3$ and
$J_b=(-1-3-5)/3=-3$.  With $\alpha=0.1$,
$m$ becomes $0.4333$ and $b$ becomes $0.3$.

> ⚠️ The factor $1/(2n)$ simplifies the derivative; forgetting
> the average changes the effective learning rate.

### NumPy snippet
```python
import numpy as np
x = np.array([0., 1., 2.])
y = np.array([1., 3., 5.])
m, b, alpha = 0.0, 0.0, 0.1
err = m*x + b - y
m -= alpha * np.mean(err*x)
b -= alpha * np.mean(err)
print(round(m, 4), round(b, 4))
```

### Summary
- Residuals drive the parameter gradient.
- Averaging keeps scale stable across data sizes.
- Convex squared loss has one global basin for a line.
