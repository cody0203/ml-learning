# Week 3: Optimization in neural networks and Newton's method

The two PDFs form one long week.
Part 1 builds optimization tools.
Part 2 uses the same derivatives inside perceptrons.

## Gradient descent and backpropagation

### Intuition
Gradient descent asks one local question:
"Which small parameter move lowers the loss fastest?"
The gradient points uphill.
Training therefore walks in the negative gradient direction.
Backpropagation is not a different calculus rule.
It is the chain rule applied in reverse through a graph.

### Definitions and formulas
For parameters $\theta$ and loss $L$:

$$\theta_{k+1}=\theta_k-\alpha\nabla_\theta L.$$

The learning rate $\alpha$ is a step-size knob.
Too large can jump across a valley.
Too small can make training crawl.

### Worked example
Let $L(w)=\frac12(w-3)^2$.
Then $L'(w)=w-3$.
At $w_0=0$ and $\alpha=0.1$:

$$w_1=0-0.1(-3)=0.3.$$

The loss falls from $4.5$ to $3.645$.
Backprop would compute the same derivative
if this loss were the last node of a neural graph.

> ⚠️ Common mistake:
> adding $\alpha\nabla L$ when minimizing.

### NumPy snippet
```python
w = w - alpha * grad_w
```

### Summary
- Gradient descent uses first derivatives.
- Backprop is reverse-mode chain rule.
- The update sign matters.

## Newton's method

### Intuition
Newton's method replaces a curve near the current point
by its tangent line.
The tangent's x-intercept becomes the next guess.
This can converge very quickly near a simple root.

### Definitions and formulas
For root finding:

$$x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)}.$$

For minimizing $g(x)$,
apply root finding to $g'(x)$:

$$x_{k+1}=x_k-\frac{g'(x_k)}{g''(x_k)}.$$

### Worked example
Take $f(x)=x^2-2$ and $x_0=1$.
Then $f(1)=-1$ and $f'(1)=2$.

$$x_1=1-\frac{-1}{2}=1.5.$$

The next step is
$1.5-(0.25/3)=1.4167$,
already close to $\sqrt2$.

> ⚠️ Common mistake:
> using Newton when $f'(x_k)$ is almost zero.

### NumPy snippet
```python
x = x - f(x) / fp(x)
```

### Summary
- Newton uses tangent-line geometry.
- It can be faster than first-order methods.
- Bad starts can make it diverge.

## Newton's method: an example

### Intuition
The slides minimize
$g(x)=e^x-\log x$.
The domain is $x>0$.
Instead of minimizing $g$ directly,
Newton solves $g'(x)=0$.

### Definitions and formulas
Here

$$g'(x)=e^x-\frac1x,$$

and

$$g''(x)=e^x+\frac1{x^2}.$$

So the iteration is

$$x_{k+1}=x_k-\frac{e^{x_k}-1/x_k}{e^{x_k}+1/x_k^2}.$$

### Worked example
With $x_0=0.05$:
$g'(x_0)\approx-18.949$.
$g''(x_0)\approx401.051$.
Thus

$$x_1\approx0.05-(-18.949/401.051)=0.097.$$

The slides continue through $0.183,0.320,0.477$,
then approach $0.567$.

> ⚠️ Common mistake:
> trying $x\le0$, where $\log x$ is invalid.

### NumPy snippet
```python
x = x - (np.exp(x) - 1/x) / (np.exp(x) + 1/x**2)
```

### Summary
- Optimization Newton solves $g'(x)=0$.
- Curvature appears through $g''(x)$.
- Domain checks are part of the method.

## The second derivative

### Intuition
The first derivative is velocity of a function.
The second derivative is acceleration of that velocity.
On a graph, it describes bending.

### Definitions and formulas

$$f''(x)=\frac{d}{dx}f'(x)=\frac{d^2f}{dx^2}.$$

If $f''(x)>0$, the curve bends upward.
If $f''(x)<0$, it bends downward.
If $f''(x)=0$, more information is needed.

### Worked example
Let $f(x)=x^3-3x$.
Then $f'(x)=3x^2-3$.
Also $f''(x)=6x$.
At $x=1$, $f''(1)=6>0$,
so the curve is locally concave up there.

> ⚠️ Common mistake:
> thinking $f''(x)=0$ always means neither min nor max.
> It is only inconclusive.

### NumPy snippet
```python
second = (f(x+h) - 2*f(x) + f(x-h)) / h**2
```

### Summary
- Second derivatives measure curvature.
- Positive suggests a bowl shape.
- Negative suggests a cap shape.

## The Hessian

### Intuition
In two variables, curvature is not one number.
The surface can bend in the $x$ direction,
the $y$ direction,
and in mixed directions.
The Hessian stores all this second-order information.

### Definitions and formulas

$$
H_f(x,y)=
\begin{bmatrix}
f_{xx} & f_{xy}\\
f_{yx} & f_{yy}
\end{bmatrix}.
$$

For smooth functions,
$f_{xy}=f_{yx}$ in most course examples.

### Worked example
For
$f(x,y)=2x^2+3y^2-xy$:

$$f_x=4x-y,\qquad f_y=6y-x.$$

So

$$H=\begin{bmatrix}4&-1\\-1&6\end{bmatrix}.$$

The off-diagonal $-1$ says the $x$ slope
changes when $y$ changes.

> ⚠️ Common mistake:
> putting first derivatives in the Hessian.

### NumPy snippet
```python
H = np.array([[4.0, -1.0], [-1.0, 6.0]])
```

### Summary
- The gradient is first-order information.
- The Hessian is second-order information.
- Mixed partials describe coupling.

## Hessians and concavity

### Intuition
A 1D second derivative has one sign.
A 2D Hessian has directional signs.
Eigenvalues reveal those signs.
All positive gives a local bowl.
All negative gives a local cap.
Mixed signs give a saddle.

### Definitions and formulas
At a critical point:

$$
\lambda_i(H)>0\ \forall i
\Rightarrow \text{local minimum}.
$$

$$
\lambda_i(H)<0\ \forall i
\Rightarrow \text{local maximum}.
$$

Mixed eigenvalue signs indicate a saddle point.

### Worked example
For $H=\begin{bmatrix}4&-1\\-1&6\end{bmatrix}$:

$$\det(H-\lambda I)=\lambda^2-10\lambda+23.$$

The roots are about $3.59$ and $6.41$.
Both are positive,
so the corresponding critical point is a minimum.

> ⚠️ Common mistake:
> using only the determinant when diagonal signs are ignored.

### NumPy snippet
```python
np.linalg.eigvals(H)
```

### Summary
- Positive eigenvalues mean convex curvature.
- Negative eigenvalues mean concave-down curvature.
- A saddle bends both ways.

## Newton's method for two variables

### Intuition
In 1D, Newton divides by a second derivative.
In 2D, "division by curvature" becomes solving
a linear system involving the Hessian.

### Definitions and formulas

$$
\mathbf{x}_{k+1}
=\mathbf{x}_k-H(\mathbf{x}_k)^{-1}\nabla f(\mathbf{x}_k).
$$

In code, solve
$H\Delta=\nabla f$
and set $\mathbf{x}_{k+1}=\mathbf{x}_k-\Delta$.

### Worked example
Let
$f=x^2+2y^2-2x-8y$.
Then
$\nabla f=(2x-2,4y-8)$
and
$H=\begin{bmatrix}2&0\\0&4\end{bmatrix}$.
At $(0,0)$, the gradient is $(-2,-8)$.
Solving $H\Delta=(-2,-8)$ gives $\Delta=(-1,-2)$.
Thus the next point is $(1,2)$.

> ⚠️ Common mistake:
> multiplying by $H$ instead of solving with $H$.

### NumPy snippet
```python
delta = np.linalg.solve(H, grad)
x = x - delta
```

### Summary
- Newton's two-variable step uses the Hessian.
- The order of matrix operations matters.
- Solving is preferred to explicit inversion.

## Regression with a perceptron

### Intuition
A regression perceptron predicts a number.
For the house example,
features can be size and number of rooms.
Weights express how much each feature changes price.

### Definitions and formulas

$$\hat y=w_1x_1+w_2x_2+b.$$

This is a single linear neuron
without a nonlinear activation.

### Worked example
Use scaled units:
$x=(2,4)$, $w=(10,5)$, and $b=3$.
Then

$$\hat y=10(2)+5(4)+3=43.$$

If the true target is $45$,
the residual is $2$.

> ⚠️ Common mistake:
> forgetting the bias term when batching examples.

### NumPy snippet
```python
yhat = X @ w + b
```

### Summary
- Regression output is continuous.
- The neuron is a weighted sum.
- Bias shifts every prediction.

## Regression with a perceptron: loss function

### Intuition
Loss turns a prediction error into one number.
The slides use squared error,
so over- and under-prediction are both penalized.
The factor $\frac12$ makes the derivative cleaner.

### Definitions and formulas

$$L(y,\hat y)=\frac12(y-\hat y)^2.$$

The derivative with respect to $\hat y$ is
$\hat y-y$.

### Worked example
If $y=5$ and $\hat y=3$,
then the error is $2$.
The loss is

$$\frac12(5-3)^2=2.$$

If $\hat y=1$ instead,
the loss becomes $8$.

> ⚠️ Common mistake:
> averaging before squaring individual errors.

### NumPy snippet
```python
loss = 0.5 * (y - yhat)**2
```

### Summary
- Squaring removes the sign of the residual.
- Larger errors get much larger penalties.
- The loss is what training minimizes.

## Regression with a perceptron: gradient descent

### Intuition
The prediction depends on $w_1,w_2,b$.
The loss depends on the prediction.
The chain rule links each parameter to the loss.

### Definitions and formulas

$$
\frac{\partial L}{\partial w_i}
=(\hat y-y)x_i,
$$

and

$$\frac{\partial L}{\partial b}=\hat y-y.$$

### Worked example
Let $x=(2,1)$, $y=5$, and $\hat y=3$.
Then $\hat y-y=-2$.
The gradients are
$dw=(-4,-2)$ and $db=-2$.
With $\alpha=0.1$,
the update adds $(0.4,0.2)$ to the weights
and adds $0.2$ to the bias.

> ⚠️ Common mistake:
> using $(y-\hat y)x_i$ as the gradient
> without also changing the update sign.

### NumPy snippet
```python
err = yhat - y
dw = err * x
db = err
```

### Summary
- Each feature scales its own weight gradient.
- Bias gradient has feature value $1$.
- Sign conventions must be consistent.

## Classification with a perceptron

### Intuition
Classification needs a discrete decision
or a probability-like score.
The slides count words such as Aack and Beep,
then feed those counts into a weighted sum.

### Definitions and formulas

$$z=w\cdot x+b,\qquad \hat y=\sigma(z).$$

The raw score $z$ can be any real number.
The sigmoid turns it into a number in $(0,1)$.

### Worked example
Let $x=(1,3)$, $w=(4.5,1.5)$, and $b=2$.
Then

$$z=4.5(1)+1.5(3)+2=11.$$

The sigmoid is very close to $1$,
so the model is confident in class $1$.

> ⚠️ Common mistake:
> treating the raw score as a probability.

### NumPy snippet
```python
z = x @ w + b
p = 1 / (1 + np.exp(-z))
```

### Summary
- The linear score comes before activation.
- Sigmoid makes binary probabilities.
- Features must be numeric counts or encodings.

## Classification with a perceptron: the sigmoid function

### Intuition
The sigmoid is an S-shaped squashing function.
Large negative scores become probabilities near $0$.
Large positive scores become probabilities near $1$.
Near zero, the model is uncertain.

### Definitions and formulas

$$\sigma(z)=\frac1{1+e^{-z}}.$$

Its derivative is

$$\sigma'(z)=\sigma(z)(1-\sigma(z)).$$

### Worked example
At $z=0$:

$$\sigma(0)=\frac12.$$

The derivative is

$$0.5(1-0.5)=0.25.$$

That is the largest slope of sigmoid.

> ⚠️ Common mistake:
> forgetting that the derivative uses the sigmoid value.

### NumPy snippet
```python
s = 1 / (1 + np.exp(-z))
sp = s * (1 - s)
```

### Summary
- Sigmoid outputs lie between 0 and 1.
- Its derivative is largest at the origin.
- Saturated scores give small gradients.

## Classification with a perceptron: gradient descent

### Intuition
With sigmoid and binary log-loss,
many chain-rule factors cancel.
The final gradient looks simple:
prediction minus target, times the input.

### Definitions and formulas

$$
\frac{\partial J}{\partial w_i}
=(\hat y-y)x_i,
$$

and

$$\frac{\partial J}{\partial b}=\hat y-y.$$

### Worked example
If $\hat y=0.8$, $y=1$, and $x=(3,1)$,
then $\hat y-y=-0.2$.
So
$dw=(-0.6,-0.2)$ and $db=-0.2$.
For $\alpha=0.5$,
the weights move by $(0.3,0.1)$.

> ⚠️ Common mistake:
> multiplying by another sigmoid derivative
> after the log-loss cancellation.

### NumPy snippet
```python
err = yhat - y
dw = err * x
b = b - alpha * err
```

### Summary
- Log-loss pairs nicely with sigmoid.
- The error term is $\hat y-y$.
- Confident wrong predictions create large loss.

## Classification with a perceptron: calculating the derivatives

### Intuition
The slides expand derivatives one link at a time.
For a weight, the route is:
loss to prediction,
prediction to score,
score to that weight.

### Definitions and formulas

$$
\frac{\partial L}{\partial w_i}
=\frac{\partial L}{\partial \hat y}
\frac{\partial \hat y}{\partial z}
\frac{\partial z}{\partial w_i}.
$$

Read the displayed product as multiplication of links.
Also,
$\partial z/\partial w_i=x_i$.

### Worked example
Suppose
$dL/d\hat y=2$,
$d\hat y/dz=0.25$,
and $dz/dw_1=3$.
Then

$$dL/dw_1=2(0.25)(3)=1.5.$$

> ⚠️ Common mistake:
> using $w_i$ instead of $x_i$
> for $\partial z/\partial w_i$.

### NumPy snippet
```python
grad_wi = dL_dyhat * dyhat_dz * x[i]
```

### Summary
- Backprop multiplies local derivatives.
- Each edge contributes one factor.
- The input value appears in weight gradients.

## Classification with a neural network

### Intuition
A 2-2-1 network has two inputs,
two hidden neurons,
and one output neuron.
Hidden activations become new features
for the output neuron.

### Definitions and formulas

$$
z^{[1]}=W^{[1]}x+b^{[1]},\quad
a^{[1]}=\sigma(z^{[1]}).
$$

Then

$$\hat y=\sigma(W^{[2]}a^{[1]}+b^{[2]}).$$

### Worked example
Let $W^{[1]}$ be the identity,
$b^{[1]}=(1,-1)$,
and $x=(2,3)$.
Then
$z^{[1]}=(3,2)$.
After sigmoid,
$a^{[1]}\approx(0.953,0.881)$.
Those two numbers feed the output layer.

> ⚠️ Common mistake:
> mixing column-vector and row-vector conventions.

### NumPy snippet
```python
a1 = sigmoid(W1 @ x + b1)
yhat = sigmoid(W2 @ a1 + b2)
```

### Summary
- Hidden units create learned features.
- The output layer is another perceptron.
- Shapes must be checked at each layer.

## Classification with a neural network: minimizing log-loss

### Intuition
The final output is a probability.
Binary log-loss rewards high probability
on the true class.
It punishes confident wrong predictions sharply.

### Definitions and formulas

$$
J(y,\hat y)=
-y\log(\hat y)-(1-y)\log(1-\hat y).
$$

For the output layer with sigmoid,
the local error is $\hat y-y$.

### Worked example
If $y=1$ and $\hat y=0.9$:

$$J=-\log(0.9)\approx0.105.$$

If the same example gets $\hat y=0.1$:

$$J=-\log(0.1)\approx2.303.$$

The second prediction is much more costly.

> ⚠️ Common mistake:
> computing $\log(0)$ when predictions are clipped nowhere.

### NumPy snippet
```python
p = np.clip(p, 1e-12, 1-1e-12)
loss = -(y*np.log(p) + (1-y)*np.log(1-p))
```

### Summary
- Log-loss is asymmetric around the true label.
- Clipping protects numerical code.
- Backprop uses this loss to tune all weights.

## Conclusion

### Intuition
This week connects calculus and neural networks.
Gradient descent supplies the training move.
Newton's method shows how curvature can improve a move.
Backprop supplies gradients for many parameters.

### Definitions and formulas
A practical training loop is:

$$
\text{forward}\rightarrow\text{loss}
\rightarrow\text{backprop}\rightarrow\text{update}.
$$

Newton's ideal step is powerful,
but full Hessians are usually too large in deep learning.

### Worked example
For a small quadratic,
Newton may jump to the optimum in one step.
For a network with millions of weights,
forming the Hessian is usually unrealistic.
First-order optimizers trade exact curvature
for cheap repeated steps.

> ⚠️ Common mistake:
> believing one optimizer is best for every scale.

### NumPy snippet
```python
for batch in data:
    loss, grads = value_and_grad(params, batch)
    params = params - lr * grads
```

### Summary
- Gradient descent is the workhorse.
- Newton explains curvature.
- Backprop makes gradients affordable.
