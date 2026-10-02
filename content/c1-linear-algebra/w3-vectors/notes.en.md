# Week 3: Vectors, matrices, and linear transformations

This week connects geometric intuition with the computations used in Machine Learning. Vectors represent data and weights; norms measure size; dot products turn many features into one score; matrices describe linear transformations; and matrix multiplication either composes transformations or processes a whole batch of examples at once. The slide examples about spam detection and perceptrons show that these operations are not just algebraic notation: they are the working language of neural networks.

## Vectors and coordinates

### Intuition
A vector has two complementary views. Geometrically, it is an arrow with direction and length. Computationally, it is an ordered list of numbers. The vector $(4,3)$ means move 4 units along the $x$ axis and 3 units along the $y$ axis. The vector $(4,3,1)$ adds a third coordinate; in ML it could just as naturally mean a data point with three features.

Many ML objects become vectors: an email can be a vector of word counts, an image can be a vector of pixel intensities, and a user can be a vector of preferences. Once data is written as vectors, we can compare, score, transform, and batch-process it with linear algebra.

### Definitions and notation
A column vector is often written as:
$$u=\begin{bmatrix}4\\3\end{bmatrix}.$$
Its first entry is $u_1=4$ and its second entry is $u_2=3$. Two vectors are equal when all corresponding entries are equal. If two arrows are drawn at different locations but have the same displacement $(4,3)$, linear algebra treats them as the same free vector.

### Fully worked example
Find the vector from $A=(1,2)$ to $B=(5,5)$.

1. A vector “from $A$ to $B$” is endpoint minus start point: $B-A$.
2. Subtract coordinates:
   $$B-A=(5-1,\;5-2).$$
3. Simplify:
   $$B-A=(4,3).$$
4. Geometric check: starting at $(1,2)$, moving right 4 and up 3 lands at $(5,5)$.

### NumPy snippet
```python
import numpy as np
A = np.array([1, 2])
B = np.array([5, 5])
u = B - A          # array([4, 3])
```

> ⚠️ Common mistakes: confusing the point $(4,3)$ with the displacement vector $(4,3)$; treating vectors like unordered sets; or forgetting that $(4,3)\neq(3,4)$ because the coordinate order matters.

### Summary
- A vector is an ordered list of numbers or a directed arrow.
- The vector from $A$ to $B$ is $B-A$.
- In ML, vectors commonly represent features, weights, embeddings, and data examples.

## Norms and distances

### Intuition
A norm measures the “size” of a vector. The slides contrast two common norms. The $L_1$ norm is like taxicab distance on a street grid: you add horizontal and vertical movement. The $L_2$ norm is straight-line distance: the shortest path “as the helicopter flies.”

The distance between two vectors is not found by measuring each vector separately. First compute the difference vector, then measure that difference:
$$d(u,v)=\|u-v\|.$$
This is why vector subtraction and norms are tightly connected.

### Formulas
For $u=(a,b)$:
$$\|u\|_1=|a|+|b|,$$
$$\|u\|_2=\sqrt{a^2+b^2}.$$
For two vectors:
$$d_1(u,v)=\|u-v\|_1,\qquad d_2(u,v)=\|u-v\|_2.$$

### Fully worked example
Let $u=(6,2)$ and $v=(1,5)$. Compute $L_1$ and $L_2$ distances.

1. Compute the difference:
   $$u-v=(6-1,\;2-5)=(5,-3).$$
2. $L_1$ distance:
   $$d_1(u,v)=|5|+|-3|=5+3=8.$$
3. $L_2$ distance:
   $$d_2(u,v)=\sqrt{5^2+(-3)^2}=\sqrt{25+9}=\sqrt{34}\approx5.83.$$
4. Interpretation: the grid path has length 8, while the straight-line path is shorter.

### NumPy snippet
```python
import numpy as np
u = np.array([6, 2])
v = np.array([1, 5])
diff = u - v
l1 = np.sum(np.abs(diff))
l2 = np.sqrt(np.sum(diff * diff))
```

> ⚠️ Common mistakes: forgetting absolute values in $L_1$; computing $\sqrt{5}+\sqrt{9}$ instead of $\sqrt{5^2+3^2}$ for $L_2$; or confusing the norm of one vector with the distance between two vectors.

### Summary
- $\|u\|_1$ sums absolute entry sizes.
- $\|u\|_2$ is Euclidean length from the Pythagorean theorem.
- The distance between two vectors is the norm of their difference.

## Direction and scalar multiplication

### Intuition
A vector has both magnitude and direction. The vector $(4,3)$ is not only length 5; it also points upward at an angle from the positive $x$ axis. Multiplying by a scalar stretches, shrinks, or reverses the vector. A positive scalar keeps direction; a negative scalar flips the vector by $180^\circ$.

This idea matters in ML because a learned weight can amplify a feature, reduce it, or reverse its effect. Multiplying by $-2$ means “twice as strong but in the opposite direction.”

### Formulas
For $u=(a,b)$ with $a\neq0$:
$$\tan\theta=\frac{b}{a}.$$
In code, prefer `atan2(b, a)` because it handles the correct quadrant. Scalar multiplication is:
$$\lambda u=\lambda(a,b)=(\lambda a,\lambda b).$$
Norms scale by:
$$\|\lambda u\|=|\lambda|\|u\|.$$

### Fully worked example
Find the direction of $u=(4,3)$ and compute $-2u$.

1. Compute the slope ratio:
   $$\tan\theta=\frac{3}{4}.$$
2. Take arctangent:
   $$\theta=\arctan(3/4)\approx36.87^\circ.$$
3. Scale by $-2$:
   $$-2u=-2(4,3)=(-8,-6).$$
4. Interpretation: the scaled vector is twice as long and points in the opposite direction.

### NumPy snippet
```python
import numpy as np
u = np.array([4, 3])
angle_deg = np.degrees(np.arctan2(u[1], u[0]))
scaled = -2 * u
```

> ⚠️ Common mistakes: using only $\arctan(b/a)$ and getting the wrong quadrant; thinking a negative scalar only changes signs without understanding the geometric reversal; or assuming scalar multiplication changes direction when the scalar is positive.

### Summary
- Positive scalars preserve direction; negative scalars reverse direction.
- Length is multiplied by the scalar’s absolute value.
- Use `atan2` for robust angle computations.

## Vector sums and differences

### Intuition
Vector addition chains movements. If $u$ is one step and $v$ is the next step, then $u+v$ is the single step that has the same total effect. Vector subtraction compares arrow tips: $u-v$ is the displacement from the head of $v$ to the head of $u$ when both tails are at the same origin.

This geometric view explains why distances use $u-v$: first find the displacement between the two vectors, then measure it.

### Formulas
For $u=(a,b)$ and $v=(c,d)$:
$$u+v=(a+c,b+d),$$
$$u-v=(a-c,b-d).$$
Addition is commutative: $u+v=v+u$. Subtraction is not: $u-v=-(v-u)$.

### Fully worked example
Let $u=(4,1)$ and $v=(1,3)$.

1. Add component-wise:
   $$u+v=(4+1,\;1+3)=(5,4).$$
2. Subtract component-wise:
   $$u-v=(4-1,\;1-3)=(3,-2).$$
3. Geometric check: the vector from the tip of $v$ at $(1,3)$ to the tip of $u$ at $(4,1)$ is $(4-1,1-3)=(3,-2)$.

### NumPy snippet
```python
import numpy as np
u = np.array([4, 1])
v = np.array([1, 3])
sum_uv = u + v
diff_uv = u - v
```

> ⚠️ Common mistakes: treating vector subtraction like component-wise multiplication; reversing the order and computing $v-u$ instead of $u-v$; or forgetting that addition is component-wise, not a change in vector length only.

### Summary
- Vector addition and subtraction are component-wise.
- $u+v$ combines displacements.
- $u-v$ measures the displacement between two vector tips and is the basis for distance.

## Dot product

### Intuition
The dot product turns two same-length vectors into one scalar by multiplying corresponding entries and summing. The slide’s shopping example is the core pattern: quantities dotted with prices gives total cost. In ML, features dotted with weights gives a linear score.

If $x$ is a feature vector and $w$ is a weight vector, $w\cdot x$ measures how strongly the example matches the learned weights. Features with larger weights contribute more to the score.

### Formula
For $u=(u_1,\dots,u_n)$ and $v=(v_1,\dots,v_n)$:
$$u\cdot v=\sum_{i=1}^n u_i v_i.$$
The $L_2$ norm is connected to the dot product:
$$\|u\|_2=\sqrt{u\cdot u}.$$

### Fully worked example
Compute $(2,4,1)\cdot(3,5,2)$.

1. Multiply matching entries:
   $$2\cdot3=6,\qquad 4\cdot5=20,\qquad 1\cdot2=2.$$
2. Sum the products:
   $$6+20+2=28.$$
3. Therefore:
   $$(2,4,1)\cdot(3,5,2)=28.$$
4. If these are quantities and prices, the total cost is 28.

### NumPy snippet
```python
import numpy as np
q = np.array([2, 4, 1])
p = np.array([3, 5, 2])
total = np.sum(q * p)   # same result as q @ p
```

> ⚠️ Common mistakes: returning the vector of products $[6,20,2]$ instead of the summed scalar; trying to dot vectors of different lengths; or confusing dot product with distance.

### Summary
- Dot product = multiply matching entries, then sum.
- The result is a scalar.
- $\sqrt{u\cdot u}$ gives the $L_2$ norm.
- Linear models often compute scores using $w\cdot x$.

## Geometric dot product

### Intuition
The dot product also measures alignment. If two vectors point mostly in the same direction, their dot product is positive. If they are perpendicular, it is zero. If they point mostly opposite, it is negative. This is the basis of cosine similarity, projection, orthogonality, and many ML algorithms.

The slide examples with $u=(6,2)$ show this sign behavior: one vector gives dot 0, another gives a positive dot product, and another gives a negative dot product.

### Formula
For two nonzero vectors:
$$u\cdot v=\|u\|\|v\|\cos\theta.$$
Equivalently:
$$\cos\theta=\frac{u\cdot v}{\|u\|\|v\|}.$$
If $u\cdot v=0$, then $\cos\theta=0$ and $\theta=90^\circ$.

### Fully worked example
Let $u=(6,2)$ and $v=(-1,3)$.

1. Compute the dot product:
   $$u\cdot v=6(-1)+2(3)=-6+6=0.$$
2. Both vectors are nonzero.
3. Since the dot product is 0, the angle between them is $90^\circ$.
4. Conclusion: $u$ and $v$ are orthogonal.

For comparison:
$$(6,2)\cdot(2,4)=12+8=20>0,$$
so the angle is acute. But:
$$(6,2)\cdot(-4,1)=-24+2=-22<0,$$
so the angle is obtuse.

### NumPy snippet
```python
import numpy as np
u = np.array([6, 2])
v = np.array([-1, 3])
dot = u @ v
angle = np.degrees(np.arccos(dot / (np.linalg.norm(u) * np.linalg.norm(v))))
```

> ⚠️ Common mistakes: saying “dot product 0 means a $90^\circ$ angle” without excluding the zero vector; forgetting to divide by both norms when computing an angle; or interpreting a negative dot product as “negative length” instead of opposite alignment.

### Summary
- Positive dot: acute angle; zero dot: orthogonal; negative dot: obtuse angle.
- The geometric dot product formula uses cosine.
- Orthogonality is central to projections and optimization.

## Matrix-vector products

### Intuition
A matrix-vector product is many dot products at once. Each row of the matrix acts like a separate weight vector and produces one output entry. This is why a system of equations can be compressed into $Ax=b$.

If $A$ is $m\times n$ and $x$ has $n$ entries, then $Ax$ has $m$ entries. The $i$-th output is row $i$ of $A$ dotted with $x$.

### Formula
$$(Ax)_i=\sum_{j=1}^n A_{ij}x_j.$$
The system:
$$a+b+c=10,\qquad a+2b+c=15,\qquad a+b+2c=12$$
can be written as:
$$
\begin{bmatrix}1&1&1\\1&2&1\\1&1&2\end{bmatrix}
\begin{bmatrix}a\\b\\c\end{bmatrix}
=\begin{bmatrix}10\\15\\12\end{bmatrix}.
$$

### Fully worked example
Compute:
$$A=\begin{bmatrix}1&2\\3&4\end{bmatrix},\qquad x=\begin{bmatrix}2\\1\end{bmatrix}.$$

1. Row 1 dotted with $x$:
   $$1\cdot2+2\cdot1=4.$$
2. Row 2 dotted with $x$:
   $$3\cdot2+4\cdot1=10.$$
3. Stack the outputs:
   $$Ax=\begin{bmatrix}4\\10\end{bmatrix}.$$

### NumPy snippet
```python
import numpy as np
A = np.array([[1, 2],
              [3, 4]])
x = np.array([2, 1])
y = A @ x
```

> ⚠️ Common mistakes: multiplying incompatible shapes; dotting columns with $x$ instead of rows; or forgetting that a $3\times2$ matrix maps a length-2 vector to a length-3 vector.

### Summary
- $Ax$ is a collection of row-vector dot products.
- If $A$ is $m\times n$, then $x$ must have length $n$ and $Ax$ has length $m$.
- Linear systems can be represented as one vector equation.

## Matrices as linear transformations

### Intuition
A matrix is not just a table of numbers; it is a function that transforms vectors. In 2D, a $2\times2$ matrix can stretch, compress, shear, rotate, reflect, or combine these effects across the whole plane.

The key slide idea is that the whole transformation is determined by what happens to the basis vectors $e_1=(1,0)$ and $e_2=(0,1)$. These transformed basis vectors are the columns of the matrix.

### Formula
If:
$$A=\begin{bmatrix}3&1\\1&2\end{bmatrix},$$
then:
$$Ae_1=\begin{bmatrix}3\\1\end{bmatrix},\qquad Ae_2=\begin{bmatrix}1\\2\end{bmatrix}.$$
For any vector $(a,b)$:
$$
A\begin{bmatrix}a\\b\end{bmatrix}
=aAe_1+bAe_2
=\begin{bmatrix}3a+b\\a+2b\end{bmatrix}.
$$

### Fully worked example
Suppose $e_1\mapsto(3,-1)$ and $e_2\mapsto(2,3)$. Build the matrix.

1. The image of $e_1$ is column 1:
   $$\begin{bmatrix}3\\-1\end{bmatrix}.$$
2. The image of $e_2$ is column 2:
   $$\begin{bmatrix}2\\3\end{bmatrix}.$$
3. Put these images into columns:
   $$A=\begin{bmatrix}3&2\\-1&3\end{bmatrix}.$$
4. Check: $A(1,1)=(3,-1)+(2,3)=(5,2)$.

### NumPy snippet
```python
import numpy as np
A = np.array([[3, 2],
              [-1, 3]])
points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
transformed = points @ A.T   # row-point convention
```

> ⚠️ Common mistakes: placing the images of $e_1,e_2$ as rows instead of columns; mixing row-point and column-vector conventions in code; or forgetting that $A(e_1+e_2)=Ae_1+Ae_2$ for a linear transformation.

### Summary
- Matrix columns are transformed basis vectors.
- A linear transformation preserves vector addition and scalar multiplication.
- The unit square maps to a parallelogram determined by the two matrix columns.

## Matrix multiplication

### Intuition
Matrix multiplication composes linear transformations. If a point is transformed by $B$ first and then by $A$, the combined transformation is $AB$. This is why order matters: the rightmost transformation acts first.

Computationally, each entry of $AB$ is a dot product between one row of $A$ and one column of $B$. Matrix multiplication is therefore the natural extension of matrix-vector multiplication.

### Formula
If $A$ is $m\times k$ and $B$ is $k\times n$, then $AB$ is $m\times n$ and:
$$(AB)_{ij}=\text{row}_i(A)\cdot\text{col}_j(B).$$

### Fully worked example
Compute:
$$
A=\begin{bmatrix}2&-1\\0&2\end{bmatrix},\qquad
B=\begin{bmatrix}3&1\\1&2\end{bmatrix}.
$$

1. Entry $(1,1)$:
   $$2\cdot3+(-1)\cdot1=5.$$
2. Entry $(1,2)$:
   $$2\cdot1+(-1)\cdot2=0.$$
3. Entry $(2,1)$:
   $$0\cdot3+2\cdot1=2.$$
4. Entry $(2,2)$:
   $$0\cdot1+2\cdot2=4.$$
5. Therefore:
   $$AB=\begin{bmatrix}5&0\\2&4\end{bmatrix}.$$

### NumPy snippet
```python
import numpy as np
A = np.array([[2, -1],
              [0,  2]])
B = np.array([[3, 1],
              [1, 2]])
C = A @ B
```

> ⚠️ Common mistakes: assuming $AB=BA$; computing with mismatched inner dimensions; or mixing up transformation order. If $B$ happens first and $A$ happens second, the combined matrix is $AB$, not $BA$.

### Summary
- Matrix multiplication is row-dot-column.
- Inner dimensions must match.
- Multiplication order represents composition order and is generally not commutative.

## Identity and inverse matrices

### Intuition
The identity matrix is the “do nothing” transformation: every input vector comes out unchanged. An inverse matrix reverses a linear transformation. If $A$ stretches or shears the plane without losing information, then $A^{-1}$ brings transformed vectors back to where they started.

Not every matrix has an inverse. If a transformation collapses different inputs to the same output, information has been lost and no inverse can recover it. In 2D, this corresponds to determinant zero.

### Formula
$$AI=IA=A,\qquad AA^{-1}=A^{-1}A=I.$$
For:
$$A=\begin{bmatrix}a&b\\c&d\end{bmatrix},$$
if $ad-bc\neq0$, then:
$$A^{-1}=\frac{1}{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}.$$

### Fully worked example
Find the inverse of:
$$A=\begin{bmatrix}5&2\\1&2\end{bmatrix}.$$

1. Compute the determinant:
   $$\det(A)=5\cdot2-2\cdot1=8.$$
2. Since $8\neq0$, the inverse exists.
3. Swap $a,d$ and negate $b,c$:
   $$\begin{bmatrix}2&-2\\-1&5\end{bmatrix}.$$
4. Divide by the determinant:
$$
A^{-1}=\frac18\begin{bmatrix}2&-2\\-1&5\end{bmatrix}
   =\begin{bmatrix}1/4&-1/4\\-1/8&5/8\end{bmatrix}.
$$

A singular example is:
$$\begin{bmatrix}1&1\\2&2\end{bmatrix},$$
because row 2 is twice row 1 and the determinant is 0.

### NumPy snippet
```python
import numpy as np
A = np.array([[5, 2],
              [1, 2]], dtype=float)
det = np.linalg.det(A)
inv = np.linalg.inv(A)  # only when det is not near zero
```

> ⚠️ Common mistakes: applying the inverse formula when the determinant is 0; forgetting to multiply the whole matrix by $\frac{1}{ad-bc}$; or swapping $a,d$ but forgetting to negate $b,c$.

### Summary
- Identity leaves every vector unchanged.
- An inverse reverses a linear transformation.
- A square matrix is invertible when it is non-singular; for $2\times2$, this means determinant nonzero.

## Neural networks with matrices

### Intuition
Neural networks use this week’s operations at scale. A perceptron computes a linear score:
$$s=w\cdot x+b.$$
Then it compares the score with a threshold, or equivalently compares $s$ with 0 after including the bias. With many examples, stack them into a data matrix $X$. With many neurons, stack weights into a matrix $W$. A linear layer is:
$$Y=XW+b.$$

### Worked spam example
The slide uses features “Lottery” and “Win”. Choose:
$$w=\begin{bmatrix}1\\1\end{bmatrix},\qquad \text{threshold}=1.5.$$
Suppose an email has $x=(1,2)$: one “Lottery” and two “Win” occurrences.

1. Compute the score:
   $$w\cdot x=1\cdot1+1\cdot2=3.$$
2. Compare with threshold:
   $$3>1.5.$$
3. Prediction: spam.

Adding bias $b=-1.5$ rewrites the rule as:
$$w\cdot x-1.5>0.$$

### Batch example
With:
$$
X=\begin{bmatrix}1&1\\2&1\\0&0\\0&2\end{bmatrix},\qquad
w=\begin{bmatrix}1\\1\end{bmatrix},
$$
we get:
$$Xw=\begin{bmatrix}2\\3\\0\\2\end{bmatrix}.$$
Comparing each score with $1.5$ gives: spam, spam, not spam, spam.

### NumPy snippet
```python
import numpy as np
X = np.array([[1, 1],
              [2, 1],
              [0, 0],
              [0, 2]])
w = np.array([1, 1])
scores = X @ w
pred = scores > 1.5
```

> ⚠️ Common mistakes: mixing shape conventions. If examples are rows, use $Xw$. If examples are columns, the formula changes. Another common mistake is forgetting the bias/threshold, which forces the decision boundary through the origin instead of shifting it.

### Summary
- A perceptron is a dot product plus bias followed by a threshold.
- Bias can be represented as the weight of a feature that is always 1.
- Matrix multiplication computes scores for many examples and many neurons at once.
