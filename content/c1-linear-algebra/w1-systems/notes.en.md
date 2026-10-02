# Week 1: Systems of linear equations

## ML motivation and matrix operations

### Intuition
In machine learning, data is rarely a single number. An image has thousands of pixels, a data table has many feature columns, and a neural network has many layers of weights. Linear algebra gives us the language for packaging those numbers into vectors and matrices, then transforming them with matrix operations.

Week 1 begins with a small but foundational question: when we have several linear pieces of information about unknown variables, is that information enough to determine the variables? If it is enough, there is one solution. If it repeats itself, there are many possible solutions. If it contradicts itself, there is no solution. The same intuition appears in ML: are the data and constraints independent enough to identify the model?

### Definitions and formulas
A linear system is commonly written as:

$$
Ax=b
$$

where $A$ is the coefficient matrix, $x$ is the vector of unknowns, and $b$ is the right-hand-side vector. In a simple linear ML layer, we also see:

$$
y=Wx
$$

where $x$ is an input vector, $W$ is a weight matrix, and $y$ is the output before any bias or activation function.

### Worked example
Suppose a toy model produces two scores:

$$
\begin{cases}
s_1=a+b\\
s_2=a+2b
\end{cases}
$$

For a particular input, we observe $s_1=10$ and $s_2=12$. Then:

$$
\begin{cases}
a+b=10\\
a+2b=12
\end{cases}
$$

Subtract the first equation from the second:

$$
(a+2b)-(a+b)=12-10
$$

$$
b=2
$$

Substitute into $a+b=10$:

$$
a+2=10 \Rightarrow a=8
$$

So the two independent scores recover the unique pair $(a,b)=(8,2)$.

### Related NumPy snippet
```python
import numpy as np

W = np.array([[1, 1],
              [1, 2]])
x = np.array([8, 2])
print(W @ x)  # [10 12]
```

Each row of $W$ is one linear constraint; `W @ x` computes both left-hand sides at once.

> ⚠️ Common mistakes: Do not conclude “two equations and two unknowns must have one solution.” The equations may repeat the same information or contradict each other. Also, ML is not purely linear; nonlinearities matter, but linear blocks remain a computational backbone.

### Summary
- Matrices represent many numbers and many computations at once.
- Linear systems are a small model for sufficient, redundant, or contradictory information.
- Matrix products appear in image processing, feature tables, embeddings, and neural networks.
- Before solving, ask whether the constraints are independent and consistent.

## Information systems: complete, redundant, contradictory

### Intuition
The slides start with sentence systems about a dog, cat, and bird to explain “systems” before any arithmetic. An information system is a set of statements describing one situation. Some systems determine one conclusion, some repeat what is already known, and some contradict themselves.

This is the intuitive version of singularity. If information is sufficient and consistent, there is one state. If information is redundant, multiple states remain possible. If information is contradictory, no state satisfies everything.

### Definitions and formulas
- **Complete / non-singular**: enough consistent information to determine exactly one result.
- **Redundant / singular**: repeated or implied information leaves multiple possibilities.
- **Contradictory / singular**: statements cannot all be true, so no possibility exists.

For equation systems later, these correspond to:

$$
\text{unique solution},\quad \text{infinitely many solutions},\quad \text{no solution}
$$

### Worked example
Consider the sentence system:

1. Among the dog, cat, and bird, exactly one is red.
2. Among the dog and cat, exactly one is orange.
3. The dog is black.

Reason step by step:

1. The dog is already black, so it is not the orange one among dog/cat.
2. Since exactly one of dog/cat is orange, the cat must be orange.
3. The dog is black and the cat is orange, so the remaining animal is the bird.
4. Since exactly one animal is red, the bird must be red.

This system is complete/non-singular for the question “what color is the bird?” because it gives one answer: the bird is red.

### Related NumPy snippet
Sentence systems do not need NumPy, but we can mimic the idea of checking possibilities:

```python
animals = ["dog", "cat", "bird"]
colors = {"dog": "black", "cat": "orange", "bird": "red"}
print(colors["bird"])  # red
```

Once we move to equations, “checking possibilities” becomes solving systems or checking rank/determinant.

> ⚠️ Common mistakes: Singular does not mean only “no solution.” Redundant systems are also singular, but they have many possible solutions. Contradictory systems are the no-solution case. Repeating a statement does not make a system stronger.

### Summary
- Complete: enough consistent information for one conclusion.
- Redundant: extra or repeated information, many possibilities remain.
- Contradictory: statements cannot all be true.
- This is the natural-language version of unique / infinite / none.

## From sentences to systems of equations

### Intuition
When a sentence contains quantities and totals, we can assign variables and write equations. “One apple and one banana cost 10” becomes $a+b=10$. Each shopping day gives a new constraint. If the days provide independent information, we can determine each price.

The important task is not only “find the numbers,” but also classify the system. A system may have a unique solution, infinitely many solutions, or no solution.

### Definitions and formulas
With two variables $a,b$, a linear system has the form:

$$
\begin{cases}
p_1a+q_1b=r_1\\
p_2a+q_2b=r_2
\end{cases}
$$

The basic technique is **elimination**: add, subtract, or scale equations by nonzero constants to remove a variable.

### Worked example
Day 1: one apple and one banana cost 10.

Day 2: one apple and two bananas cost 12.

Let $a$ be the apple price and $b$ the banana price:

$$
\begin{cases}
a+b=10\\
a+2b=12
\end{cases}
$$

Step 1: subtract equation 1 from equation 2:

$$
(a+2b)-(a+b)=12-10
$$

$$
b=2
$$

Step 2: substitute $b=2$ into equation 1:

$$
a+2=10
$$

$$
a=8
$$

Step 3: check:

$$
8+2=10,\quad 8+2\cdot2=12
$$

So an apple costs 8 and a banana costs 2.

### Related NumPy snippet
```python
import numpy as np

A = np.array([[1, 1],
              [1, 2]], dtype=float)
b = np.array([10, 12], dtype=float)
x = np.linalg.solve(A, b)
print(x)  # [8. 2.]
```

For learning, solve small systems by hand first; NumPy is useful for checking and for larger systems.

> ⚠️ Common mistakes: If the second equation is $2a+2b=20$, it is only twice the first equation $a+b=10$ and gives no new information. If it is $2a+2b=24$, it contradicts the first equation because doubling the first equation should give 20, not 24.

### Summary
- Words → variables → equations.
- Elimination is the core hand method for small systems.
- Scaling an equation by a nonzero constant does not create new information.
- Always verify a solution by substituting it back into every equation.

## Linear equations as lines

### Intuition
In two variables, a linear equation is a line. Every point on the line is a solution of that one equation. A two-equation system asks for points common to two lines.

Geometry makes the three cases visible: two lines cross at one point, coincide completely, or are distinct and parallel. These are exactly unique, infinite, and no-solution cases.

### Definitions and formulas
A two-variable linear equation has the form:

$$
pa+qb=r
$$

If $q\ne0$, it can be written as:

$$
b=-\frac{p}{q}a+\frac{r}{q}
$$

The slope is $-\frac{p}{q}$ and the vertical intercept is $\frac{r}{q}$.

### Worked example
Consider:

$$
\begin{cases}
a+b=10\\
a+2b=12
\end{cases}
$$

The first line is:

$$
b=-a+10
$$

Some points on it are $(10,0)$, $(0,10)$, and $(8,2)$.

The second line is:

$$
2b=-a+12 \Rightarrow b=-\frac12a+6
$$

Some points on it are $(12,0)$, $(0,6)$, and $(8,2)$.

The two slopes are different, so the lines intersect. Their common point is $(8,2)$ because:

$$
8+2=10,\quad 8+2\cdot2=12
$$

### Related NumPy snippet
```python
import numpy as np

A = np.array([[1, 1],
              [1, 2]], dtype=float)
b = np.array([10, 12], dtype=float)
point = np.linalg.solve(A, b)
print(point)  # [8. 2.]
```

If the determinant of $A$ is zero, `np.linalg.solve` will fail because the two lines do not have a unique intersection.

> ⚠️ Common mistakes: Do not look at two equations that “look different” and immediately conclude they intersect. $a+b=10$ and $2a+2b=24$ have the same slope but different intercepts, so they are parallel and inconsistent.

### Summary
- One equation in two variables is a line.
- Intersecting lines: one solution, non-singular.
- Coincident lines: infinitely many solutions, singular redundant.
- Distinct parallel lines: no solution, singular contradictory.

## Systems as matrices and singularity

### Intuition
Matrix form separates two things: the coefficient structure and the observed constants. The coefficient matrix $A$ says how variables are combined. The vector $b$ says what those combinations must equal.

Singularity is a property of $A$. If $A$ is non-singular, the transformation from $x$ to $Ax$ does not lose information and can be reversed. If $A$ is singular, at least one constraint depends on another, so information is missing or may conflict with $b$.

### Definitions and formulas
The system:

$$
\begin{cases}
a+b=10\\
a+2b=12
\end{cases}
$$

has:

$$
A=\begin{bmatrix}1&1\\1&2\end{bmatrix},\quad
x=\begin{bmatrix}a\\b\end{bmatrix},\quad
b=\begin{bmatrix}10\\12\end{bmatrix}
$$

and:

$$
Ax=b
$$

For a square matrix:

- non-singular: one solution for every $b$;
- singular: infinitely many solutions or no solution, depending on $b$.

### Worked example
Compare two coefficient matrices:

$$
A_1=\begin{bmatrix}1&1\\1&2\end{bmatrix}
$$

The rows are not multiples of each other, so the system can have a unique solution.

$$
A_2=\begin{bmatrix}1&1\\2&2\end{bmatrix}
$$

Row 2 = 2 × row 1. Therefore the second equation adds no new direction of information. If $b=\begin{bmatrix}10\\20\end{bmatrix}$, the system has infinitely many solutions because the two equations are the same line. If $b=\begin{bmatrix}10\\24\end{bmatrix}$, the system has no solution because the same left-hand side is forced to equal incompatible values.

### Related NumPy snippet
```python
import numpy as np

A = np.array([[1, 1],
              [2, 2]], dtype=float)
print(np.linalg.matrix_rank(A))  # 1
print(np.linalg.det(A))          # 0.0
```

Rank smaller than the number of variables indicates missing independent constraints.

> ⚠️ Common mistakes: Do not include the right-hand-side column when computing the determinant of the coefficient matrix. The augmented matrix $[A|b]$ is useful for consistency, but singular/non-singular for the coefficient matrix is a property of $A$ alone.

### Summary
- $A$ holds coefficients, $x$ holds unknowns, and $b$ holds constants.
- Singularity depends on $A$.
- For singular $A$, the same coefficient matrix may produce infinitely many solutions or no solution depending on $b$.
- Rank and determinant are tools for checking the structure of $A$.

## Linear dependence and independence

### Intuition
An equation is useful only if it adds a new constraint. If a row of a matrix can be built from other rows, it does not carry independent information. That is linear dependence.

In 2×2 matrices, dependence is often easy to see: one row is a multiple of the other. In 3×3 matrices, it is subtler: one row may be a sum, difference, or average of other rows.

### Definitions and formulas
Rows $r_1,\ldots,r_k$ are linearly dependent if there are coefficients, not all zero, such that:

$$
c_1r_1+c_2r_2+\cdots+c_kr_k=0
$$

If no such relationship exists, they are linearly independent.

For square matrices in this week:

$$
\text{row independence} \Longleftrightarrow \text{non-singular} \Longleftrightarrow \det(A)\ne0
$$

### Worked example
Consider:

$$
A=\begin{bmatrix}
1&0&0\\
0&1&0\\
1&1&0
\end{bmatrix}
$$

Let the rows be:

$$
r_1=(1,0,0),\quad r_2=(0,1,0),\quad r_3=(1,1,0)
$$

We see:

$$
r_1+r_2=(1,0,0)+(0,1,0)=(1,1,0)=r_3
$$

So row 3 depends on rows 1 and 2. The matrix does not have three independent constraints, so it is singular.

### Related NumPy snippet
```python
import numpy as np

A = np.array([[1, 0, 0],
              [0, 1, 0],
              [1, 1, 0]], dtype=float)
print(np.linalg.matrix_rank(A))  # 2
```

Rank 2 means there are only two independent row directions.

> ⚠️ Common mistakes: Do not only check whether one row is a multiple of another. In 3×3 matrices, a dependent row may be the sum of two other rows, as in the example above.

### Summary
- Linear dependence means redundant constraints.
- Linear independence means each row adds new information.
- Rank counts the number of independent row/column directions.
- For square matrices, dependence is directly connected to determinant 0.

## The 2×2 determinant

### Intuition
The 2×2 determinant is a single number that summarizes whether two constraints are independent. Geometrically, it measures the signed area of the parallelogram formed by two row or column vectors. If the area is zero, the vectors lie on the same line, so the transformation has flattened space and lost information.

### Definitions and formulas
For:

$$
A=\begin{bmatrix}a&b\\c&d\end{bmatrix}
$$

the determinant is:

$$
\det(A)=ad-bc
$$

If $\det(A)=0$, the matrix is singular. If $\det(A)\ne0$, it is non-singular.

### Worked example
Matrix 1:

$$
A=\begin{bmatrix}5&1\\-1&3\end{bmatrix}
$$

Compute:

$$
\det(A)=5\cdot3-1\cdot(-1)
$$

$$
=15+1=16
$$

Since $16\ne0$, $A$ is non-singular.

Matrix 2:

$$
B=\begin{bmatrix}2&-1\\-6&3\end{bmatrix}
$$

Compute:

$$
\det(B)=2\cdot3-(-1)\cdot(-6)
$$

$$
=6-6=0
$$

Since the determinant is 0, $B$ is singular.

### Related NumPy snippet
```python
import numpy as np

A = np.array([[5, 1],
              [-1, 3]], dtype=float)
print(np.linalg.det(A))  # approximately 16
```

When learning the formula, compute $ad-bc$ by hand first, then use NumPy as a check.

> ⚠️ Common mistakes: The most common error is sign handling. For $(-1)(-6)$, the product is $+6$, so $6-6=0$, not $6-(-6)=12$. Also do not use $ad+bc$.

### Summary
- 2×2 determinant = main diagonal minus off diagonal.
- Determinant 0 ⇔ singular ⇔ no unique solution.
- Nonzero determinant ⇔ non-singular ⇔ one solution for every $b$.
- The minus sign in $ad-bc$ is essential.

## Systems of equations in 3 variables

### Intuition
With three variables, we usually need three independent constraints to determine one solution. But “three equations” does not automatically mean “enough information.” If one equation follows from the others, freedom remains. If equations conflict, there is no solution.

In the fruit example from the slides, three shopping days provide enough independent information to determine apple, banana, and cherry prices.

### Definitions and formulas
A 3×3 system has the form:

$$
\begin{cases}
a_{11}x+a_{12}y+a_{13}z=b_1\\
a_{21}x+a_{22}y+a_{23}z=b_2\\
a_{31}x+a_{32}y+a_{33}z=b_3
\end{cases}
$$

If the 3×3 coefficient matrix is non-singular, the system has a unique solution. If it is singular, the right-hand side determines whether there are infinitely many solutions or none.

### Worked example
The fruit system:

$$
\begin{cases}
a+b+c=10\\
a+2b+c=15\\
a+b+2c=12
\end{cases}
$$

Step 1: subtract equation 1 from equation 2:

$$
(a+2b+c)-(a+b+c)=15-10
$$

$$
b=5
$$

Step 2: subtract equation 1 from equation 3:

$$
(a+b+2c)-(a+b+c)=12-10
$$

$$
c=2
$$

Step 3: substitute into equation 1:

$$
a+5+2=10
$$

$$
a=3
$$

Step 4: check:

$$
3+5+2=10,\quad 3+2\cdot5+2=15,\quad 3+5+2\cdot2=12
$$

Therefore $(a,b,c)=(3,5,2)$.

### Related NumPy snippet
```python
import numpy as np

A = np.array([[1, 1, 1],
              [1, 2, 1],
              [1, 1, 2]], dtype=float)
b = np.array([10, 15, 12], dtype=float)
print(np.linalg.solve(A, b))  # [3. 5. 2.]
```

### Common mistakes
> ⚠️ Do not conclude a 3-equation 3-variable system has a unique solution just because the counts match. For example, $a+b+c=10$, $2a+2b+2c=20$, $3a+3b+3c=30$ repeat the same constraint and have infinitely many solutions.

### Summary
- 3×3 systems extend the 2×2 ideas directly.
- Elimination is still the main hand method.
- A unique solution requires linear independence and consistency.
- A singular system may have infinitely many solutions or no solution.

## Planes and 3D geometry

### Intuition
In two variables, one linear equation is a line. In three variables, one linear equation is a plane. The solutions of a three-equation system are the common intersection of three planes.

3D geometry is richer than 2D geometry. Three planes may meet at one point, share a line, coincide as a plane, or have no common point. Therefore “not unique” has several possible shapes.

### Definitions and formulas
A plane equation is:

$$
pa+qb+rc=s
$$

If a point $(a_0,b_0,c_0)$ satisfies:

$$
pa_0+qb_0+rc_0=s
$$

then the point lies on the plane.

### Worked example
Consider the plane:

$$
a+b+c=1
$$

Check three points:

$$
(1,0,0):\quad 1+0+0=1
$$

$$
(0,1,0):\quad 0+1+0=1
$$

$$
(0,0,1):\quad 0+0+1=1
$$

All three points lie on the plane. They are the intercepts with the three coordinate axes.

For the plane:

$$
3a-5b+2c=0
$$

the origin $(0,0,0)$ lies on the plane because:

$$
3\cdot0-5\cdot0+2\cdot0=0
$$

### Related NumPy snippet
```python
import numpy as np

normal = np.array([1, 1, 1])
point = np.array([1, 0, 0])
rhs = 1
print(normal @ point == rhs)  # True
```

The coefficient vector $(p,q,r)$ is the normal vector of the plane $pa+qb+rc=s$.

> ⚠️ Common mistakes: Do not imagine three planes exactly like two lines in 2D. Two planes may intersect in a line, and three planes may have no common intersection even if each pair intersects.

### Summary
- 3 variables → a plane.
- System solutions are common intersections of planes.
- A point lies on a plane when substitution satisfies the equation.
- Geometry helps interpret unique / infinite / none before computing determinants.

## The 3×3 determinant

### Intuition
The 3×3 determinant plays the same role as the 2×2 determinant, but in three dimensions. It checks whether three constraints or three vectors create nonzero volume, or whether they collapse into a plane/line.

If the determinant is nonzero, the three directions are independent and a square system has a unique solution. If the determinant is zero, there is linear dependence and the matrix is singular.

### Definitions and formulas
For:

$$
A=\begin{bmatrix}
a&b&c\\
d&e&f\\
g&h&i
\end{bmatrix}
$$

Sarrus' rule gives:

$$
\det(A)=aei+bfg+cdh-ceg-bdi-afh
$$

The three “down-right” products are added, and the three “down-left” products are subtracted.

### Worked example
Compute the determinant:

$$
A=\begin{bmatrix}
1&1&1\\
1&2&1\\
1&1&2
\end{bmatrix}
$$

The positive products are:

$$
1\cdot2\cdot2=4,\quad 1\cdot1\cdot1=1,\quad 1\cdot1\cdot1=1
$$

The negative products are:

$$
1\cdot2\cdot1=2,\quad 1\cdot1\cdot1=1,\quad 1\cdot1\cdot2=2
$$

Therefore:

$$
\det(A)=4+1+1-2-1-2=1
$$

Since the determinant is nonzero, the matrix is non-singular.

### Related NumPy snippet
```python
import numpy as np

A = np.array([[1, 1, 1],
              [1, 2, 1],
              [1, 1, 2]], dtype=float)
print(round(np.linalg.det(A)))  # 1
```

With floating-point arithmetic, NumPy may return $0.9999999999$ instead of exactly 1, so `round` or `np.isclose` can be useful.

> ⚠️ Common mistakes: Sarrus' rule applies directly only to 3×3 matrices. Also, the last three products must be subtracted; adding all six products gives the wrong answer.

### Summary
- The 3×3 determinant checks singularity for systems in three variables.
- Sarrus: three positive products, three negative products.
- Det 0 ⇔ linear dependence ⇔ singular.
- Nonzero det ⇔ non-singular ⇔ one solution for every $b$.
