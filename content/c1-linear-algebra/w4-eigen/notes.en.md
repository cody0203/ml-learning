## ML motivation: PCA

### Intuition
PCA starts with a practical ML problem.
A data set may have many columns.
Some columns move together.
Then the cloud of points is thin.
It may sit near a line or a plane.
PCA finds those important directions.

In the slides, the story moves from 2D to 1D.
Later it moves from 8D to 3D.
The goal is not to delete columns blindly.
The goal is to keep directions of large spread.

### Definitions and formulas
Center the data matrix first.
The covariance matrix is symmetric.
Its eigenvectors give principal directions.
Its eigenvalues measure variance in those directions.

If the eigenvalues are 9 and 1, PC1 keeps
$9/(9+1)=0.9$ of the variance.
For a centered data matrix $X$, projection onto a unit
direction $u$ is simply $Xu$.

### Worked example
Suppose a centered 2D data set has covariance
$C=[[4,0],[0,1]]$.
The first coordinate varies four times as much.
The eigenvectors are $(1,0)$ and $(0,1)$.
Their eigenvalues are 4 and 1.
Keeping one dimension means keeping the x-axis.
The retained variance ratio is $4/5=0.8$.

### NumPy snippet
```python
import numpy as np
Xc = X - X.mean(axis=0)
C = np.cov(Xc, rowvar=False)
w, V = np.linalg.eigh(C)
pc1 = V[:, np.argmax(w)]
z = Xc @ pc1
```

> ⚠️ Common mistake: using PCA before centering.

### Summary
- PCA chooses directions, not original column names.
- Large eigenvalue means large variance.
- Projection turns many features into fewer scores.

## Singularity and rank of linear transformations

### Intuition
A 2D matrix transforms the unit square.
If the square becomes a parallelogram with area, both
independent directions survived.
If it collapses to a line, one direction was lost.
If it collapses to the origin, both directions were lost.

This is why rank is geometric.
Rank 2 fills a plane.
Rank 1 fills a line.
Rank 0 fills only one point.

### Definitions and formulas
The rank of a transformation is the dimension of its image.
A square matrix is non-singular when it has full rank.
For a 2 by 2 matrix, full rank is equivalent to nonzero
determinant.

### Worked example
Let $A=[[1,1],[2,2]]$.
Both columns point in the same direction.
The image of $(a,b)$ is $(a+b,2a+2b)$.
Every output is a multiple of $(1,2)$.
So the whole plane is squeezed onto one line.
The determinant is $1\cdot2-1\cdot2=0$.
Thus $A$ is singular and rank 1.

### NumPy snippet
```python
A = np.array([[1, 1], [2, 2]])
np.linalg.matrix_rank(A)
```

> ⚠️ Common mistake: saying singular means no output.
> A singular map still has outputs; it has too few directions.

### Summary
- Rank counts independent output directions.
- Singular maps lose at least one input direction.
- Nonzero determinant in 2D means rank 2.

## Determinant as oriented area

### Intuition
The determinant measures how area changes.
The sign also records orientation.
A negative determinant means the ordered pair of basis
vectors has been flipped.

The slides show columns $(3,1)$ and $(1,2)$.
They form a parallelogram of area 5.
The reversed-looking example has determinant -5.
Its geometric area is still 5.

### Definitions and formulas
For $A=[[a,b],[c,d]]$,
the determinant is $ad-bc$.
The area scale is $|\det(A)|$.
The sign tells whether orientation is preserved.

### Worked example
For $A=[[1,3],[2,1]]$,
$\det(A)=1\cdot1-3\cdot2$.
That equals $-5$.
The unit square becomes a parallelogram of area 5.
Because the sign is negative, the orientation is reversed.

### NumPy snippet
```python
A = np.array([[1, 3], [2, 1]], dtype=float)
detA = np.linalg.det(A)
area = abs(detA)
```

> ⚠️ Common mistake: reporting negative area.
> The determinant is signed; area is not.

### Summary
- Determinant is signed area scale.
- Zero area means collapse.
- Absolute value gives geometric area.

## Determinant of a product

### Intuition
Matrix multiplication composes transformations.
If $B$ first triples area and $A$ then scales area by 5,
the combined map scales area by 15.

This explains the slide formula.
The determinant of a product is the product of determinants.

### Definitions and formulas
$\det(AB)=\det(A)\det(B)$.
The order of matrix multiplication matters.
The determinant scale still multiplies as a scalar.

### Worked example
Let $\det(A)=5$ and $\det(B)=8$.
Then $\det(AB)=40$.
If $B$ is singular, $\det(B)=0$.
Then $\det(AB)=\det(A)\cdot0=0$.
So a non-singular matrix times a singular one is singular.

### NumPy snippet
```python
left = np.linalg.det(A @ B)
right = np.linalg.det(A) * np.linalg.det(B)
np.allclose(left, right)
```

> ⚠️ Common mistake: applying this rule to sums.
> In general, det(A+B) is not det(A)+det(B).

### Summary
- Product determinants multiply.
- Any zero factor makes the product singular.
- The rule reflects repeated area scaling.

## Determinant of the inverse

### Intuition
An inverse undoes a transformation.
If a matrix multiplies area by 5,
its inverse must divide area by 5.
That is why inverse determinants are reciprocals.

### Definitions and formulas
If $A$ is invertible, then $AA^{-1}=I$.
Since $\det(I)=1$,
$\det(A)\det(A^{-1})=1$.
Therefore $\det(A^{-1})=1/\det(A)$.

### Worked example
The slides use a matrix with determinant 5.
Its inverse has determinant 0.2.
Indeed $5\cdot0.2=1$.
For determinant 8, the inverse determinant is 0.125.
For determinant 0, there is no inverse.

### NumPy snippet
```python
A = np.array([[3, 1], [1, 2]], dtype=float)
np.linalg.det(np.linalg.inv(A))
```

> ⚠️ Common mistake: trying to invert a singular matrix.

### Summary
- Inverse area scaling is reciprocal.
- The identity matrix has determinant 1.
- Singular matrices cannot be inverted.

## Bases

### Intuition
A basis is a coordinate system.
In 2D, two non-parallel vectors are enough.
They let every point be described uniquely.
If the two vectors are parallel, they describe only a line.

### Definitions and formulas
A basis must span the space.
It must also be linearly independent.
In a 2D column matrix, determinant nonzero checks both
conditions at once.

### Worked example
The vectors $(3,1)$ and $(1,2)$ form a basis of the plane.
Place them as columns: $A=[[3,1],[1,2]]$.
The determinant is 5.
Since it is nonzero, every 2D vector has unique coordinates
in this basis.

The vectors $(1,0)$ and $(2,0)$ do not form a 2D basis.
They only move along the x-axis.

### NumPy snippet
```python
B = np.array([[3, 1], [1, 2]])
is_basis = abs(np.linalg.det(B)) > 1e-9
```

> ⚠️ Common mistake: counting vectors only.
> Two vectors in 2D can still fail if they are parallel.

### Summary
- Basis means spanning plus independence.
- Basis size equals dimension.
- Nonzero 2D determinant confirms a 2D basis.

## Span

### Intuition
Span is the set of all possible linear combinations.
One nonzero vector spans a line through the origin.
Two non-parallel vectors in the plane span all of 2D.
Adding another vector on the same line adds nothing new.

### Definitions and formulas
The span of $v_1$ and $v_2$ is all
$c_1v_1+c_2v_2$ for real scalars.
The dimension of a row or column span equals rank.

### Worked example
The vector $(3,6)$ lies in the span of $(1,2)$.
It equals $3(1,2)$.
The vector $(2,3)$ does not.
No scalar $c$ satisfies both $c=2$ and $2c=3$.

For rows $(1,1)$ and $(2,2)$, the row span is one line.
The rank is 1.

### NumPy snippet
```python
v = np.array([1, 2])
w = np.array([3, 6])
parallel = abs(v[0]*w[1] - v[1]*w[0]) < 1e-9
```

> ⚠️ Common mistake: thinking span can be shifted.
> Linear span always passes through the origin.

### Summary
- Span contains all linear combinations.
- Dependent vectors do not enlarge span.
- Rank is the dimension of the span.

## Eigenbasis

### Intuition
An eigenbasis chooses axes that the transformation does not mix.
Along each eigen-axis, the matrix only stretches or shrinks.
This makes repeated transformations much easier to understand.

### Definitions and formulas
If a matrix has enough independent eigenvectors,
write $A=PDP^{-1}$.
The columns of $P$ are eigenvectors.
The diagonal entries of $D$ are eigenvalues.

### Worked example
For $A=[[2,1],[0,3]]$,
$(1,0)$ is an eigenvector with eigenvalue 2.
Also, $(1,1)$ is an eigenvector with eigenvalue 3.
They are not parallel, so they form a basis.
In that basis, the same transformation is diagonal:
one coordinate is doubled and the other is tripled.

### NumPy snippet
```python
w, P = np.linalg.eig(A)
D = np.diag(w)
np.allclose(A, P @ D @ np.linalg.inv(P))
```

> ⚠️ Common mistake: assuming every matrix has a full
> eigenbasis over the real numbers.

### Summary
- Eigenbasis uses independent eigenvectors.
- Diagonal form separates the directions.
- It is useful for powers and dynamics.

## Eigenvalues and eigenvectors

### Intuition
Most vectors change direction under a matrix.
Eigenvectors are special.
They stay on the same line.
Only their length and sign may change.

### Definitions and formulas
An eigenvector is nonzero and satisfies $Av=\lambda v$.
The scalar $\lambda$ is the eigenvalue.
Find eigenvalues from $\det(A-\lambda I)=0$.
Then solve $(A-\lambda I)v=0$.

### Worked example
The quiz matrix is $[[9,4],[4,3]]$.
Its trace is 12 and determinant is 11.
So the characteristic polynomial is
$\lambda^2-12\lambda+11$.
It factors as $(\lambda-11)(\lambda-1)$.
The eigenvalues are 11 and 1.
Matching eigenvectors are multiples of $(2,1)$ and $(-1,2)$.

### NumPy snippet
```python
A = np.array([[9, 4], [4, 3]], dtype=float)
w, V = np.linalg.eig(A)
```

> ⚠️ Common mistake: accepting the zero vector.
> It satisfies the equation for every lambda, so it is excluded.

### Summary
- Eigenvectors preserve direction.
- Eigenvalues are scale factors.
- The characteristic equation finds candidates.

## Characteristic polynomial and PCA

### Intuition
The characteristic polynomial is the bridge from a matrix
to its eigenvalues.
PCA uses that bridge on a covariance matrix.
Large roots correspond to directions with large variance.

### Definitions and formulas
For a 2 by 2 matrix, the polynomial can be written as
$\lambda^2-\operatorname{tr}(A)\lambda+\det(A)$.
This is equivalent to $\det(A-\lambda I)$.
For covariance matrices, use symmetric eigensolvers.

### Worked example
For $A=[[2,1],[0,3]]$,
the trace is 5 and determinant is 6.
The polynomial is $\lambda^2-5\lambda+6$.
It factors as $(\lambda-2)(\lambda-3)$.
So the roots are 2 and 3.

If a covariance matrix has roots 7, 2, and 1,
the first two PCs keep $(7+2)/10=0.9$ of total variance.

### NumPy snippet
```python
C = np.array([[4, 0], [0, 1]], dtype=float)
w, V = np.linalg.eigh(C)
order = np.argsort(w)[::-1]
```

> ⚠️ Common mistake: sorting PCA components by vector length.
> Sort by eigenvalue instead.

### Summary
- Roots of the characteristic polynomial are eigenvalues.
- Trace and determinant give the 2D polynomial quickly.
- PCA ranks covariance eigenvectors by eigenvalue.
