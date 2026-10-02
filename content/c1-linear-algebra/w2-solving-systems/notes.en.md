# Week 2: Solving systems of linear equations

## ML motivation

### Intuition
Week 2 explains why solving systems is more than a hand-calculation trick. In a neural network, one layer applies many weighted sums at once; in sound recognition, a matrix may store features across time; in music generation, compression often keeps only the directions that carry useful variation. Row reduction is the small-scale version of asking which equations or features carry independent information.

A useful mental model is a mixing board. Each row of a matrix listens to the input vector in a different way. If two rows listen in essentially the same way, the model has redundancy. If the rows point in genuinely different directions, the output preserves more information and systems become easier to solve.

### Core facts
A linear system is written as $Ax=b$. The coefficient matrix $A$ stores the left-hand side, $x$ stores the unknowns, and $b$ stores the target values. Row operations let us replace the original system by an equivalent one whose solution set is easier to read.

For ML intuition, rank measures how many independent directions the matrix can express. A low-rank feature table can be compressed, but a low-rank square coefficient matrix cannot identify every unknown uniquely.

### Worked example
Suppose two sensors report

$$
\begin{cases}
2u+v=9\\
-u+3v=8
\end{cases}
$$

Add half of the first equation to the second: $-u+3v+(u+0.5v)=8+4.5$, so $3.5v=12.5$ and $v=25/7$. Then $2u+25/7=9$, hence $u=19/7$. The rows are not multiples, so the two sensor readings contain two independent pieces of information.

### NumPy check
```python
import numpy as np
A = np.array([[2, 1], [-1, 3]], dtype=float)
b = np.array([9, 8], dtype=float)
print(np.linalg.solve(A, b))
```

> ⚠️ Pitfall: a matrix can have many rows and still carry little information if most rows are linear combinations of earlier rows.

### Summary
- Systems connect course algebra to feature maps and compression.
- Rank is an information count, not just a shape description.
- Row reduction reveals independence by exposing pivots.
- Solving by hand builds the intuition behind numerical routines.

## Non-singular linear systems

### Intuition
A non-singular square system has exactly one solution. Geometrically in two variables, two non-parallel lines meet at one point. Algebraically, elimination creates a pivot for every unknown, so back-substitution determines each variable without choices.

The slides use shopping equations such as $a+b=10$ and $a+2b=12$. The second equation changes the banana coefficient while keeping the apple coefficient, so subtracting the first equation isolates the banana price.

### Core facts
For a square coefficient matrix $A$, non-singular means full rank. In a $2\times2$ case, this is equivalent to $\det(A)\ne0$. During elimination, each pivot must be non-zero; if the current entry is zero, swap with a lower row when possible.

### Worked example
Solve

$$
\begin{cases}
5a+b=17\\
4a-3b=6
\end{cases}
$$

Divide the first equation by $5$ to get $a+0.2b=3.4$. Divide the second by $4$ to get $a-0.75b=1.5$. Subtract the second normalized equation from the first: $0.95b=1.9$, so $b=2$. Substitute into $a+0.2b=3.4$: $a+0.4=3.4$, so $a=3$.

### NumPy check
```python
import numpy as np
A = np.array([[5, 1], [4, -3]], dtype=float)
b = np.array([17, 6], dtype=float)
print(np.linalg.det(A), np.linalg.solve(A, b))
```

> ⚠️ Pitfall: never divide by a pivot before checking that it is non-zero; row swapping is a valid operation, division by zero is not.

### Summary
- One pivot per variable gives one solution.
- Non-zero determinant in $2\times2$ confirms independence.
- Elimination plus back-substitution is the central manual method.
- Always verify the answer in the original equations.

## Singular linear systems

### Intuition
A singular system does not determine one point. It may be redundant, giving infinitely many solutions, or contradictory, giving no solution. The same coefficient pattern can lead to either case depending on the right-hand side.

This distinction matters in ML because repeated features are not automatically bad, but contradictory constraints cannot be satisfied. Row reduction separates these cases by showing either a zero row with zero on the right or a zero row with a non-zero right-hand side.

### Core facts
If elimination gives $0=0$, one equation brought no new information. If it gives $0=c$ with $c\ne0$, the system is inconsistent. Singular means the coefficient matrix lacks full rank; it does not by itself say whether $b$ is compatible.

### Worked example
Compare two systems:

$$
\begin{cases}
a+b=10\\
2a+2b=20
\end{cases}
\qquad
\begin{cases}
a+b=10\\
2a+2b=24
\end{cases}
$$

For the first, replace row 2 by row 2 minus $2$ row 1: $0=0$. Every pair with $a=10-b$ works. For the second, the same operation gives $0=4$, impossible. Both coefficient matrices are singular, but their solution sets differ.

### NumPy check
```python
import numpy as np
A = np.array([[1, 1], [2, 2]], dtype=float)
print(np.linalg.matrix_rank(A))
```

> ⚠️ Pitfall: do not say “singular means no solution.” Redundant consistent systems are singular and still have infinitely many solutions.

### Summary
- Redundancy produces free variables.
- Contradiction appears as $0=c$ with $c\ne0$.
- The coefficient matrix alone cannot classify consistency.
- The augmented matrix must be reduced with the right-hand side.

## Systems with more variables

### Intuition
With three or more unknowns, elimination is the same idea repeated. First use a pivot to remove the first variable from lower equations. Then use a second pivot to remove the second variable from equations below. When the system becomes triangular, solve from the bottom upward.

A triangular system is like a locked chain: the last equation unlocks the last variable, that value unlocks the previous row, and so on.

### Core facts
For $n$ unknowns, a unique solution requires a pivot in every variable column. If fewer pivots appear and the system is consistent, at least one variable is free. If an impossible row appears, there is no solution.

### Worked example
Consider

$$
\begin{cases}
a+b+2c=12\\
3a-3b-c=3\\
2a-b+6c=24
\end{cases}
$$

After normalizing the first column as in the slides, the system becomes

$$
\begin{cases}
a+b+2c=12\\
-2b-\frac73c=-11\\
-\frac32b+c=0
\end{cases}
$$

Normalize the last two rows: $b+\frac76c=\frac{11}{2}$ and $b-\frac23c=0$. Subtract to get $\frac{11}{6}c=\frac{11}{2}$, so $c=3$. Then $b=2$, and the first row gives $a=4$.

### NumPy check
```python
import numpy as np
A = np.array([[1,1,2],[3,-3,-1],[2,-1,6]], dtype=float)
b = np.array([12,3,24], dtype=float)
print(np.linalg.solve(A, b))
```

> ⚠️ Pitfall: solving top-down after reaching triangular form usually reintroduces unknowns; back-substitution starts from the lowest pivot row.

### Summary
- Forward elimination creates an upper-triangular shape.
- Back-substitution reads variables from bottom to top.
- Missing pivots become free variables in consistent systems.
- A single inconsistent row ends the search for solutions.

## Matrix row reduction

### Intuition
Writing the augmented matrix $[A\mid b]$ removes repeated variable symbols and focuses attention on rows. Each row still represents an equation, including the right-hand side. Row reduction is therefore equation manipulation in a compact table.

The slides move from equations such as $5a+b=17$ to matrices like $[5\;1\mid17]$. This notation becomes essential when systems are large.

### Core facts
The valid row operations are: swap two rows; multiply a row by a non-zero scalar; add a multiple of one row to another row. These operations preserve the solution set of the augmented system. The right-hand column must move with the same operations as the coefficient columns.

### Worked example
Start with

$$
\left[\begin{array}{cc|c}
1&2&5\\
2&5&12
\end{array}\right].
$$

Use $R_2\leftarrow R_2-2R_1$:

$$
\left[\begin{array}{cc|c}
1&2&5\\
0&1&2
\end{array}\right].
$$

The second row says $y=2$. The first row says $x+4=5$, so $x=1$. Reducing only the first two columns while leaving $12$ unchanged would describe a different problem.

### NumPy check
```python
import numpy as np
M = np.array([[1,2,5],[2,5,12]], dtype=float)
M[1] = M[1] - 2*M[0]
print(M)
```

> ⚠️ Pitfall: the vertical bar in $[A\mid b]$ is visual only; row operations cross it because they transform entire equations.

### Summary
- Augmented matrices encode systems compactly.
- Row operations must be applied entry by entry across the full row.
- The right-hand side is not optional bookkeeping.
- Echelon form makes the solution structure visible.

## Row operations and singularity

### Intuition
Row operations can change the appearance of a matrix without changing whether it is singular. The determinant may change sign or scale, but zero stays zero and non-zero stays non-zero when the operation is valid.

This is why elimination is safe: we may simplify a coefficient matrix while preserving the essential question “does it have full independent information?”

### Core facts
Swapping two rows flips the determinant sign. Multiplying a row by $k\ne0$ multiplies the determinant by $k$. Adding a multiple of one row to another leaves the determinant unchanged. None of these operations turns a zero determinant into a non-zero determinant or the reverse.

### Worked example
Let

$$
A=\begin{bmatrix}5&1\\4&3\end{bmatrix},
\quad \det(A)=5\cdot3-1\cdot4=11.
$$

Swapping rows gives determinant $-11$. Multiplying the first row by $2$ gives determinant $22$. Replacing row 2 by row 2 minus row 1 gives

$$
\begin{bmatrix}5&1\\-1&2\end{bmatrix},
$$

whose determinant is $10+1=11$. In every case, singularity status is preserved.

### NumPy check
```python
import numpy as np
A = np.array([[5,1],[4,3]], dtype=float)
B = A.copy(); B[1] = B[1] - B[0]
print(np.linalg.det(A), np.linalg.det(B))
```

> ⚠️ Pitfall: multiplying a row by zero is not a valid row operation because it destroys information and can change singularity.

### Summary
- Valid row operations preserve zero-versus-non-zero determinant status.
- Row replacement is especially useful because it leaves determinant unchanged.
- Scaling by a non-zero number is reversible.
- Elimination relies on these reversible information-preserving moves.

## Rank of a matrix

### Intuition
Rank counts independent information. In row reduction, each pivot marks one row direction that was not explained by previous rows. A zero row after reduction means that row was redundant.

The slides connect this to sentence systems: two identical statements may look like two rows, but they contain only one piece of information. Rank formalizes that count for matrices.

### Core facts
Rank equals the number of pivots in row echelon form. For a square $n\times n$ matrix, rank $n$ means non-singular; rank below $n$ means singular. In homogeneous systems $Ax=0$, the dimension of the solution space is $n-\operatorname{rank}(A)$.

### Worked example
For

$$
A=\begin{bmatrix}1&1\\2&2\end{bmatrix},
$$

replace row 2 by row 2 minus $2$ row 1:

$$
\begin{bmatrix}1&1\\0&0\end{bmatrix}.
$$

There is one pivot, so rank is $1$. The homogeneous system $a+b=0$ has one free variable; for example $b=t$ and $a=-t$.

### NumPy check
```python
import numpy as np
A = np.array([[1,1],[2,2]], dtype=float)
print(np.linalg.matrix_rank(A))
```

> ⚠️ Pitfall: rank is not the number of non-zero entries. A dense row can still be dependent on earlier rows.

### Summary
- Count pivots, not entries.
- Zero rows after elimination do not contribute rank.
- Full rank square matrices are non-singular.
- Rank also predicts degrees of freedom in $Ax=0$.

## Rank in the general case

### Intuition
Rectangular matrices also have rank. The maximum possible rank is limited by the smaller dimension: an $m\times n$ matrix can have at most $\min(m,n)$ pivots. Extra rows or columns may help only if they add independent directions.

In data tables, this explains why a table with many measured features can still have low intrinsic dimension when columns are combinations of a few hidden factors.

### Core facts
For any matrix, row reduction exposes pivot columns and zero rows. The rank is unique even though the sequence of row operations is not unique. If a matrix has $n$ columns and rank $r$, then a homogeneous system has $n-r$ free variables.

### Worked example
Reduce

$$
B=\begin{bmatrix}
1&1&1\\
1&1&2\\
1&1&3
\end{bmatrix}.
$$

Subtract row 1 from rows 2 and 3:

$$
\begin{bmatrix}
1&1&1\\
0&0&1\\
0&0&2
\end{bmatrix}.
$$

Then replace row 3 by row 3 minus $2$ row 2 to get a zero row. There are two pivots, in columns 1 and 3, so rank is $2$.

### NumPy check
```python
import numpy as np
B = np.array([[1,1,1],[1,1,2],[1,1,3]], dtype=float)
print(np.linalg.matrix_rank(B))
```

> ⚠️ Pitfall: a skipped column is not a mistake. It means that variable column did not become a pivot column in this reduction.

### Summary
- Rank never exceeds $\min(m,n)$.
- Pivot count is independent of the particular valid reduction path.
- Non-pivot columns correspond to possible free variables.
- Rectangular rank is central for compression and least-squares intuition.

## Row echelon form

### Intuition
Row echelon form (REF) is a staircase. Zero rows are placed at the bottom, each pivot sits to the right of the pivot in the row above, and entries below pivots are zero. The matrix is not fully solved yet, but it is organized enough for rank and back-substitution.

The course slides allow a pivot in REF to be any non-zero number. It does not have to be $1$ until we ask for reduced row echelon form.

### Core facts
A matrix is in REF when three conditions hold: all non-zero rows are above zero rows; leading non-zero entries move right as you go down; entries below each leading entry are zero. REF is generally not unique.

### Worked example
Start with

$$
\begin{bmatrix}2&4&6\\1&3&5\\0&2&4\end{bmatrix}.
$$

Swap rows 1 and 2 to get a simpler first pivot. Replace the new row 2 by row 2 minus $2$ row 1:

$$
\begin{bmatrix}1&3&5\\0&-2&-4\\0&2&4\end{bmatrix}.
$$

Then replace row 3 by row 3 plus row 2:

$$
\begin{bmatrix}1&3&5\\0&-2&-4\\0&0&0\end{bmatrix}.
$$

This is REF with two pivots.

### NumPy check
```python
import numpy as np
M = np.array([[1,3,5],[0,-2,-4],[0,0,0]], dtype=float)
print(np.count_nonzero(np.any(M != 0, axis=1)))
```

> ⚠️ Pitfall: do not reject REF merely because a pivot is $-2$; non-zero is enough at this stage.

### Summary
- REF is a staircase form for reading pivots.
- Pivots move strictly right as rows go down.
- Entries below each pivot are zero.
- REF supports rank counting and back-substitution.

## General row echelon form

### Intuition
In larger or rectangular matrices, the staircase may skip columns. A skipped column simply means no row has its first non-zero entry there. The important invariant is the pivot pattern, not whether every column participates.

This topic extends the clean $2\times2$ picture to real ML matrices, where rows and columns need not match.

### Core facts
General REF works for any shape. Zero rows remain at the bottom, leading entries move right, and each pivot clears entries below it. The number of pivots is rank. Columns without pivots are free-variable columns when the matrix is used in a system.

### Worked example
Consider

$$
C=\begin{bmatrix}
0&2&4&1\\
0&0&3&6\\
0&0&0&0
\end{bmatrix}.
$$

This is already in REF. The first pivot is in column 2, the second in column 3, and column 1 has no pivot. Rank is $2$. If this were a homogeneous system with four variables, there would be $4-2=2$ free variables.

### NumPy check
```python
import numpy as np
C = np.array([[0,2,4,1],[0,0,3,6],[0,0,0,0]], dtype=float)
print(np.linalg.matrix_rank(C))
```

> ⚠️ Pitfall: a leading zero column does not prevent REF; it only delays the first pivot to a later column.

### Summary
- REF applies to non-square matrices.
- Pivot columns may be separated by skipped columns.
- Rank is still the pivot count.
- Free variables appear in non-pivot columns of a consistent system.

## Reduced row echelon form

### Intuition
Reduced row echelon form (RREF) is the cleaned-up endpoint. Every pivot equals $1$, and each pivot column has zeros both below and above the pivot. In a solved square non-singular system, the coefficient side becomes the identity matrix.

RREF is valuable because it displays the solution directly. It is stricter than REF and, for a given matrix, unique.

### Core facts
RREF requires all REF conditions, plus unit pivots and clean pivot columns. For augmented matrices, an RREF row such as $[0\;0\mid5]$ still signals inconsistency. A row like $[0\;0\mid0]$ contributes no pivot.

### Worked example
Start from the augmented REF

$$
\left[\begin{array}{cc|c}
1&2&7\\
0&3&6
\end{array}\right].
$$

Scale row 2 by $1/3$:

$$
\left[\begin{array}{cc|c}
1&2&7\\
0&1&2
\end{array}\right].
$$

Clear above the second pivot with $R_1\leftarrow R_1-2R_2$:

$$
\left[\begin{array}{cc|c}
1&0&3\\
0&1&2
\end{array}\right].
$$

The RREF gives $x=3$, $y=2$ immediately.

### NumPy check
```python
import numpy as np
M = np.array([[1,2,7],[0,3,6]], dtype=float)
M[1] = M[1] / 3
M[0] = M[0] - 2*M[1]
print(M)
```

> ⚠️ Pitfall: stopping at upper-triangular REF is fine for back-substitution, but it is not yet RREF unless pivot columns are cleaned above.

### Summary
- RREF adds unit pivots and zeros above pivots.
- It reveals solutions with minimal back-substitution.
- Unlike REF, RREF is unique for a fixed matrix.
- Inconsistent augmented rows remain impossible in any form.

