# Signal*Per$\!\!\;f\!$ect*

The underlying signal is modeled by a quadratic spline. A spline is a piecewise polynomial. Each piece or segment of the spline is a polynomial and is separated by $x$-values called knots. A spline is subject to constraints such at the knots, the $y$-values of adjacent polynomial pieces are equal such that the pieces meet, and the derivatives of adjacent polynomial pieces are equal such that the spine is smooth and without kinks. The quadratic spline by which the underlying signal is modeled has an extra constraint such that its average value over each interval in the time series is equal to the value of that time series for that interval.

For all $i$ in $\{1,\dots,n\}$:

$$ f_i(t) = a_i t^2 + b_i t + c_i $$

We have $3n$ unknowns, and thus need $3n$ equations.

## "Zero-derivative boundary conditions" variant

### Knot constraint

The $y$-values of adjacent quadratic pieces must be equal to each other at each knot such that the pieces meet. Therefore, for all $i$ in $\{1,\dots,n-1\}$:

$$
\begin{aligned}
& f_i(x_i)=f_{i+1}(x_i) \\
& \implies a_i x_i^2 + b_i x_i + c_i = a_{i+1} x_i^2 + b_{i+1} x_i + c_{i+1} \\
& \implies (-x_i^2) a_i + (-x_i) b_i + (-1) c_i + (x_i^2) a_{i+1} + (x_i) b_{i+1} + (1) c_{i+1} = 0
\end{aligned}
$$

This represents a set of $n-1$ linear equations, which can be expressed in the form $A_1\mathbf{x}=\mathbf{b}_1$, where:

$$ \mathbf{x} = \begin{bmatrix} a_1 & b_1 & c_1 & \cdots & a_n & b_n & c_n \end{bmatrix}^\mathrm{T} $$

$$ \mathbf{b}_1 = \begin{bmatrix} 0 & \dots & 0 \end{bmatrix}^\mathrm{T} $$

```python
# Written in Python:
b1 = np.zeros((n-1, 1))
```

$$
A_1 =
\begin{bmatrix}
    -x_1^2 & -x_1 & -1 & x_1^2 & x_1 & 1 & 0 & 0 & 0 & \cdots & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & -k4^2 & -x_2 & -1 & x_2^2 & x_2 & 1 & \cdots & 0 & 0 & 0 & 0 & 0 & 0 \\
    \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \dots & -x_n^2 & -x_n & -1 & x_n^2 & x_n & 1
\end{bmatrix}
$$

```python
A1 = np.array([[0, 0, 0] * (i-1) + [-x[i]**2, -x[i], -1, x[i]**2, x[i], 1] + [0, 0, 0] * (n-i-1) for i in range(1, n)])
```

When using `numpy.PPoly`, however, each polynomial piece, regardless of what range on the $x$-axis it spans in the piecewise function,  is expressed in terms of $x$ starting at zero. This is a way of "normalizing" each polynomial and avoiding sensitive coefficients. Thus, the $x_i$ for an $(a_i,b_i,c_i)$ triple is replaced with zero and the $x_i$ for an $(a_{i+1},b_{i+1},c_{i+1})$ triple is replaced with its distance from zero, $x_i-x_{i-1}$, as follows:

$$
A_1 =
\begin{bmatrix}
    -(x_1 - x_0)^2 & -(x_1 - x_0) & -1 & 0 & 0 & 1 & 0 & 0 & 0 & \cdots & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & -(x_2 - x_1)^2 & -(x_2 - x_1) & -1 & 0 & 0 & 1 & \cdots & 0 & 0 & 0 & 0 & 0 & 0 \\
    \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \dots & -(x_n - x_{n-1})^2 & -(x_n - x_{n-1}) & -1 & 0 & 0 & 1
\end{bmatrix}
$$

```python
A1 = np.array([[0, 0, 0] * (i-1) + [-(x[i] - x[i-1])**2, -(x[i] - x[i-1]), -1, 0, 0, 1] + [0, 0, 0] * (n-i-1) for i in range(1, n)])
```

We need $2n+1$ more equations.

### Knot derivative constraint

The derivatives of adjacent quadratic pieces must be equal to each other at each knot such that the spine is smooth and without kinks. The derivative of quadratic piece $i$ is:

$$ f_i'(t) = 2 a_i t + b_i $$

Therefore, for all $i$ in $\{1,\dots,n-1\}$:

$$
\begin{aligned}
& f_i'(x_i)=f_{i+1}'(x_i) \\
& \implies 2 a_i x_i + b_i = 2 a_{i+1} x_i + b_{i+1} \\
& \implies (-2 x_i) a_i + (-1) b_i + (2 x_i) a_{i+1} + (1) b_{i+1} = 0
\end{aligned}
$$

This represents another $n-1$ linear equations, which can be expressed in the form $A_2\mathbf{x}=\mathbf{b}_2$, where $\mathbf{x}$ is the same as before and:

$$ \mathbf{b}_2 = \begin{bmatrix} 0 & \dots & 0 \end{bmatrix}^\mathrm{T} $$

```python
b2 = np.zeros((n-1, 1))
```

$$
A_2 =
\begin{bmatrix}
    -2 x_1 & -1 & 0 & 2 x_1 & 1 & 0 & 0 & 0 & 0 & \dots & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & -2 x_2 & -1 & 0 & 2 x_2 & 1 & 0 & \dots & 0 & 0 & 0 & 0 & 0 & 0 \\
    \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \dots & -2 x_n & -1 & 0 & 2 x_n & 1 & 0
\end{bmatrix}
$$

```python
A2 = np.array([[0, 0, 0] * (i-1) + [-2 * x[i], -1, 0, 2 * x[i], 1, 0] + [0, 0, 0] * (n-i-1)  for i in range(1, n)])
```

Again, replacing the $x_i$ for each $(a_i,b_i,c_i)$ triple with zero, and replacing the $x_i$ for each $(a_{i+1},b_{i+1},c_{i+1})$ triple with $x_i-x_{i-1}$, yields:

$$
A_2 =
\begin{bmatrix}
    -2 (x_1 - x_0) & -1 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & \dots & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & -2 (x_2 - x_1) & -1 & 0 & 0 & 1 & 0 & \dots & 0 & 0 & 0 & 0 & 0 & 0 \\
    \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \dots & -2 (x_n - x_{n-1}) & -1 & 0 & 0 & 1 & 0
\end{bmatrix}
$$

```python
A2 = np.array([[0, 0, 0] * (i-1) + [-2 * (x[i] - x[i-1]), -1, 0, 0, 1, 0] + [0, 0, 0] * (n-i-1)  for i in range(1, n)])
```

Now we need $n+2$ more equations.

### Interval average constraint

The average value of the quadratic spline over each interval in the time series $y[t]$ must be equal to the value of that time series for that interval. The spline's average over an interval is expressed in terms of the definite integral of each quadratic piece divided by the width of the interval. The indefinite integral is:

$$ F_i(t) = \int f_i(t) \ \mathrm{d}t = \frac{a_i}{3} t^3 + \frac{b_i}{2} t^2 + c_i t + C $$

In the "zero-derivative boundary conditions" variant, the knots separate the intervals in the time series. Therefore, for all $i$ in $\{1,\dots,n\}$:

$$
\begin{aligned}
& \text{average over interval } i = \frac{1}{x_i - x_{i-1}} \int_{x_{i-1}}^{x_i} f_i(t) \ \mathrm{d}t = \frac{1}{x_i - x_{i-1}} (F_i(x_i) - F_i(x_{i-1})) = y[i] \\
& \implies F_i(x_i) - F_i(x_{i-1}) = (x_i - x_{i-1}) \ y[i] \\
& \implies \left( \frac{a_i}{3} x_i^3 + \frac{b_i}{2} x_i^2 + c_i x_i + C \right) - \left( \frac{a_i}{3} x_{i-1}^3 + \frac{b_i}{2} x_{i-1}^2 + c_i x_{i-1} + C \right) = (x_i - x_{i-1}) \ y[i] \\
& \implies 2 a_i x_i^3 + 3 b_i x_i^2 + 6 c_i x_i - 2 a_i x_{i-1}^3 - 3 b_i x_{i-1}^2 - 6 c_i x_{i-1} = 6 (x_i - x_{i-1}) \ y[i] \\
& \implies 2 (x_i^3 - x_{i-1}^3) a_i + 3 (x_i^2 - x_{i-1}^2) b_i + 6 (x_i - x_{i-1}) c_i = 6 (x_i - x_{i-1}) \ y[i]
\end{aligned}
$$

This represents another $n$ linear equations, which can be expressed in the form $A_3\mathbf{x}=\mathbf{b}_3$, where $\mathbf{x}$ is the same as before and:

$$
\mathbf{b}_3 =
\begin{bmatrix}
    6 (x_1 - x_0) \ y[1] \\
    6 (x_2 - x_1) \ y[2] \\
    \dots \\
    6 (x_n - x_{n-1}) \ y[n]
\end{bmatrix}
$$

```python
b3 = np.array([[6 * y[i] * (x[i] - x[i-1])] for i in range(1, n+1)])
```

$$
A_3 =
\begin{bmatrix}
    2 (x_1^3 - x_0^3) & 3 (x_1^2 - x_0^2) & 6 (x_1 - x_0) & 0 & 0 & 0 & \dots & 0 & 0 & 0 \\
    0 & 0 & 0 & 2 (x_2^3 - x_1^3) & 3 (x_2^2 - x_1^2) & 6 (x_2 - x_1) & \dots & 0 & 0 & 0 \\
    \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots \\
    0 & 0 & 0 & 0 & 0 & 0 & \dots & 2 (x_n^3 - x_{n-1}^3) & 3 (x_n^2 - x_{n-1}^2) & 6 (x_n - x_{n-1})
\end{bmatrix}
$$

```python
A3 = np.array([[0, 0, 0] * (i-1) + [2 * (x[i]**3 - x[i-1]**3), 3 * (x[i]**2 - x[i-1]**2), 6 * (x[i] - x[i-1])] + [0, 0, 0] * (n-i) for i in range(1, n+1)])
```

Replacing the $x_i$ and $x_{i+1}$ for each $(a_i,b_i,c_i)$ triple with zero and $x_i-x_{i-1}$ respectively yields:

$$
A_3 =
\begin{bmatrix}
    2 (x_1 - x_0)^3 & 3 (x_1 - x_0)^2 & 6 (x_1 - x_0) & 0 & 0 & 0 & \dots & 0 & 0 & 0 \\
    0 & 0 & 0 & 2 (x_2 - x_1)^3 & 3 (x_2 - x_1)^2 & 6 (x_2 - x_1) & \dots & 0 & 0 & 0 \\
    \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots \\
    0 & 0 & 0 & 0 & 0 & 0 & \dots & 2 (x_n - x_{n-1})^3 & 3 (x_n - x_{n-1})^2 & 6 (x_n - x_{n-1})
\end{bmatrix}
$$

```python
A3 = np.array([[0, 0, 0] * (i-1) + [2 * (x[i] - x[i-1])**3, 3 * (x[i] - x[i-1])**2, 6 * (x[i] - x[i-1])] + [0, 0, 0] * (n-i) for i in range(1, n+1)])
```

(While it includes $x_i$ terms, the same substitution in $\mathbf{b}_3$ results in no change.)

Now we need $2$ more equations, from two boundary conditions.

### Boundary conditions

In the "zero-derivative boundary conditions" variant, the slope of the spline's endpoints are prescribed to be zero. Therefore,

$$ f_1'(x_0) = 0 \implies 2 a_1 x_0 + b_1 = 0 $$

$$ f_n'(x_n) = 0 \implies 2 a_n x_n + b_n = 0 $$

This represents another $2$ linear equations, which can be expressed in the form $A_4\mathbf{x}=\mathbf{b}_4$, where $\mathbf{x}$ is the same as before and:

$$ \mathbf{b}_4 = \begin{bmatrix} 0 & 0 \end{bmatrix}^\mathrm{T} $$

```python
b4 = np.array([[0], [0]])
```

$$
A_4 =
\begin{bmatrix}
    2 x_0 & 1 & 0 & \cdots & 0 & 0 & 0 \\
    0 & 0 & 0 & \cdots & 2 x_n & 1 & 0
\end{bmatrix}
$$

```python
A4 = np.array([[2 * x[0], 1, 0] + [0, 0, 0] * (n-1), [0, 0, 0] * (n-1) + [2 * x[n], 1, 0]])
```

Replacing $x_0$ and $x_n$ with zero and $x_n-x_{n-1}$ respectively yields:

$$
A_4 =
\begin{bmatrix}
    0 & 1 & 0 & \cdots & 0 & 0 & 0 \\
    0 & 0 & 0 & \cdots & 2 (x_n - x_{n-1}) & 1 & 0
\end{bmatrix}
$$

```python
A4 = np.array([[0, 1, 0] + [0, 0, 0] * (n-1), [0, 0, 0] * (n-1) + [2 * (x[n] - x[n-1]), 1, 0]])
```

## Zero-second-derivative boundary conditions

The knot and knot derivative constraints are the same as before, except there are $n$ equations for each (one more than before) and any knot $x_i$ is replaced by the following:

$$ \frac{x_{i+1} + x_i}{2} $$

In which $x_i$ will no longer represent a knot (which is now at the midpoint of each interval) but will still divide intervals. A shorthand $x_{i+0.5}$ will be used for the above expression.

For example, $x_1$ will be replaced by $\frac{x_2 + x_1}{2}$ for which the shorthand $x_{1.5}$ will be used.

Knot constraint:

```python
A1 = np.array([[0, 0, 0] * (i-1) + [-(x[i+1] + x[i])**2, -2 * (x[i+1] + x[i]), -4, (x[i+1] + x[i])**2, 2 * (x[i+1] + x[i]), 4] + [0, 0, 0] * (n-i-1) for i in range(0, n)])
```

```python
A1 = np.array([[0, 0, 0] * (i-1) + [-(x[i+1] - x[i])**2, -2 * (x[i+1] - x[i]), -4, 0, 0, 4] + [0, 0, 0] * (n-i-1) for i in range(0, n)])
```

Knot derivative constraint:

```python
A2 = np.array([[0, 0, 0] * (i-1) + [-(x[i+1] + x[i]), -1, 0, (x[i+1] + x[i]), 1, 0] + [0, 0, 0] * (n-i-1)  for i in range(0, n)])
```

```python
A2 = np.array([[0, 0, 0] * (i-1) + [-(x[i+1] - x[i]), -1, 0, 0, 1, 0] + [0, 0, 0] * (n-i-1)  for i in range(0, n)])
```

### Interval average constraint

For all $i$ in $\{0,\dots,n-1\}$:

$$
\begin{aligned}
& \text{average over interval } i = \frac{1}{x_{i+1} - x_i} [(F_{i+1}(x_{i+1}) - F_{i+1}(x_{i+0.5})) + (F_i(x_i) - F_i(x_{i+0.5}))] = y[i] \\
& \implies \frac{1}{x_{i+1} - x_i} \left[ \left( \left( \frac{a_{i+1}}{3} x_i^3 + \frac{b_{i+1}}{2} x_i^2 + c_{i+1} x_i \right) - \left( \frac{a_{i+1}}{3} x_{i+0.5}^3 + \frac{b_{i+1}}{2} x_{i+0.5}^2 + c_{i+1} x_{i+0.5} \right) \right) + \left( \left( \frac{a_i}{3} x_{i+0.5}^3 + \frac{b_i}{2} x_{i+0.5}^2 + c_i x_{i+0.5} \right) - \left( \frac{a_i}{3} x_i^3 + \frac{b_i}{2} x_i^2 + c_i x_i \right) \right) \right] = y[i] \\
& \implies 2 a_{i+1} x_{i+1}^3 + 3 b_{i+1} x_{i+1}^2 + 6 c_{i+1} x_{i+1} - 2 a_{i+1} x_{i+0.5}^3 - 3 b_{i+1} x_{i+0.5}^2 - 6 c_{i+1} x_{i+0.5} + 2 a_i x_{i+0.5}^3 + 3 b_i x_{i+0.5}^2 + 6 c_i x_{i+0.5} - 2 a_i x_i^3 - 3 b_i x_i^2 - 6 c_i x_i = 6 (x_{i+1} - x_i) \ y[i] \\
& \implies 2 (x_{i+1}^3 - x_{i+0.5}^3) a_{i+1} + 3 (x_{i+1}^2 - x_{i+0.5}^2)  b_{i+1} + 6 (x_{i+1} - x_{i+0.5}) c_{i+1} + 2 (x_{i+0.5}^3 - x_i^3)  a_i + 3 (x_{i+0.5}^2 - x_i^2) b_i + 6 (x_{i+0.5} - x_i) c_i = 6 (x_{i+1} - x_i) \ y[i]
\end{aligned}
$$

This represents $n+1$ linear equations, which can be expressed in the form $A_3\mathbf{x}=\mathbf{b}_3$, where $\mathbf{x}$ is the same as before and:

$$
\mathbf{b}_3 =
\begin{bmatrix}
    6 (x_1 - x_0) \ y[1] \\
    6 (x_2 - x_1) \ y[2] \\
    \dots \\
    6 (x_n - x_{n-1}) \ y[n]
\end{bmatrix}
$$

$$
A_3 =
\begin{bmatrix}
    2 (x_{0.5}^3 - x_0^3) & 3 (x_{0.5}^2 - x_0^2) & 6 (x_{0.5} - x_0) & 2 (x_1^3 - x_{0.5}^3) & 3 (x_1^2 - x_{0.5}^2) & 6 (x_1 - x_{0.5}) & \dots & 0 & 0 & 0 & 0 & 0 & 0 \\
    \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
    0 & 0 & 0 & 0 & 0 & 0 & \dots & 2 (x_{n-0.5}^3 - x_{n-1}^3) & 3 (x_{n-0.5}^2 - x_{n-1}^2) & 6 (x_{n-0.5} - x_{n-1}) & 2 (x_n^3 - x_{n-0.5}^3) & 3 (x_n^2 - x_{n-0.5}^2) & 6 (x_n - x_{n-0.5})
\end{bmatrix}
$$

(The second row has been omitted here for brevity.)

$$
A_3 =
\begin{bmatrix}
    2 (x_{0.5}^3 - x_0^3) & 3 (x_{0.5}^2 - x_0^2) & 6 (x_{0.5} - x_0) & 2 (x_1 - x_{0.5})^3 & 3 (x_1 - x_{0.5})^2 & 6 (x_1 - x_{0.5}) & \dots & 0 & 0 & 0 & 0 & 0 & 0 \\
    \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
    0 & 0 & 0 & 0 & 0 & 0 & \dots & 2 (x_{n-0.5}^3 - x_{n-1}^3) & 3 (x_{n-0.5}^2 - x_{n-1}^2) & 6 (x_{n-0.5} - x_{n-1}) & 2 (x_n - x_{n-0.5})^3 & 3 (x_n - x_{n-0.5})^2 & 6 (x_n - x_{n-0.5})
\end{bmatrix}
$$

$$
A_3 =
\begin{bmatrix}
    2 (((x_1 + x_0) / 2)^3 - x_0^3) & 3 (((x_1 + x_0) / 2)^2 - x_0^2) & 3 (x_1 - x_0) & 2 (x_1 - ((x_1 + x_0) / 2))^3 & 3 (x_1 - ((x_1 + x_0) / 2))^2 & 6 (x_1 - ((x_1 + x_0) / 2)) & \dots & 0 & 0 & 0 & 0 & 0 & 0 \\
    \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
    0 & 0 & 0 & 0 & 0 & 0 & \dots & 2 (((x_n + x_{n-1}) / 2)^3 - x_{n-1}^3) & 3 (((x_n + x_{n-1}) / 2)^2 - x_{n-1}^2) & 6 (((x_n + x_{n-1}) / 2) - x_{n-1}) & 2 (x_n - ((x_n + x_{n-1}) / 2))^3 & 3 (x_n - ((x_n + x_{n-1}) / 2))^2 & 6 (x_n - ((x_n + x_{n-1}) / 2))
\end{bmatrix}
$$

### Boundary conditions

TODO
