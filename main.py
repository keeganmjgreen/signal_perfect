import numpy as np
import scipy as sp
import pandas as pd
pd.options.plotting.backend = "plotly"

k = np.array([0, 0.5, 2, 3, 4, 5, 6.5, 7, 8])
y = np.array([np.nan, 3, 1, 4, 1, 5, 9, 2, 6])
n = len(k) - 1

boundary_condition = "zero-curvature"

A1 = np.array([[0, 0, 0] * (i-1) + [-(k[i] - k[i-1])**2, -(k[i] - k[i-1]), -1, 0, 0, 1] + [0, 0, 0] * (n-i-1) for i in range(1, n)])
b1 = np.zeros((n-1, 1))

A2 = np.array([[0, 0, 0] * (i-1) + [-2 * (k[i] - k[i-1]), -1, 0, 0, 1, 0] + [0, 0, 0] * (n-i-1)  for i in range(1, n)])
b2 = np.zeros((n-1, 1))

A3 = np.array([[0, 0, 0] * (i-1) + [2 * (k[i] - k[i-1])**3, 3 * (k[i] - k[i-1])**2, 6 * (k[i] - k[i-1])] + [0, 0, 0] * (n-i) for i in range(1, n+1)])
b3 = np.array([[6 * y[i] * (k[i] - k[i-1])] for i in range(1, n+1)])

if boundary_condition == "zero-slope":
    A4 = np.array([[0, 1, 0] + [0, 0, 0] * (n-1), [0, 0, 0] * (n-1) + [2 * (k[n] - k[n-1]), 1, 0]])
elif boundary_condition == "zero-curvature":
    A4 = np.array([[2, 0, 0] + [0, 0, 0] * (n-1), [0, 0, 0] * (n-1) + [2, 0, 0]])
b4 = np.array([[0], [0]])

A = np.concat([A1, A2, A3, A4])
b = np.concat([b1, b2, b3, b4])

x = np.linalg.solve(A, b)

ppoly = sp.interpolate.PPoly(c=x.reshape((n, 3)).T, x=k)
xs = np.linspace(k[0], k[-1], 1000)
ys = ppoly(xs)
pd.Series(ys, index=xs).plot().show()#.write_image("tmp.png")
