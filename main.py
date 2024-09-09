import numpy as np
import scipy as sp
import pandas as pd
pd.options.plotting.backend = "plotly"

x = np.array([0, 0.5, 2, 3, 4, 5, 6.5, 7, 8])
y = np.array([np.nan, 3, 1, 4, 1, 5, 9, 2, 6])
n = len(x) - 1

boundary_condition = "zero-second-derivative"

if boundary_condition == "zero-derivative":
    A1 = np.array([[0, 0, 0] * (i-1) + [-(x[i] - x[i-1])**2, -(x[i] - x[i-1]), -1, 0, 0, 1] + [0, 0, 0] * (n-i-1) for i in range(1, n)])
    b1 = np.zeros((n-1, 1))

    A2 = np.array([[0, 0, 0] * (i-1) + [-2 * (x[i] - x[i-1]), -1, 0, 0, 1, 0] + [0, 0, 0] * (n-i-1)  for i in range(1, n)])
    b2 = np.zeros((n-1, 1))

    A3 = np.array([[0, 0, 0] * (i-1) + [2 * (x[i] - x[i-1])**3, 3 * (x[i] - x[i-1])**2, 6 * (x[i] - x[i-1])] + [0, 0, 0] * (n-i) for i in range(1, n+1)])
    b3 = np.array([[6 * y[i] * (x[i] - x[i-1])] for i in range(1, n+1)])

    A4 = np.array([[0, 1, 0] + [0, 0, 0] * (n-1), [0, 0, 0] * (n-1) + [2 * (x[n] - x[n-1]), 1, 0]])
    b4 = np.array([[0], [0]])

    A = np.concat([A1, A2, A3, A4])
    b = np.concat([b1, b2, b3, b4])

elif boundary_condition == "zero-second-derivative":
    A1 = np.array([[0, 0, 0] * (i-1) + [-(x[i+1] - x[i])**2, -2 * (x[i+1] - x[i]), -4, 0, 0, 4] + [0, 0, 0] * (n-i-1) for i in range(0, n)])
    b1 = np.zeros((n, 1))

    A2 = np.array([[0, 0, 0] * (i-1) + [-(x[i+1] - x[i]), -1, 0, 0, 1, 0] + [0, 0, 0] * (n-i-1)  for i in range(0, n)])
    b2 = np.zeros((n, 1))

    A3 = np.array([[0, 0, 0] * i + [2 * (((x[i+1] + x[i]) / 2)^3 - x[i]^3), 3 * (((x[i+1] + x[i]) / 2)^2 - x[i]^2), 3 * (x[i+1] - x[i]), 2 * (), 3 * (), 6 * ()]    for i in range(0, n)])

x = np.linalg.solve(A, b)

ppoly = sp.interpolate.PPoly(c=x.reshape((n, 3)).T, x=x)
xs = np.linspace(x[0], x[-1], 1000)
ys = ppoly(xs)
pd.Series(ys, index=xs).plot().show()#.write_image("tmp.png")
