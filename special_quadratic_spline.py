from __future__ import annotations

from copy import deepcopy
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.csgraph

BoundaryCondition = Literal["zero-slope", "zero-curvature"]


class SpecialQuadraticSpline(scipy.interpolate.PPoly):
    k: list[float]
    """Knots: x-values splitting up the signal into intervals/blocks."""
    y: list[float]
    """The average value of the signal over each interval."""
    boundary_condition: BoundaryCondition

    def __init__(
        self,
        k: list[float],
        y: list[float],
        boundary_condition: BoundaryCondition = "zero-curvature",
    ):
        self.k = deepcopy(k)
        k = SpecialQuadraticSpline._convert_knots(knots=k)
        self.y = y
        self.boundary_condition = boundary_condition

        n = len(k) - 1

        A1 = np.array(
            [
                [0, 0, 0] * (i - 1)
                + [-((k[i] - k[i - 1]) ** 2), -(k[i] - k[i - 1]), -1, 0, 0, 1]
                + [0, 0, 0] * (n - i - 1)
                for i in range(1, n)
            ]
        )
        b1 = np.zeros((n - 1, 1))

        A2 = np.array(
            [
                [0, 0, 0] * (i - 1)
                + [-2 * (k[i] - k[i - 1]), -1, 0, 0, 1, 0]
                + [0, 0, 0] * (n - i - 1)
                for i in range(1, n)
            ]
        )
        b2 = np.zeros((n - 1, 1))

        A3 = np.array(
            [
                [0, 0, 0] * i
                + [
                    2 * (k[i + 1] - k[i]) ** 3,
                    3 * (k[i + 1] - k[i]) ** 2,
                    6 * (k[i + 1] - k[i]),
                ]
                + [0, 0, 0] * (n - i - 1)
                for i in range(0, n)
            ]
        )
        b3 = np.array([[6 * y[j] * (k[j + 1] - k[j])] for j in range(0, n)])

        if boundary_condition == "zero-slope":
            A4 = np.array(
                [
                    [0, 1, 0] + [0, 0, 0] * (n - 1),
                    [0, 0, 0] * (n - 1) + [2 * (k[n] - k[n - 1]), 1, 0],
                ]
            )
        elif boundary_condition == "zero-curvature":
            A4 = np.array(
                [[2, 0, 0] + [0, 0, 0] * (n - 1), [0, 0, 0] * (n - 1) + [2, 0, 0]]
            )
        b4 = np.array([[0], [0]])

        A = np.concat([A1, A2, A3, A4])
        b = np.concat([b1, b2, b3, b4])

        graph = scipy.sparse.csr_array(A)
        permutation = scipy.sparse.csgraph.reverse_cuthill_mckee(graph)

        graph = graph[permutation, :][:, permutation]
        # ^ TODO: More efficient after `.toarray()`?
        A = graph.toarray()
        b = b[permutation]

        # x = np.linalg.solve(A, b)
        n_below, n_above = scipy.linalg.bandwidth(A)
        ab = SpecialQuadraticSpline._to_banded(n_above, n_below, a=A)
        n_below, _ = scipy.linalg.bandwidth(ab[::-1])  # TODO: Check.
        l = u = ab.shape[0] - n_below - 1  # TODO: Check.
        x = scipy.linalg.solve_banded(l_and_u=(l, u), ab=ab, b=b)

        x = x[pd.Series(permutation).sort_values().index]

        super().__init__(c=x.reshape((n, 3)).T, x=k)

    @staticmethod
    def _to_banded(n_below: int, n_above: int, a: np.ndarray) -> np.ndarray:
        """Convert a square, banded matrix `a` to diagonal ordered form (consumable by
        `np.linalg.solve_banded`).

        Function copied from the following recent SciPy PR, until it is released.
        https://github.com/scipy/scipy/pull/21726/files.
        """

        n = a.shape[0]
        rows = n_above + n_below + 1
        ab = np.zeros((rows, n), dtype=a.dtype)
        ab[n_above] = np.diag(a)
        for i in range(1, n_above + 1):
            ab[n_above - i, i:] = np.diag(a, i)
        for i in range(1, n_below + 1):
            ab[n_above + i, :-i] = np.diag(a, -i)
        return ab

    @staticmethod
    def _convert_knots(knots: list[float]) -> list[float]:
        if isinstance(pd.Index(knots), pd.DatetimeIndex):
            knots = (pd.Index(knots).astype(int) / 1e3).to_list()
        return knots

    @classmethod
    def from_regular_series(
        cls,
        regular_series: pd.Series,
        boundary_condition: BoundaryCondition = "zero-curvature",
    ) -> SpecialQuadraticSpline:

        return cls(
            k=SpecialQuadraticSpline._index_to_knots(regular_series.index),
            y=regular_series.to_list(),
            boundary_condition=boundary_condition,
        )

    @staticmethod
    def _index_to_knots(index: pd.Index) -> list[float]:
        if isinstance(index, pd.RangeIndex):
            delta = index.step
        elif isinstance(index, pd.DatetimeIndex):
            delta = pd.Timedelta(index.freq)
        else:
            raise NotImplementedError
        return [*index, index[-1] + delta]

    def plot(
        self,
        include: list[Literal["input-data-series", "spline"]] = [
            "input-data-series",
            "spline",
        ],
    ) -> go.Figure:
        fig = go.Figure()

        fig.update_layout(title="SignalPerfect", plot_bgcolor="#ECEFF1")
        fig.update_xaxes(zeroline=False, showgrid=False, showticklabels=False)
        fig.update_yaxes(zeroline=False, showgrid=False, showticklabels=False)

        if "input-data-series" in include:
            fig.update_xaxes(
                showticklabels=True,
                tickmode="array",
                tickvals=self.k,
                ticktext=[
                    f"<i>k</i><sub>{i}</sub> = {knot}" for i, knot in enumerate(self.k)
                ],
                color="#1E88E5",
            )
            # Plot knots:
            for knot in self.k:
                fig.add_vline(x=knot, line_dash="dash", line_width=0.5)
            # Plot input data series:
            fig.add_scatter(
                x=self.k,
                y=[*self.y, self.y[-1]],
                line_shape="hv",
                mode="lines",
                line_color="#42A5F5",
                name="Input data series",
                showlegend=True,
            )

        xs = np.linspace(self.k[0], self.k[-1], 1000)
        ys = self(xs)
        if "spline" in include:
            # Plot special quadratic spline:
            fig.add_scatter(
                x=xs,
                y=ys,
                line_color="#EF5350",
                name="Special quadratic spline<br>(fit to input data series)",
                showlegend=True,
            )

        fig.update_yaxes(range=[min(ys), max(ys)])

        return fig

    def get_series(self, k: list[float], plot: bool = False) -> list[float]:
        k = SpecialQuadraticSpline._convert_knots(knots=k)
        series = [
            float(self.integrate(a=k[i], b=k[i + 1]) / (k[i + 1] - k[i]))
            for i in range(len(k) - 1)
        ]
        if plot:
            fig = self.plot(include="spline")
            fig.add_scatter(
                x=k,
                y=[*series, series[-1]],
                line_shape="hv",
                mode="lines",
                line_color="#66BB6A",
                name="Output data series<br>(sampled from special quadratic spline)",
            )
            for i, knot in enumerate(k):
                fig.add_vline(
                    x=knot,
                    line_dash="dash",
                    line_width=0.5,
                    label=dict(
                        text=f"<i>k</i><sub>{i}</sub> = {knot}",
                        font_color="#43A047",
                        textposition="end",
                        textangle=0,
                        padding=3,
                    ),
                )
            fig.show()
        return series

    def get_regular_series(self, index: pd.Index, plot: bool = False) -> pd.Series:
        return pd.Series(
            self.get_series(k=SpecialQuadraticSpline._index_to_knots(index), plot=plot),
            index=index,
        )
