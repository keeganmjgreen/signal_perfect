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


class SpecialQuadraticSpline(scipy.interpolate.PPoly):
    k: list[float]
    """Knots: x-values splitting up the signal into intervals/blocks."""
    y: list[float]
    """The average value of the signal over each interval."""

    def __init__(self, k: list[float], y: list[float]):
        self.k = deepcopy(k)
        k = SpecialQuadraticSpline._convert_knots(knots=k)
        self.y = y

        n = len(k) - 1

        # Initialize matrix A and vector b to zeros.
        A = np.zeros((6 * n - 2, 10))
        # ^ Banded storage format (not stored as square matrix).
        b = np.zeros(6 * n - 2)

        # Set row elements of matrix A corresponding to submatrix A1 (knot constraint):
        for i in range(1, n):
            A[4 + 6 * (i - 1), 1 : 1 + 9] = [
                -((k[i] - k[i - 1]) ** 2),
                -(k[i] - k[i - 1]),
                -1,
                0,
                0,
                0,
                0,
                0,
                1,
            ]

        # Set row elements of matrix A corresponding to submatrix A2 (knot derivative constraint):
        for i in range(1, n):
            A[5 + 6 * (i - 1), 0 : 0 + 9] = [
                -2 * (k[i] - k[i - 1]),
                -1,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
            ]

        # Set row elements of matrix A corresponding to submatrix A3, and elements of vector b
        #     corresponding to subvector b3 (interval average constraint):
        for i in range(0, n):
            if not np.isnan(y[i]):
                A[0 + 6 * i, 5 : 5 + 3] = [
                    2 * (k[i + 1] - k[i]) ** 3,
                    3 * (k[i + 1] - k[i]) ** 2,
                    6 * (k[i + 1] - k[i]),
                ]
                b[0 + 6 * i] = 6 * y[i] * (k[i + 1] - k[i])
            else:
                # This interval's y value is missing. Constrain this interval's spline segment to be
                #     linear (by constraining its polynomial term a to zero):
                A[0 + 6 * i, 5] = 1

        # A4:
        for i in range(0, n):
            A[1 + 6 * i, 4 : 4 + 6] = [
                2 / 5 * (k[i + 1] - k[i]) ** 5,
                1 / 2 * (k[i + 1] - k[i]) ** 4,
                2 / 3 * (k[i + 1] - k[i]) ** 3,
                1 / 3 * (k[i + 1] - k[i]) ** 2,
                (-((k[i + 1] - k[i]) ** 2) if i != n - 1 else 0),
                (-2 * (k[i + 1] - k[i]) if i != n - 1 else 0),
            ]
            b[1 + 6 * i] = 2 / 3 * y[i] * (k[i + 1] - k[i]) ** 3

        # A5:
        for i in range(0, n):
            A[2 + 6 * i, 2 : 2 + 7] = [
                (1 if i != 0 else 0),
                1 / 2 * (k[i + 1] - k[i]) ** 4,
                2 / 3 * (k[i + 1] - k[i]) ** 3,
                (k[i + 1] - k[i]) ** 2,
                1 / 2 * (k[i + 1] - k[i]),
                (-(k[i + 1] - k[i]) if i != n - 1 else 0),
                (-1 if i != n - 1 else 0),
            ]
            b[2 + 6 * i] = y[i] * (k[i + 1] - k[i]) ** 2

        # A6:
        for i in range(0, n):
            A[3 + 6 * i, 0 : 0 + 8] = [
                (1 if i != 0 else 0),
                0,
                2 / 3 * (k[i + 1] - k[i]) ** 3,
                (k[i + 1] - k[i]) ** 2,
                2 * (k[i + 1] - k[i]),
                1,
                (-1 if i != n - 1 else 0),
                0,
            ]
            b[3 + 6 * i] = 2 * y[i] * (k[i + 1] - k[i])

        # Convert matrix A from "row-major banded storage format" (which was easy to construct, as
        #     above) to "column-major banded storage format" (required by
        #     `scipy.linalg.solve_banded`):
        for col in range(10):
            A[:, col] = np.roll(A[:, col], shift=(col - 5), axis=0)
        A = np.rot90(A)

        x = scipy.linalg.solve_banded(l_and_u=(5, 4), ab=A, b=b)

        super().__init__(c=np.resize(x, 6 * n).reshape((2 * n, 3))[::2].T, x=k)

    @staticmethod
    def _convert_knots(knots: list[float]) -> list[float]:
        if isinstance(pd.Index(knots), pd.DatetimeIndex):
            knots = (pd.Index(knots).astype(int) / 1e3).to_list()
        return knots

    @classmethod
    def from_regular_series(cls, regular_series: pd.Series) -> SpecialQuadraticSpline:

        return cls(
            k=SpecialQuadraticSpline._index_to_knots(regular_series.index),
            y=regular_series.to_list(),
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
                x=[self.k[0], *self.k[1:-1].repeat(2), self.k[-1]],
                y=[*self.y[:-1].repeat(2), self.y[-1]],
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
            fig = self.plot_output_series(k, series)
            fig.show()
        return series

    def plot_output_series(self, k: list[float], series: list[float]) -> go.Figure:
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
        return fig

    def get_regular_series(self, index: pd.Index, plot: bool = False) -> pd.Series:
        return pd.Series(
            self.get_series(k=SpecialQuadraticSpline._index_to_knots(index), plot=plot),
            index=index,
        )
