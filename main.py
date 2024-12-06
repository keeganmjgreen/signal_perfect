from typing import Literal

import numpy as np
import plotly.graph_objects as go
import scipy as sp

BoundaryCondition = Literal["zero-slope", "zero-curvature"]


class SpecialQuadraticSpline(sp.interpolate.PPoly):
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
        self.k = k
        self.y = y
        self.boundary_condition = boundary_condition

        n = len(k) - 1

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

        super().__init__(c=x.reshape((n, 3)).T, x=k)

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
                y=[*self.y[1:], self.y[-1]],
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


if __name__ == "__main__":
    k = np.array([0, 0.5, 2, 3, 4, 5, 6.5, 7, 8])
    sqs = SpecialQuadraticSpline(
        k=k,
        y=np.array([np.nan, 3, 1, 4, 1, 5, 9, 2, 6]),
        boundary_condition="zero-curvature",
    )
    sqs.get_series(k=[5, 6, 7, 8], plot=True)
