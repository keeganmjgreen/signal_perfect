import numpy as np

from special_quadratic_spline import SpecialQuadraticSpline

sqs = SpecialQuadraticSpline(
    k=np.array([0, 0.5, 2, 3, 4, 5, 6.5, 7, 8]),
    y=np.array([3, 1, 4, np.nan, 5, 9, 2, np.nan]),
    boundary_condition="zero-curvature",
)
sqs.plot(include=["input-data-series"]).write_html("docs/demo/input.html")
sqs.plot(include=["input-data-series", "spline"]).write_html(
    "docs/demo/input,spline.html"
)

k = [0, 5, 6, 7, 7.5]
series = sqs.get_series(k)
sqs.plot_output_series(k, series).write_html("docs/demo/spline,output.html")
print(f">>> sqs.get_series(k={k}, plot=True)")
print(series)
