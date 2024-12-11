import re
import subprocess

import pandas as pd
import plotly.graph_objects as go

MILLISECONDS_PER_SECOND = 1000

pytest_output = subprocess.run(
    ["pytest", "test_special_quadratic_spline.py"],
    stdout=subprocess.PIPE,
).stdout.decode()

# Remove non-plaintext characters:
pytest_output = re.sub(r"\x1b\[([0-9;]*[mK])", "", pytest_output)

lines = pytest_output.split("\n")

# Convert n tab characters to a space character:
lines = [re.sub(r"\s+", " ", line) for line in lines]
# Split up each line by its whitespace:
whitespace_separated_lines = [line.split(" ") for line in lines]

performance_lines = [
    line
    for line in whitespace_separated_lines
    if line[0].startswith("test_performance")
]
mean_exec_times = (
    pd.Series(
        {
            int(re.findall(r"test_performance\[(\d+)\]", line[0])[0]): float(
                line[5].replace(",", "")
            )
            / MILLISECONDS_PER_SECOND
            for line in performance_lines
        },
        name="mean_exec_time_s",
    )
    .rename_axis("signal_length")
    .sort_index()
)

fig: go.Figure = mean_exec_times.plot.bar(backend="plotly")
fig = fig.update_layout(
    xaxis_range=[0, float("inf")],
    xaxis_title="Signal Length",
    yaxis_range=[0, float("inf")],
    yaxis_title="Mean Execution Time (Seconds)",
    showlegend=False,
    title="SignalPerfect benchmark",
)
fig.write_html("docs/performance/benchmark.html")
