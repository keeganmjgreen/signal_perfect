# Demo

Consider the following signal that has been split up into intervals by $t$-values called knots ($k$). The average value ($y$) of the signal is recorded over each interval:

--8<-- "docs/demo/input.html"

A special quadratic spline can be fit to these data as follows, by instantiating the `SpecialQuadraticSpline` class from the knots and $y$ values:

```python
>>> sqs = SpecialQuadraticSpline(
>>>     k=np.array([0, 0.5, 2, 3, 4, 5, 6.5, 7, 8]),  # Knots: x-values splitting up the signal into intervals/blocks.
>>>     y=np.array([3, 1, 4, 1, 5, 9, 2, 6]),  # The average value of the signal over each interval.
>>> )
>>> sqs.plot(include=["input-data-series", "spline"])
```

--8<-- "docs/demo/input,spline.html"

Note that this is quite different to something like a polynomial fit or cubic smoothing spline, putting the "special" in "special quadratic spline". This spline is not fit through any data points, because we do not have any instantaneous data points of the original signal --- only its average values over certain intervals of time. The spline's average over each of these intervals is constrained to be equal to the signal's average over each of these intervals.

Now, the special quadratic spline can be sampled using the `get_series` method:

```python
>>> sqs.get_series(k=[5, 6, 7, 8], plot=True)
[10.124367947293063, 4.3756320527069406, 6.0]
```

--8<-- "docs/demo/spline,output.html"
