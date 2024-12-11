import random

import numpy as np
import pandas as pd
import pytest

from special_quadratic_spline import SpecialQuadraticSpline


def test_special_quadratic_spline():
    sqs = SpecialQuadraticSpline(
        k=[0, 0.5, 2, 3, 4, 5, 6.5, 7, 8],
        y=[3, 1, 4, 1, 5, 9, 2, 6],
        boundary_condition="zero-curvature",
    )
    series = sqs.get_series(k=[5, 6, 7, 8])
    np.testing.assert_array_almost_equal(series, [10.124367, 4.375632, 6.0])


def test_from_regular_series():
    sqs = SpecialQuadraticSpline.from_regular_series(
        pd.Series([3, 1, 4, 1, 5, 9, 2, 6])
    )
    series = sqs.get_series(k=[5, 6, 7, 8])
    np.testing.assert_array_almost_equal(series, [9.0, 2.0, 6.0])


def test_from_regular_time_series():
    sqs = SpecialQuadraticSpline.from_regular_series(
        pd.Series(
            [3, 1, 4, 1, 5, 9, 2, 6], index=pd.date_range("2000-01-01 00:00", periods=8)
        )
    )
    series = sqs.get_series(k=pd.date_range("2000-01-06 00:00", periods=4).to_list())
    np.testing.assert_array_almost_equal(series, [9.0, 2.0, 6.0])


def test_get_regular_series():
    sqs = SpecialQuadraticSpline(
        k=[0, 0.5, 2, 3, 4, 5, 6.5, 7, 8],
        y=[3, 1, 4, 1, 5, 9, 2, 6],
        boundary_condition="zero-curvature",
    )
    index = pd.RangeIndex(5, 8)
    series = sqs.get_regular_series(index=index)
    pd.testing.assert_series_equal(
        series, pd.Series([10.1243, 4.3756, 6.0], index=index)
    )


def test_get_regular_time_series():
    sqs = SpecialQuadraticSpline.from_regular_series(
        pd.Series(
            [3, 1, 4, 1, 5, 9, 2, 6], index=pd.date_range("2000-01-01 00:00", periods=8)
        )
    )
    index = pd.date_range("2000-01-06 00:00", periods=3)
    series = sqs.get_regular_series(index=index)
    pd.testing.assert_series_equal(series, pd.Series([9.0, 2.0, 6.0], index=index))


def make_random_signal(signal_length: int) -> pd.Series:
    initial_value = random.gauss()
    signal_values = [initial_value]
    for _ in range(signal_length):
        signal_values.append(random.gauss())
    return pd.Series(signal_values)


@pytest.mark.parametrize("signal_length", [5000])
def test_performance(signal_length, benchmark):
    def make_spline():
        sqs = SpecialQuadraticSpline.from_regular_series(
            make_random_signal(signal_length=signal_length)
        )

    benchmark(make_spline)
