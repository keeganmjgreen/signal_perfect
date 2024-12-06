from main import SpecialQuadraticSpline


def test_special_quadratic_spline():
    sqs = SpecialQuadraticSpline(
        k=[0, 0.5, 2, 3, 4, 5, 6.5, 7, 8],
        y=[3, 1, 4, 1, 5, 9, 2, 6],
        boundary_condition="zero-curvature",
    )
    series = sqs.get_series(k=[5, 6, 7, 8])
    assert series == [10.124367947293063, 4.3756320527069406, 6.0]
