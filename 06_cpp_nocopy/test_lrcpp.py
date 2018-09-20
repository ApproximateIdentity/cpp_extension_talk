import pytest

from data import generate_data
from lrpy import LogisticRegression as LogisticRegressionPy
from lrcpp import LogisticRegression as LogisticRegressionCpp


# Generate some random parameters for input to tests
params = []
for size in [20, 50, 100, 500]:
    for mean1 in [(-1,0) , (1, 1)]:
        for mean2 in [(0,-1), (1, -1)]:
            params.append((size, mean1, mean2))

@pytest.mark.parametrize("size,mean1,mean2", params)

def test_compute_coefficients(size, mean1, mean2):
    X, Y = generate_data(size, mean1, mean2)
    lrpy = LogisticRegressionPy()
    lrpy_coef = lrpy.compute_coefficients(X, Y)
    lrcpp = LogisticRegressionCpp()
    lrcpp_coef = lrcpp.compute_coefficients(X, Y)
    assert abs(lrpy_coef[0] - lrcpp_coef[0]) < 0.01
    assert abs(lrpy_coef[1] - lrcpp_coef[1]) < 0.01
    assert abs(lrpy_coef[2] - lrcpp_coef[2]) < 0.01
