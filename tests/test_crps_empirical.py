import numpy as np

from validation_engine.metrics import crps_empirical


def test_crps_empirical_two_point():
    samples = np.array([0.0, 2.0])
    x = 1.0
    # mean abs to x = (1 + 1)/2 = 1. pairwise mean abs = 1. CRPS = 1 - 0.5*1 = 0.5
    crps = crps_empirical(samples, x)
    assert abs(crps - 0.5) < 1e-12


def test_crps_empirical_degenerate():
    samples = np.array([1.0, 1.0, 1.0])
    x = 1.0
    crps = crps_empirical(samples, x)
    assert abs(crps - 0.0) < 1e-12


def test_crps_empirical_vectorized():
    # two predictive vectors as columns
    samples = np.array([[0.0, 1.0], [2.0, 3.0]])  # shape (2,2)
    x = np.array([1.0, 2.0])
    crps = crps_empirical(samples, x)
    # first column same as test_crps_empirical_two_point -> 0.5
    assert abs(crps[0] - 0.5) < 1e-12
    # second column samples [1,3], x=2 => symmetric -> 0.5
    assert abs(crps[1] - 0.5) < 1e-12
