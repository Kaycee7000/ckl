import numpy as np
import pytest

from simulation_engine.sensitivity import sobol_sensitivity


def test_sobol_skip_if_missing():
    try:
        import SALib  # type: ignore
    except Exception:
        pytest.skip("SALib not installed; skipping Sobol integration test")


def test_sobol_simple_additive():
    # This test requires SALib; skip if not present
    try:
        from SALib.sample import saltelli
    except Exception:
        pytest.skip("SALib not installed; skipping Sobol integration test")

    # simple additive model Y = x0 + 2*x1 over [0,1]
    def model(X):
        return X[:, 0] + 2.0 * X[:, 1]

    problem = {"num_vars": 2, "names": ["x0", "x1"], "bounds": [[0, 1], [0, 1]]}
    # Generate Saltelli samples
    N = 256
    param_values = saltelli.sample(problem, N, calc_second_order=False)
    Y = model(param_values)
    Si = sobol_sensitivity(param_values, Y, ["x0", "x1"], calc_second_order=False)

    # For Y = x0 + 2*x1, relative first-order indices should reflect variance ratios (~1:4)
    s1 = Si["S1"]
    # normalize for comparison
    s1_norm = s1 / np.nansum(s1)
    assert s1_norm[1] > s1_norm[0]
