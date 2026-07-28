import numpy as np
import pytest


def test_pymc_normal_update():
    pymc = pytest.importorskip("pymc")
    az = pytest.importorskip("arviz")
    from modeling_engine.bayes_pymc import build_normal_model, sample_posterior

    # generate small synthetic data from theta=1.5
    rng = np.random.default_rng(0)
    data = list(rng.normal(1.5, 1.0, size=30))
    model = build_normal_model(0.0, 5.0, data, obs_sd=1.0)
    try:
        res = sample_posterior(model, draws=200, tune=100, chains=2)
    except Exception as e:
        msg = str(e)
        # Skip the test when PyTensor/PyMC fails to compile native extensions in CI
        if "Compilation failed" in msg or "lazylinker" in msg or "pytensor" in msg:
            pytest.skip(f"PyMC/PyTensor compilation failed: {msg}")
        raise
    assert "summary" in res
