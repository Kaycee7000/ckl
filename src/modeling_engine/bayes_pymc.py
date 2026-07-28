"""PyMC integration for non-conjugate Bayesian updates.

This module provides utilities to build a simple PyMC model from a specification
and run MCMC (NUTS) sampling. If PyMC is not installed, the functions raise
an informative error.
"""
from typing import Dict, Any

def build_normal_model(prior_mean: float, prior_sd: float, data: list, obs_sd: float = 1.0):
    try:
        import pymc as pm
    except Exception:
        raise RuntimeError("PyMC is not installed. Install pymc to use the MCMC updater.")

    model = pm.Model()
    with model:
        theta = pm.Normal("theta", mu=prior_mean, sigma=prior_sd)
        obs = pm.Normal("obs", mu=theta, sigma=obs_sd, observed=data)
    return model


def sample_posterior(model, draws: int = 1000, tune: int = 500, chains: int = 2, target_accept: float = 0.9) -> Dict[str, Any]:
    try:
        import pymc as pm
        import arviz as az
    except Exception:
        raise RuntimeError("PyMC or arviz not installed; install pymc and arviz to run sampling")

    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept, cores=1, progressbar=False)
    summary = az.summary(idata)
    return {"inference_data": idata, "summary": summary.to_dict()}
