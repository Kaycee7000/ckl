import numpy as np
from typing import Tuple


def conjugate_normal_normal(prior_mean: float, prior_var: float, data_mean: float, data_var: float, n: int) -> Tuple[float, float]:
    """Return posterior mean and variance for Normal-Normal conjugate update.

    Prior: theta ~ N(prior_mean, prior_var)
    Likelihood: data ~ N(theta, data_var) (n iid)
    """
    if n <= 0:
        return prior_mean, prior_var
    # posterior variance
    post_var = 1.0 / (1.0 / prior_var + n / data_var)
    post_mean = post_var * (prior_mean / prior_var + n * data_mean / data_var)
    return post_mean, post_var


def beta_binomial_update(prior_a: float, prior_b: float, successes: int, trials: int) -> Tuple[float, float]:
    """Return posterior alpha,beta for Beta-Binomial conjugacy."""
    return prior_a + successes, prior_b + (trials - successes)


def pymc_update_placeholder(*args, **kwargs):
    try:
        import pymc as pm
    except Exception:
        raise RuntimeError("PyMC not installed; install pymc to use the MCMC updater")
    # This is a placeholder to show where you'd implement PyMC model fitting
    raise NotImplementedError("PyMC-based updater not implemented in scaffold")
