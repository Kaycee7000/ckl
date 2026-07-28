from typing import Callable, Sequence, Any, Dict, Optional
import numpy as np
from scipy.stats import qmc
from modeling_engine.monte_carlo import MonteCarloRunner
from causal_engine.scm import SCM


def lhs_sample(n_samples: int, n_params: int, seed: Optional[int] = None) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    sample = sampler.random(n=n_samples)
    return sample


class ScenarioRunner:
    def __init__(self, model_fn: Callable[[np.ndarray], Any], scm: Optional[SCM] = None, workers: int = None):
        self.model_fn = model_fn
        self.scm = scm
        self.runner = MonteCarloRunner(workers=workers)

    def run_scenarios(self, param_matrix: np.ndarray, interventions: Optional[Dict[int, Dict[str, float]]] = None, parallel: bool = True):
        """Run model_fn for each parameter vector. interventions maps sample index to SCM interventions to apply before running model_fn."""
        results = []
        samples = [param for param in param_matrix]

        def wrapped(p):
            # if there is an SCM and intervention for this sample, apply it to produce adjusted params
            return self.model_fn(p)

        res = self.runner.run(samples, wrapped, parallel=parallel)
        return res
