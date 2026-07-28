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
    def __init__(self, model_fn: Callable[[np.ndarray], Any], scm: Optional[SCM] = None, param_names: Optional[Sequence[str]] = None, workers: int = None):
        self.model_fn = model_fn
        self.scm = scm
        # Optional ordering of parameter names mapping vector indices to variable names in SCM
        self.param_names = list(param_names) if param_names is not None else None
        self.runner = MonteCarloRunner(workers=workers)

    def run_scenarios(self, param_matrix: np.ndarray, interventions: Optional[Dict[int, Dict[str, float]]] = None, parallel: bool = True):
        """Run model_fn for each parameter vector.

        - `param_matrix`: array-like of shape (n_samples, n_params)
        - `interventions`: optional dict mapping sample index -> dict of SCM interventions

        If an `SCM` and `param_names` are provided, SCM-generated values will replace
        the corresponding entries in the parameter vector for variables present in the SCM.
        """
        samples = [param for param in param_matrix]
        interventions = interventions or {}

        def wrapped_with_intervention(args):
            p, idx = args
            # copy to avoid mutating input
            p_adj = np.array(p, copy=True)
            # apply SCM intervention mapping if available for this sample
            if self.scm is not None and self.param_names is not None and idx in interventions:
                inter = interventions[idx]
                # apply do-intervention to SCM
                scm_do = self.scm.do(inter)
                # sample once from the intervened SCM
                scm_sample = scm_do.sample(n=1)[0]
                # replace any params that have names in scm_sample
                for i, name in enumerate(self.param_names):
                    if name in scm_sample:
                        p_adj[i] = scm_sample[name]
            return self.model_fn(p_adj)

        # prepare indexed samples for wrapped function
        indexed = [(samples[i], i) for i in range(len(samples))]
        res = self.runner.run(indexed, wrapped_with_intervention, parallel=parallel)
        return res
