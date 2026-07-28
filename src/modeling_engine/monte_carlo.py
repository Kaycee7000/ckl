import multiprocessing as mp
import numpy as np
from typing import Callable, Iterable, Sequence, Any, Optional


class MonteCarloRunner:
    def __init__(self, workers: Optional[int] = None):
        self.workers = workers or max(1, mp.cpu_count() - 1)

    def _worker_wrap(self, args):
        func, params = args
        return func(params)

    def run(self, param_samples: Sequence[Any], model_fn: Callable[[Any], Any], parallel: bool = True) -> np.ndarray:
        """Run model_fn over param_samples. Returns numpy array of results."""
        if parallel:
            try:
                with mp.Pool(self.workers) as pool:
                    results = pool.map(self._worker_wrap, [(model_fn, p) for p in param_samples])
            except Exception:
                # Fall back to serial if model_fn isn't picklable or other multiprocessing error
                results = [model_fn(p) for p in param_samples]
        else:
            results = [model_fn(p) for p in param_samples]
        try:
            return np.array(results)
        except Exception:
            return np.array(list(results))

    def vectorized_run(self, param_array: np.ndarray, vectorized_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Run a vectorized function on a NumPy array of parameters."""
        return vectorized_fn(param_array)
