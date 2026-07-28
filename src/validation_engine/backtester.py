from typing import Callable, Dict, Any, List, Optional
import numpy as np
from .metrics import crps_gaussian, logscore_gaussian


class WalkForwardBacktester:
    def __init__(self, model_fn: Callable[[np.ndarray, int], Dict[str, Any]], window_size: int = 100, horizon: int = 1, step: int = 1):
        """model_fn takes (train_series: np.ndarray, horizon: int) and returns dict with keys 'mean' and 'std' (arrays of length horizon) or 'samples' (nsamples x horizon)."""
        self.model_fn = model_fn
        self.window_size = window_size
        self.horizon = horizon
        self.step = step

    def run(self, series: np.ndarray) -> List[Dict[str, Any]]:
        n = len(series)
        results: List[Dict[str, Any]] = []
        # start at window_size, step forward until space for horizon
        for start in range(self.window_size, n - self.horizon + 1, self.step):
            train = series[start - self.window_size:start]
            test_idx = np.arange(start, start + self.horizon)
            test_obs = series[test_idx]
            pred = self.model_fn(train, self.horizon)
            # normalize outputs
            if 'samples' in pred:
                samples = np.array(pred['samples'])
                mean = samples.mean(axis=0)
                std = samples.std(axis=0)
            else:
                mean = np.array(pred.get('mean'))
                std = np.array(pred.get('std'))
                if std is None:
                    std = np.zeros_like(mean)
            # compute CRPS and logscore per horizon point
            crps = [crps_gaussian(m, s if s > 0 else 1e-6, o) for m, s, o in zip(mean, std, test_obs)]
            logs = [logscore_gaussian(m, s if s > 0 else 1e-6, o) for m, s, o in zip(mean, std, test_obs)]
            results.append({
                'train_end': start - 1,
                'test_idx': test_idx,
                'obs': test_obs,
                'mean': mean,
                'std': std,
                'crps': crps,
                'logscore': logs,
            })
        return results
