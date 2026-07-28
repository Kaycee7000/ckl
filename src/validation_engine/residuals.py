import numpy as np
from typing import Dict, Any


def residual_profile(pred_mean: np.ndarray, obs: np.ndarray) -> Dict[str, Any]:
    res = obs - pred_mean
    return {
        'mean_residual': float(np.mean(res)),
        'std_residual': float(np.std(res)),
        'min_residual': float(np.min(res)),
        'max_residual': float(np.max(res)),
        'quantiles': np.quantile(res, [0.25, 0.5, 0.75]).tolist(),
    }
