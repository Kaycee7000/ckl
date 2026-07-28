import math
from typing import List, Dict


def confidence_from_backtests(backtest_results: List[Dict]) -> float:
    """Compute a simple empirical confidence score in [0,1] from backtest results.

    backtest_results: list of dicts each containing 'crps' (list) and optional 'n_samples'
    Score increases with more historical validations and lower CRPS.
    """
    if not backtest_results:
        return 0.0
    # collect mean CRPS per fold and sample counts
    crps_means = [sum(r['crps']) / len(r['crps']) if len(r['crps']) > 0 else float('inf') for r in backtest_results]
    n_folds = len(crps_means)
    mean_crps = sum(crps_means) / n_folds
    # use number of effective historical analogues = total folds
    N = n_folds
    # transform into confidence: higher when mean_crps small and N large
    score = (1.0 / (1.0 + mean_crps)) * (1.0 - math.exp(-math.sqrt(N)))
    # clamp
    return max(0.0, min(1.0, score))
