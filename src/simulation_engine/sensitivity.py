import numpy as np
from typing import Tuple, Sequence
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
try:
    from SALib.analyze import sobol
    from SALib.util import read_param_file
    SALIB_AVAILABLE = True
except Exception:
    SALIB_AVAILABLE = False


def correlation_sensitivity(params: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    # params: samples x features, outputs: samples or samples x targets
    if outputs.ndim > 1:
        y = outputs.mean(axis=1)
    else:
        y = outputs
    corrs = []
    for i in range(params.shape[1]):
        r = np.corrcoef(params[:, i], y)[0, 1]
        corrs.append(abs(r))
    corrs = np.array(corrs)
    if np.isnan(corrs).any():
        corrs = np.nan_to_num(corrs)
    # normalize
    if corrs.sum() == 0:
        return corrs
    return corrs / corrs.sum()


def permutation_importance_sensitivity(params: np.ndarray, outputs: np.ndarray, n_repeats: int = 10) -> np.ndarray:
    # Fit a surrogate model and compute permutation importance as proxy for global sensitivity
    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(params, outputs)
    res = permutation_importance(rf, params, outputs, n_repeats=n_repeats, random_state=0)
    imp = res.importances_mean
    if imp.sum() == 0:
        return imp
    return imp / imp.sum()


def sobol_sensitivity(params: np.ndarray, outputs: np.ndarray, param_names: Sequence[str], calc_second_order: bool = False) -> dict:
    """Compute Sobol sensitivity indices using SALib's `sobol.analyze`.

    - `params`: matrix of samples (N x k) generated with Saltelli/Sobol sampling
    - `outputs`: vector of model outputs corresponding to `params`
    - `param_names`: list of parameter names of length k
    Returns the dictionary returned by `sobol.analyze` (keys: S1, ST, S2...)
    """
    if not SALIB_AVAILABLE:
        raise RuntimeError("SALib is not installed. Install SALib to compute Sobol indices.")

    # Build problem dict required by SALib.analyze.sobol
    k = params.shape[1]
    # We don't have explicit bounds here; assume [0,1] if unknown. Users should sample accordingly.
    problem = {"num_vars": k, "names": list(param_names), "bounds": [[0.0, 1.0]] * k}

    # SALib expects output as 1D array
    Y = outputs if outputs.ndim == 1 else outputs.ravel()
    Si = sobol.analyze(problem, Y, calc_second_order=calc_second_order, print_to_console=False)
    return Si
