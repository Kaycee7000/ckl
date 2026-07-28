import numpy as np
from typing import Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


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
