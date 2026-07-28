import numpy as np
from scipy.stats import norm
from typing import Sequence, Tuple


def crps_gaussian(mu: float, sigma: float, x: float) -> float:
    """Closed-form CRPS for Gaussian predictive distribution N(mu, sigma^2) and observation x."""
    if sigma <= 0:
        return abs(x - mu)
    z = (x - mu) / sigma
    crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(crps)


def logscore_gaussian(mu: float, sigma: float, x: float) -> float:
    if sigma <= 0:
        # degenerate
        return float((x - mu) ** 2)
    return 0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * ((x - mu) ** 2) / (sigma ** 2)


def mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(y_true == 0, np.finfo(float).eps, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def brier_score(prob: Sequence[float], outcome: Sequence[int]) -> float:
    prob = np.array(prob)
    outcome = np.array(outcome)
    return float(np.mean((prob - outcome) ** 2))


def coverage_probability(mu: Sequence[float], sigma: Sequence[float], y: Sequence[float], alpha: float = 0.9) -> float:
    """Compute empirical coverage for central (alpha) prediction intervals."""
    lower_q = (1 - alpha) / 2
    upper_q = 1 - lower_q
    lower = norm.ppf(lower_q, loc=np.array(mu), scale=np.array(sigma))
    upper = norm.ppf(upper_q, loc=np.array(mu), scale=np.array(sigma))
    y = np.array(y)
    return float(np.mean((y >= lower) & (y <= upper)))


def calibration_curve(prob_preds: Sequence[float], outcomes: Sequence[int], n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    probs = np.array(prob_preds)
    y = np.array(outcomes)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins) - 1
    prop_true = np.zeros(n_bins)
    prop_pred = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for i in range(n_bins):
        mask = bin_ids == i
        counts[i] = mask.sum()
        if counts[i] > 0:
            prop_true[i] = y[mask].mean()
            prop_pred[i] = probs[mask].mean()
        else:
            prop_true[i] = np.nan
            prop_pred[i] = np.nan
    return prop_pred, prop_true
