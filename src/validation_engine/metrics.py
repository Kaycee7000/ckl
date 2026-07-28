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


def crps_empirical(samples: np.ndarray, x: float) -> float:
    """Empirical CRPS for predictive samples and observation x.

    Uses the Monte Carlo estimator:
      CRPS = (1/N) sum_i |s_i - x| - 0.5 * (1/N^2) sum_{i,j} |s_i - s_j|

    Accepts `samples` as a 1-D array of predictive draws, or a 2-D array
    of shape (N, M) to compute CRPS for M separate predictive vectors (column-wise).
    Returns a float for 1-D input or a numpy array of length M for 2-D input.
    """
    s = np.asarray(samples)
    if s.ndim == 1:
        s = s.reshape(-1, 1)

    N, M = s.shape
    if N == 0:
        raise ValueError("samples must contain at least one draw")

    # sort each column for efficient pairwise-diff computation
    s_sorted = np.sort(s, axis=0)

    # mean absolute deviation to observation x
    abs_to_x = np.mean(np.abs(s - np.asarray(x).reshape(1, -1)), axis=0)

    # compute sum_{i<j} (s_j - s_i) efficiently for each column
    # formula: sum_{i<j} (s_j - s_i) = sum_k s_sorted[k] * (2*k - N + 1)
    idx = np.arange(N)
    coeffs = (2 * idx - N + 1).reshape(N, 1)
    sum_pairwise = np.sum(s_sorted * coeffs, axis=0)

    mean_pairwise_abs = (2.0 / (N * N)) * sum_pairwise

    crps = abs_to_x - 0.5 * mean_pairwise_abs
    # if single column, return scalar
    if crps.size == 1:
        return float(crps[0])
    return crps
