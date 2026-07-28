import numpy as np
from validation_engine.backtester import WalkForwardBacktester
from validation_engine.metrics import mape, coverage_probability
from validation_engine.residuals import residual_profile


def simple_persistence_model(train: np.ndarray, horizon: int):
    # predict last observed value with std equal to training std
    last = float(train[-1])
    std = float(np.std(train))
    mean = np.array([last for _ in range(horizon)])
    stds = np.array([std for _ in range(horizon)])
    return {'mean': mean, 'std': stds}


def test_backtester_and_metrics():
    np.random.seed(0)
    n = 200
    time = np.arange(n)
    series = 0.1 * time + np.random.randn(n)  # linear trend + noise
    backtester = WalkForwardBacktester(simple_persistence_model, window_size=50, horizon=1, step=10)
    results = backtester.run(series)
    assert len(results) > 0
    # aggregate predictions and obs
    preds = np.array([r['mean'][0] for r in results])
    stds = np.array([r['std'][0] for r in results])
    obs = np.array([r['obs'][0] for r in results])
    # compute MAPE
    _mape = mape(obs, preds)
    assert _mape >= 0
    # coverage probability for 90% interval
    cp = coverage_probability(preds, stds, obs, alpha=0.9)
    assert 0.0 <= cp <= 1.0
    rp = residual_profile(preds, obs)
    assert 'mean_residual' in rp
