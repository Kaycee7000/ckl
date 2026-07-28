import numpy as np
from continuous_learning.residual_monitor import ResidualMonitor
from continuous_learning.proposer import propose_adjustment_from_residuals
from continuous_learning.gatekeeper import Gatekeeper
from validation_engine.backtester import WalkForwardBacktester


def model_factory_from_artifact(artifact):
    # returns a model function that uses artifact['structural_equations']['bias'] as bias added to persistence prediction
    bias = artifact.get('structural_equations', {}).get('bias', 0.0)

    def model(train, horizon):
        last = float(train[-1])
        mean = np.array([last + bias for _ in range(horizon)])
        std = np.array([np.std(train) for _ in range(horizon)])
        return {'mean': mean, 'std': std}

    return model


def test_residual_proposal_and_gatekeeper():
    np.random.seed(0)
    # create series with true bias of -0.5 relative to model with bias 0
    n = 150
    series = 0.5 * np.arange(n) + np.random.randn(n) - 0.5
    # artifact with zero bias
    artifact = {"id": "a1", "structural_equations": {"bias": 0.0}}

    # simulate predictions and record residuals
    monitor = ResidualMonitor()
    for t in range(100, 110):
        train = series[t-50:t]
        pred = model_factory_from_artifact(artifact)(train, 1)
        monitor.record(artifact['id'], pred['mean'][0], series[t])

    summary = monitor.summary_for_artifact('a1')
    proposal = propose_adjustment_from_residuals(artifact, summary, threshold=0.01)
    assert proposal['action'] == 'update_artifact'

    # gatekeeper evaluation: use a backtester (window 50 horizon 1) and series
    backtester = WalkForwardBacktester(model_factory_from_artifact(artifact), window_size=50, horizon=1, step=10)
    gate = Gatekeeper(backtester, series)
    result = gate.evaluate_proposal(artifact, proposal, model_factory_from_artifact)
    assert 'accept' in result
