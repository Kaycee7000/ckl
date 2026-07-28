import numpy as np
from modeling_engine.monte_carlo import MonteCarloRunner


def test_monte_carlo_serial_and_parallel():
    runner = MonteCarloRunner(workers=2)

    def model(p):
        # simple deterministic function of parameter
        return p[0] + p[1]

    samples = [np.array([i, i * 2]) for i in range(10)]
    res_serial = runner.run(samples, model, parallel=False)
    res_parallel = runner.run(samples, model, parallel=True)
    assert res_serial.shape == res_parallel.shape
    assert np.allclose(res_serial, res_parallel)
