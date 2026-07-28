from causal_engine.scm import SCM


def test_do_and_counterfactual():
    scm = SCM()
    # X has no parents, mean 1 noise
    scm.add_node("X", [], lambda parents: 0.0, noise_fn=lambda: 1.0)
    # Y = 2 * X + U_y, where deterministic_fn returns 2*X
    scm.add_node("Y", ["X"], lambda parents: 2.0 * parents["X"], noise_fn=lambda: 0.0)

    observed = {"X": 1.0, "Y": 3.0}  # implies U_y = 1.0
    # Counterfactual: if X had been 2.0, Y would be 2*2 + U_y = 5.0
    cf = scm.counterfactual(observed, {"X": 2.0}, "Y")
    assert abs(cf - 5.0) < 1e-6


def test_causal_attribution_linear():
    scm = SCM()
    scm.add_node("X", [], lambda parents: 1.0, noise_fn=lambda: 0.0)  # X deterministic=1
    scm.add_node("Y", ["X"], lambda parents: 2.0 * parents["X"], noise_fn=lambda: 0.0)
    # Attribution of X on Y should be approx 2.0 * E[X] = 2.0
    attr = scm.causal_attribution("X", "Y", trials=10)
    assert abs(attr - 2.0) < 1e-6
