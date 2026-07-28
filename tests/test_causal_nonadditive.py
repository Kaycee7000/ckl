from causal_engine.scm import SCM


def test_nonadditive_abduction_with_inverse():
    scm = SCM()
    # X root node deterministic=2 (no noise)
    scm.add_node("X", [], lambda parents: 2.0, noise_fn=None)
    # Y = X * U (multiplicative noise); provide inverse_fn to compute U = Y / f(parents)
    def y_det(parents):
        return parents["X"]

    def y_inverse(observed, parents):
        base = parents["X"]
        return observed / base

    scm.add_node("Y", ["X"], deterministic_fn=y_det, noise_fn=None, inverse_fn=y_inverse)

    observed = {"X": 2.0, "Y": 10.0}
    us = scm.abduct_noise(observed)
    # U for Y should be 5.0 (because 2 * U = 10)
    assert abs(us["Y"] - 5.0) < 1e-9
