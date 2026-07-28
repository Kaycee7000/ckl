from modeling_engine.bayes import conjugate_normal_normal, beta_binomial_update


def test_conjugate_normal_normal():
    prior_mean, prior_var = 0.0, 1.0
    data_mean, data_var, n = 1.0, 1.0, 10
    post_mean, post_var = conjugate_normal_normal(prior_mean, prior_var, data_mean, data_var, n)
    assert post_var < prior_var
    assert abs(post_mean) > 0


def test_beta_binomial():
    a, b = 1.0, 1.0
    s, t = 3, 5
    na, nb = beta_binomial_update(a, b, s, t)
    assert na == a + s
    assert nb == b + (t - s)
