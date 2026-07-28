import numpy as np
from simulation_engine.scenario import lhs_sample, ScenarioRunner
from simulation_engine.clustering import cluster_trajectories_kmeans, cluster_trajectories_gmm
from simulation_engine.sensitivity import correlation_sensitivity, permutation_importance_sensitivity


def simple_model(p):
    # p is a vector; produce a trajectory: t -> sum(p) * t
    tot = float(np.sum(p))
    t = np.linspace(0, 1, 5)
    return tot * t


def test_lhs_and_runner():
    samp = lhs_sample(10, 3, seed=0)
    assert samp.shape == (10, 3)
    runner = ScenarioRunner(simple_model)
    results = runner.run_scenarios(samp, parallel=False)
    assert results.shape[0] == 10


def test_clustering():
    # create 6 synthetic trajectories: 3 of type A, 3 of type B
    A = np.array([np.linspace(0, 1, 5) for _ in range(3)])
    B = np.array([np.linspace(1, 0, 5) for _ in range(3)])
    traj = np.vstack([A, B])
    labels, centers = cluster_trajectories_kmeans(traj, n_clusters=2)
    assert set(labels) == {0, 1}
    labels2, centers2, probs = cluster_trajectories_gmm(traj, n_components=2)
    assert len(probs) == 2


def test_sensitivity():
    np.random.seed(0)
    params = np.random.rand(100, 3)
    # output depends mostly on param 1
    outputs = params[:, 0] * 2.0 + 0.1 * np.random.randn(100)
    corr = correlation_sensitivity(params, outputs)
    assert corr.shape[0] == 3
    perm = permutation_importance_sensitivity(params, outputs)
    assert perm.shape[0] == 3
