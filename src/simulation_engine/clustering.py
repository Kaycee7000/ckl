import numpy as np
from typing import Tuple
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def cluster_trajectories_kmeans(trajectories: np.ndarray, n_clusters: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Clusters trajectories (samples x timesteps x features) by flattening features and returns labels and cluster centers."""
    n_samples = trajectories.shape[0]
    X = trajectories.reshape(n_samples, -1)
    km = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    centers = km.cluster_centers_.reshape(n_clusters, *trajectories.shape[1:])
    return km.labels_, centers


def cluster_trajectories_gmm(trajectories: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = trajectories.shape[0]
    X = trajectories.reshape(n_samples, -1)
    gmm = GaussianMixture(n_components=n_components, random_state=0).fit(X)
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X).mean(axis=0)
    centers = gmm.means_.reshape(n_components, *trajectories.shape[1:])
    return labels, centers, probs
