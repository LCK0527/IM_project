import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import entropy


def compute_anchor(vectors: np.ndarray) -> np.ndarray:
    """Compute mean vector as anchor (1D array of shape [d])."""
    return np.mean(vectors, axis=0)


def compute_novelty(vectors: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """L2 distance of each vector to the anchor.

    vectors: [n, d]
    anchor:  [d]
    returns: [n]
    """
    anchor_2d = anchor.reshape(1, -1)
    return np.linalg.norm(vectors - anchor_2d, axis=1)


def fit_diversity_model(vectors: np.ndarray, n_clusters: int = 5, random_state: int = 42) -> KMeans:
    """Fit KMeans for diversity estimation."""
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    kmeans.fit(vectors)
    return kmeans


def compute_diversity_from_model(vectors: np.ndarray, kmeans: KMeans) -> float:
    """Compute entropy of cluster distribution using a fitted KMeans."""
    labels = kmeans.predict(vectors)
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return float(entropy(probs))