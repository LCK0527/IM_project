import numpy as np


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2 normalize each row vector.

    vectors: [n, d] float array
    returns: [n, d] float array
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    return vectors / norms