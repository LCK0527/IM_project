import numpy as np
from typing import Any


def predict(model: Any, novelty: np.ndarray, diversity: np.ndarray) -> np.ndarray:
    X = np.vstack([novelty, diversity]).T
    return model.predict(X)
