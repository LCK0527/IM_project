from typing import Any, Literal

import numpy as np
from joblib import dump, load
from sklearn.linear_model import LinearRegression, BayesianRidge


def fit_weights(
    novelty: np.ndarray,
    diversity: np.ndarray,
    y: np.ndarray,
    method: Literal["ols", "bayes"] = "ols",
) -> Any:
    """Fit a regression model to learn weights from novelty/diversity.

    Returns the fitted model.
    """
    X = np.vstack([novelty, diversity]).T
    if method == "ols":
        model = LinearRegression()
    elif method == "bayes":
        model = BayesianRidge()
    else:
        raise ValueError(f"Unknown method: {method}")
    model.fit(X, y)
    return model


def save_model(model: Any, path: str) -> None:
    dump(model, path)


def load_model(path: str) -> Any:
    return load(path)
