from dataclasses import dataclass
from typing import Any, Dict, Optional
import os

import numpy as np
from sklearn.cluster import KMeans
from joblib import dump, load


@dataclass
class TrainingState:
    anchor: np.ndarray
    kmeans: KMeans
    scaler_stats: Dict[str, Dict[str, float]]
    weight_model: Any


def save_state(state: TrainingState, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(state, path)


def load_state(path: str) -> TrainingState:
    return load(path)

