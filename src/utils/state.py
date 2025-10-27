from dataclasses import dataclass
from typing import Any, Dict, Optional
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load


@dataclass
class TrainingState:
    anchor: np.ndarray
    kmeans: KMeans
    scaler_stats: Dict[str, Dict[str, float]]
    weight_model: Any
    text_vectorizer: Optional[TfidfVectorizer] = None


def save_state(state: TrainingState, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(state, path)


def load_state(path: str) -> TrainingState:
    return load(path)

