def compute_ati(novelty: np.ndarray, diversity: float, wN: float, wD: float) -> np.ndarray:
    """ATI = 100 * (wN * novelty + wD * diversity)"""
    return 100 * (wN * novelty + wD * diversity)
