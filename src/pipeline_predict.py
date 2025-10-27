import os
from typing import Tuple

import numpy as np
import pandas as pd

from data_loading import load_posts, validate_schema
from embedding import l2_normalize
from novelty_diversity import compute_novelty, compute_diversity_from_model
from predict import predict as predict_with_model
from utils.state import load_state
from ati_compute import compute_ati


def transform_text(vectorizer, texts: pd.Series) -> np.ndarray:
    X = vectorizer.transform(texts.fillna("")).astype(np.float32)
    return X.toarray()


def main(input_csv: str, state_path: str, output_csv: str) -> None:
    df = load_posts(input_csv)
    missing = validate_schema(df)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    state = load_state(state_path)

    # 文字轉向量（沿用訓練期向量化器）
    X_text = transform_text(state.text_vectorizer, df["text"])  # type: ignore
    X_text = l2_normalize(X_text)

    # novelty / diversity 使用訓練期 anchor / kmeans
    novelty = compute_novelty(X_text, state.anchor)
    div_value = compute_diversity_from_model(X_text, state.kmeans)
    diversity = np.full_like(novelty, div_value)

    # engagement 預測與 ATI
    y_pred = predict_with_model(state.weight_model, novelty, diversity)
    ati = compute_ati(novelty, diversity, wN=1.0, wD=1.0)  # 權重可替換為 model.coef_

    out = df.copy()
    out["novelty"] = novelty
    out["diversity"] = diversity
    out["ATI"] = ati
    out["engagement_pred"] = y_pred

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved predictions -> {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to posts.csv")
    parser.add_argument("--state", default="experiments/models/training_state.joblib")
    parser.add_argument("--output", default="experiments/reports/predictions.csv")
    args = parser.parse_args()

    main(args.input, args.state, args.output)
