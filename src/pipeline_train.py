import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from data_loading import load_posts, validate_schema, zscore_metadata
from embedding import l2_normalize
from novelty_diversity import compute_anchor, compute_novelty, fit_diversity_model
from regression_fit import fit_weights
from utils.state import TrainingState, save_state


def build_text_embeddings(texts: pd.Series) -> np.ndarray:
    vectorizer = TfidfVectorizer(max_features=4096)
    X = vectorizer.fit_transform(texts.fillna("")).astype(np.float32)
    return vectorizer, X.toarray()


def split_by_time(df: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values("ts")
    split = int(len(df_sorted) * train_ratio)
    return df_sorted.iloc[:split], df_sorted.iloc[split:]


def main(input_csv: str, state_path: str, method: str = "ols", n_clusters: int = 10) -> None:
    df = load_posts(input_csv)
    missing = validate_schema(df)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 基本派生
    if "engagement_total" not in df.columns or df["engagement_total"].isna().all():
        df["engagement_total"] = df[["likes", "comments", "shares"]].fillna(0).sum(axis=1)

    # 文字嵌入（僅 text，ocr_text 可之後合併）
    vectorizer, X_text = build_text_embeddings(df["text"])
    X_text = l2_normalize(X_text)

    # 時序切分
    train_df, test_df = split_by_time(df)
    split = len(train_df)
    X_train = X_text[:split]
    X_test = X_text[split:]

    # 訓練期 anchor/kmeans
    anchor = compute_anchor(X_train)
    kmeans = fit_diversity_model(X_train, n_clusters=n_clusters, random_state=42)

    # novelty/diversity 作為特徵
    novelty_train = compute_novelty(X_train, anchor)
    novelty_test = compute_novelty(X_test, anchor)

    # 將 diversity 先以整體熵替代，每筆複製（之後可換每貼文版本）
    from novelty_diversity import compute_diversity_from_model

    div_train_value = compute_diversity_from_model(X_train, kmeans)
    div_test_value = compute_diversity_from_model(X_test, kmeans)
    diversity_train = np.full_like(novelty_train, div_train_value)
    diversity_test = np.full_like(novelty_test, div_test_value)

    # 標籤
    y = df["engagement_total"].to_numpy()
    y_train, _ = y[:split], y[split:]

    # 訓練權重模型
    weight_model = fit_weights(novelty_train, diversity_train, y_train, method=method)

    # 保存訓練狀態
    state = TrainingState(
        anchor=anchor,
        kmeans=kmeans,
        scaler_stats={},
        weight_model=weight_model,
        text_vectorizer=vectorizer,
    )
    save_state(state, state_path)
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    print(f"Saved TrainingState -> {state_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to posts.csv")
    parser.add_argument("--state", default="experiments/models/training_state.joblib")
    parser.add_argument("--method", default="ols", choices=["ols", "bayes"]) 
    parser.add_argument("--n_clusters", type=int, default=10)
    args = parser.parse_args()

    main(args.input, args.state, method=args.method, n_clusters=args.n_clusters)
