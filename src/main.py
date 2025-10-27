# main.py
import numpy as np
from sklearn.metrics import r2_score

from embedding import l2_normalize
from novelty_diversity import (
    compute_anchor,
    compute_novelty,
    fit_diversity_model,
    compute_diversity_from_model,
)
from regression_fit import fit_weights
from predict import predict
from utils.state import TrainingState, save_state, load_state


if __name__ == "__main__":
    np.random.seed(42)

    # 模擬 200 筆「貼文」的向量 (假裝是 CLIP 輸出)
    fake_vectors = np.random.randn(200, 128)
    fake_vectors = l2_normalize(fake_vectors)

    # 時序切分 70/30
    n = fake_vectors.shape[0]
    split = int(0.7 * n)
    train_vecs = fake_vectors[:split]
    test_vecs = fake_vectors[split:]

    # 訓練期：估計 anchor 與 diversity 模型
    anchor = compute_anchor(train_vecs)
    kmeans = fit_diversity_model(train_vecs, n_clusters=10, random_state=42)

    # 計算 novelty 與 diversity（訓練與測試）
    novelty_train = compute_novelty(train_vecs, anchor)
    novelty_test = compute_novelty(test_vecs, anchor)

    div_train_value = compute_diversity_from_model(train_vecs, kmeans)
    div_test_value = compute_diversity_from_model(test_vecs, kmeans)
    diversity_train = np.full_like(novelty_train, div_train_value)
    diversity_test = np.full_like(novelty_test, div_test_value)

    # 假的 engagement label（讓模型可學到權重）
    y = 10 + 2 * np.concatenate([novelty_train, novelty_test]) + 5 * np.concatenate([diversity_train, diversity_test]) + np.random.randn(n)
    y_train, y_test = y[:split], y[split:]

    # 訓練權重模型（可切換 'ols' 或 'bayes'）
    weight_model = fit_weights(novelty_train, diversity_train, y_train, method="ols")

    # 保存訓練狀態（示範）
    state = TrainingState(anchor=anchor, kmeans=kmeans, scaler_stats={}, weight_model=weight_model)
    save_state(state, "experiments/models/training_state.joblib")

    # 載入並預測（示範）
    state2 = load_state("experiments/models/training_state.joblib")
    y_pred = predict(state2.weight_model, novelty_test, diversity_test)

    print("回歸係數:", getattr(state2.weight_model, "coef_", None))
    print("R^2 on test:", r2_score(y_test, y_pred))
    print("前 5 筆預測 vs 真值:")
    for yp, yt in list(zip(y_pred, y_test))[:5]:
        print(f"pred={yp:.2f}, true={yt:.2f}")