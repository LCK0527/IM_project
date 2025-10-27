## 專案簡介
本專案提供一條以向量表徵為核心的內容評估管線，計算每筆貼文的「新穎度」（novelty）與整體「多樣性」（diversity），並以這兩個特徵學習權重、組合成 ATI 指標，用於預測/評估內容的風險或成效（例如 engagement）。目前包含可運行的 Demo，並預留資料載入、多模態嵌入、Bayesian 權重學習、與層級彙整等擴充點。

## 核心概念（簡述）
- **Novelty（新穎度）**: 以訓練資料的平均向量作為錨點，計算每筆向量到錨點的距離。
- **Diversity（多樣性）**: 對向量做 KMeans 分群後，計算群集標籤分佈的訊息熵（分佈越均勻，多樣性越高）。
- **ATI 指標**: \( ATI = 100 \times (w_N \cdot \text{Novelty} + w_D \cdot \text{Diversity}) \)。權重 \(w_N, w_D\) 可由訓練資料學得（目前提供 OLS，後續可換 Bayesian）。
- **訓練/預測切分**: 以時間序列排序後的前 70% 作為訓練，後 30% 作為測試（避免資料洩漏）。

## 目錄結構
```text
/Users/lck/workspace/ATI-Project/
  config/
    default.yaml
  data/
    processed/
    raw/
    sample/
  docs/
    DATASET.md
    MODELING.md
    PIPELINE.md
    README.md
    ROADMAP.md
  experiments/
    logs/
    models/
    reports/
  notebooks/
  src/
    ati_compute.py
    data_loading.py
    embedding.py
    main.py
    novelty_diversity.py
    predict.py
    regression_fit.py
    utils/
  tests/
    test_embedding.py
    test_novelty_diversity.py
  README.md
  requirements.txt
  setup.py
```

## 快速開始（Demo）
1) 建議使用 Python 3.10+ 並建立虛擬環境。
```bash
python -m venv .venv && source .venv/bin/activate
```

2) 安裝最小依賴（Demo 需要）。
```bash
pip install numpy scikit-learn scipy
```

3) 執行內建 Demo（會隨機生成向量，跑完 novelty/diversity → 訓練 → 預測 → 輸出 R^2）。
```bash
python /Users/lck/workspace/ATI-Project/src/main.py
```

## 訓練與預測管線（設計藍圖）
- **(1) 資料準備**: 收集一年貼文資料（文字、影像、metadata、engagement），按時間排序；前 70% 訓練、後 30% 測試。
- **(2) 多模態嵌入**: 使用 CLIP 生成文字/影像向量並 L2-normalize；metadata 做 z-score；多模態融合為貼文向量。
- **(3) ATI 計算（訓練期）**: 用訓練資料估計錨點與叢集，計算每貼文 novelty 與整體 diversity；學得 \(w_N, w_D\)（Bayesian 或 OLS）；組合為 ATI。
- **(4) 模型驗證**: 進行 posterior predictive checks（先以 OLS 殘差與擬合度檢查起步，後續可接 PyMC/ArviZ）。
- **(5) 預測（測試期）**: 以訓練期的錨點、叢集、標準化器，對測試貼文計算 novelty/diversity，輸出 ATI。
- **(6) 輸出**: 於貼文層級生成 ATI，並可彙整至門市與品牌層級。

> 註：目前 `src/main.py` 以隨機資料示範流程。後續會新增 `pipeline_train.py` / `pipeline_predict.py` 將訓練產物（錨點、KMeans、標準化器、權重模型）持久化並在預測期重用。

## 模組說明（src/）1.      Data Preparation

Collect one year of Facebook post data (text, image, metadata, engagement stats) in the bubble tea market, at the post level, to capture seasonality (brands promote differently across seasons).
Training is performed at the post-level, but results can later be aggregated to the store level and brand level.
Sort posts by time. Train on the earliest 70% of posts, and test on the latest 30%.
 

2.      Multimodal Embeddings

Compute embeddings for text and image using CLIP, with L2-normalization for both.
Rescale metadata (z-score).
Fuse modalities into a post-level vector.
- `embedding.py`: `l2_normalize` 向量的 L2 正規化。
- `novelty_diversity.py`: `compute_novelty`（與平均向量距離）、`compute_diversity`（KMeans 分群 + 熵）。
- `ati_compute.py`: `compute_ati`，以學得權重組合 novelty 與 diversity。
- `regression_fit.py`: 擬合線性迴歸模型（後續可切換 BayesianRidge 或 PyMC）。
- `predict.py`: 以訓練好的模型與特徵進行預測。
- `data_loading.py`: 預留資料載入與前處理介面（目前為雛形）。
- `main.py`: 可直接執行的 Demo，串接所有功能，用來快速驗證。

## 設定與參數
- `config/default.yaml`: 集中管理超參（叢集數、隨機種子、split 比例、模型類型、CLIP 模型名、資料路徑等）。目前為空白，待填。
- 建議之後以 `pyyaml` 讀取設定、`joblib` 持久化訓練產物（錨點、KMeans、標準化器、權重模型）。

## 測試與文件
- `tests/`: 預留單元測試檔，預計會先補 `novelty`、`diversity`、z-score、跟持久化/載入的測試。
- `docs/`: `PIPELINE.md`、`MODELING.md`、`DATASET.md`、`ROADMAP.md` 為方法、資料與規劃的補充文件（目前為空白，待填）。

## 後續工作（Roadmap 精簡版）
- 串接 CLIP（文字/影像）與 metadata z-score，完成多模態融合。
- 將訓練產物（平均向量、KMeans、標準化器、權重模型）持久化，並在預測期重用。
- 加入 Bayesian 權重學習：先以 `BayesianRidge` 起步，評估導入 PyMC/ArviZ 做完整後驗與 PPC。
- 實作貼文→門市→品牌的層級彙整與輸出 schema。
- 充實 `config/default.yaml`、`requirements.txt`，與最小可行的單元測試。

## 常見問題
- **是否需要真實資料才能開始？** 不需要。可先用合成資料（`main.py` 已示範）打通流程，等真實資料就定位後再替換資料來源與嵌入器。
- **最少安裝哪些套件即可跑 Demo？** `numpy`、`scikit-learn`、`scipy`。

## Interfaces（開發對齊）
- embedding
  - `l2_normalize(vectors: np.ndarray) -> np.ndarray`
- novelty_diversity
  - `compute_anchor(vectors: np.ndarray) -> np.ndarray`  # 訓練期估計平均向量錨點
  - `compute_novelty(vectors: np.ndarray, anchor: np.ndarray) -> np.ndarray`  # 測試期沿用訓練錨點
  - `fit_diversity_model(vectors: np.ndarray, n_clusters: int, random_state: int) -> KMeans`
  - `compute_diversity_from_model(vectors: np.ndarray, kmeans: KMeans) -> float`
- ati_compute
  - `compute_ati(novelty: np.ndarray, diversity: np.ndarray | float, wN: float, wD: float) -> np.ndarray`
- regression_fit
  - `fit_weights(novelty: np.ndarray, diversity: np.ndarray, y: np.ndarray, method: Literal['ols','bayes']='ols') -> Any`
  - `save_model(model: Any, path: str) -> None`
  - `load_model(path: str) -> Any`
- predict
  - `predict(model: Any, novelty: np.ndarray, diversity: np.ndarray) -> np.ndarray`
- data_loading
  - `load_posts(path: str) -> pd.DataFrame`  # post_id, store_id, brand_id, ts, text, image_path, metadata, engagement
  - `zscore_metadata(df: pd.DataFrame, fit: bool, stats: dict | None=None) -> tuple[pd.DataFrame, dict]`
- utils/state（新增檔案）
  - `TrainingState(anchor: np.ndarray, kmeans: KMeans, scaler_stats: dict, weight_model: Any)`  # dataclass
  - `save_state(state: TrainingState, path: str) -> None`
  - `load_state(path: str) -> TrainingState`

