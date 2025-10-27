from __future__ import annotations

from typing import Dict, Tuple, List, Set
import pandas as pd

# 標準欄位（Google Sheets / CSV 模板）
TEMPLATE_COLUMNS: List[str] = [
    "post_id",
    "platform",
    "ts",
    "brand_id",
    "brand",
    "store_id",
    "url",
    "text",
    "ocr_text",
    "image_urls",
    "likes",
    "comments",
    "shares",
    "is_lottery",
    "engagement_total",
]

# 不做 z-score 的欄位（識別、文字、標籤等）
NON_FEATURE_COLUMNS: Set[str] = set(TEMPLATE_COLUMNS)


def load_posts(path: str) -> pd.DataFrame:
    """Load posts table from CSV using the standard schema.

    Expected columns (order not required):
    post_id, platform, ts, brand_id, brand, store_id, url, text, ocr_text,
    image_urls, likes, comments, shares, is_lottery, engagement_total
    """
    if path.endswith(".csv"):
        df = pd.read_csv(path, encoding="utf-8")
        return df
    raise NotImplementedError("Only CSV loading is implemented for now.")


def validate_schema(df: pd.DataFrame) -> List[str]:
    """Return a list of missing required columns according to TEMPLATE_COLUMNS."""
    missing = [c for c in TEMPLATE_COLUMNS if c not in df.columns]
    return missing


def zscore_metadata(
    df: pd.DataFrame,
    fit: bool,
    stats: Dict[str, Dict[str, float]] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Z-score numeric metadata columns. Returns transformed df and stats.

    - 排除 NON_FEATURE_COLUMNS（如識別欄、文字欄、engagement 與衍生標籤）
    - 僅對數值欄位進行標準化
    """
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]

    if fit:
        stats = {
            c: {"mean": float(df[c].mean()), "std": float(df[c].std() or 1.0)} for c in feature_cols
        }

    if not stats:
        stats = {c: {"mean": 0.0, "std": 1.0} for c in feature_cols}

    out = df.copy()
    for c in feature_cols:
        mean = stats[c]["mean"]
        std = stats[c]["std"] or 1.0
        out[c] = (out[c] - mean) / std

    return out, stats

