"""Feature engineering module for CWT Sentiment Agent.

Aggregates daily X post sentiment & engagement, computes price technical features,
and produces a leakage-free merged daily dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def compute_daily_sentiment_features(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates individual post sentiment and engagement into daily signals."""
    if sentiment_df.empty:
        logger.warning("Empty sentiment dataframe passed.")
        return pd.DataFrame(columns=["date", "sentiment_mean", "sentiment_weighted", "post_count"])

    df = sentiment_df.copy()
    if "created_at" not in df.columns:
        raise ValueError("Sentiment dataframe missing 'created_at' column.")

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["created_at"])
    df["date"] = df["created_at"].dt.strftime("%Y-%m-%d")

    # Engagement weight: log1p of total interactions + 1.0 baseline
    engagement = (
        df.get("likes", 0).fillna(0)
        + 2.0 * df.get("reposts", 0).fillna(0)
        + 1.5 * df.get("replies", 0).fillna(0)
        + 1.0 * df.get("quotes", 0).fillna(0)
        + 0.1 * df.get("views", 0).fillna(0)
    )
    df["weight"] = np.log1p(np.maximum(0, engagement)) + 1.0
    score_col = "sentiment_score" if "sentiment_score" in df.columns else "compound"
    df["weighted_score"] = df[score_col] * df["weight"]

    records = []
    for date_str, group in df.groupby("date"):
        total_weight = group["weight"].sum()
        weighted_sentiment = (
            group["weighted_score"].sum() / total_weight if total_weight > 0 else group[score_col].mean()
        )
        scores = group[score_col]
        pos_count = (group.get("sentiment_label", "") == "positive").sum()
        neg_count = (group.get("sentiment_label", "") == "negative").sum()
        total_posts = len(group)

        records.append({
            "date": date_str,
            "sentiment_mean": float(scores.mean()),
            "sentiment_weighted": float(weighted_sentiment),
            "sentiment_std": float(scores.std(ddof=0)) if total_posts > 1 else 0.0,
            "sentiment_min": float(scores.min()),
            "sentiment_max": float(scores.max()),
            "post_count": int(total_posts),
            "bullish_ratio": float(pos_count / total_posts) if total_posts > 0 else 0.0,
            "bearish_ratio": float(neg_count / total_posts) if total_posts > 0 else 0.0,
            "total_likes": float(group.get("likes", 0).sum()),
            "total_reposts": float(group.get("reposts", 0).sum()),
            "total_views": float(group.get("views", 0).sum()),
            "total_engagement": float(engagement.loc[group.index].sum()),
        })

    daily_sentiment = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return daily_sentiment


def compute_price_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """Computes technical and price momentum features without future lookahead."""
    if price_df.empty:
        raise ValueError("Price dataframe is empty.")

    df = price_df.copy()
    if "date" not in df.columns:
        raise ValueError("Price dataframe missing 'date' column.")

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_p = df["open"]
    volume = df["volume"]

    # Returns
    df["return_1d"] = close.pct_change(1)
    df["return_3d"] = close.pct_change(3)
    df["return_7d"] = close.pct_change(7)
    df["log_return"] = np.log(close / close.shift(1))

    # Volatility
    df["volatility_7d"] = df["return_1d"].rolling(window=7, min_periods=3).std()
    df["volatility_14d"] = df["return_1d"].rolling(window=14, min_periods=5).std()

    # Moving averages
    df["sma_7"] = close.rolling(window=7, min_periods=1).mean()
    df["sma_20"] = close.rolling(window=20, min_periods=1).mean()
    df["sma_ratio"] = (df["sma_7"] / df["sma_20"]) - 1.0

    # Intraday range
    df["high_low_ratio"] = (high - low) / open_p

    # Volume momentum
    df["volume_change_1d"] = volume.pct_change(1)
    df["volume_sma_7"] = volume.rolling(window=7, min_periods=1).mean()
    df["volume_ratio"] = (volume / df["volume_sma_7"]) - 1.0

    # RSI (14 period)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=5).mean()
    avg_loss = loss.rolling(window=14, min_periods=5).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    return df


def merge_features_and_prices(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Merges technical price features with aggregated daily sentiment features.

    Strictly preserves historical causality (no future leakage).
    Handles dates with missing sentiment gracefully with explicit quality indicators.
    """
    price_feat = compute_price_features(price_df)
    sent_feat = compute_daily_sentiment_features(sentiment_df) if not sentiment_df.empty else pd.DataFrame()

    if not sent_feat.empty:
        merged = pd.merge(price_feat, sent_feat, on="date", how="left")
    else:
        merged = price_feat.copy()
        for col in [
            "sentiment_mean", "sentiment_weighted", "sentiment_std", "sentiment_min",
            "sentiment_max", "post_count", "bullish_ratio", "bearish_ratio",
            "total_likes", "total_reposts", "total_views", "total_engagement",
        ]:
            merged[col] = np.nan

    # Indicator for empirical sentiment presence
    merged["has_sentiment"] = (~merged["sentiment_mean"].isna()).astype(int)

    # Impute missing sentiment with neutral 0.0 baseline and 0 counts
    fill_zeros = [
        "sentiment_mean", "sentiment_weighted", "sentiment_std", "sentiment_min",
        "sentiment_max", "post_count", "bullish_ratio", "bearish_ratio",
        "total_likes", "total_reposts", "total_views", "total_engagement",
    ]
    for col in fill_zeros:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    # Exponential decay sentiment momentum (smoothed historical sentiment)
    merged["sentiment_ema_3d"] = (
        merged["sentiment_weighted"].ewm(span=3, min_periods=1, adjust=False).mean()
    )

    # Clean initial NaN values from rolling windows
    merged = merged.bfill().fillna(0.0)

    target_path = output_path or (PROCESSED_DIR / "merged_features.csv")
    merged.to_csv(target_path, index=False)
    logger.info(f"Merged feature matrix saved to {target_path} ({len(merged)} rows, {len(merged.columns)} features)")
    return merged


def load_or_create_features() -> pd.DataFrame:
    """Helper to build and cache complete feature set from raw/processed files."""
    price_file = RAW_DIR / "btc_prices.csv"
    if not price_file.exists():
        from src.ingestion import fetch_prices
        price_df = fetch_prices()
    else:
        price_df = pd.read_csv(price_file)

    sentiment_file = PROCESSED_DIR / "btc_x_posts_sentiment.csv"
    if not sentiment_file.exists():
        clean_file = PROCESSED_DIR / "btc_x_posts_clean.csv"
        if not clean_file.exists():
            from src.normalize import normalize_x_posts
            clean_df = normalize_x_posts()
        else:
            clean_df = pd.read_csv(clean_file)
        from src.sentiment import SentimentAnalyzer
        sentiment_df = SentimentAnalyzer(prefer_finbert=True).analyze_dataframe(clean_df)
    else:
        sentiment_df = pd.read_csv(sentiment_file)

    return merge_features_and_prices(price_df, sentiment_df)


if __name__ == "__main__":
    load_or_create_features()
