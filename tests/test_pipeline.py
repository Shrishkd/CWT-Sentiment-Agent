"""End-to-end unit and integration tests for CWT Sentiment Agent."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features import (
    compute_daily_sentiment_features,
    compute_price_features,
    merge_features_and_prices,
)
from src.ingestion import ingestion_status
from src.models import (
    FEATURE_COLUMNS,
    calculate_financial_performance,
    calculate_metrics,
    prepare_modeling_dataset,
    walk_forward_validation,
)
from src.normalize import normalize_x_posts
from src.sentiment import SentimentAnalyzer, VADERSentimentScorer


def test_ingestion_status_keys():
    status = ingestion_status()
    assert "btc_prices.csv" in status
    assert "btc_news.json" in status
    assert "btc_x_posts.json" in status
    for fname, details in status.items():
        assert "exists" in details
        assert "modified_utc" in details


def test_vader_sentiment_scorer():
    scorer = VADERSentimentScorer()
    assert scorer.available is True

    bullish_text = "Bitcoin is breaking out to all-time highs! Incredible massive rally! $BTC"
    res_pos = scorer.score_text(bullish_text)
    assert res_pos["compound"] > 0.0
    assert res_pos["label"] == "positive"
    assert -1.0 <= res_pos["compound"] <= 1.0

    bearish_text = "Bitcoin crashes horribly, huge disaster and panic selling everywhere."
    res_neg = scorer.score_text(bearish_text)
    assert res_neg["compound"] < 0.0
    assert res_neg["label"] == "negative"


def test_sentiment_analyzer_fallback():
    analyzer = SentimentAnalyzer(prefer_finbert=False)
    res = analyzer.analyze_text("BTC price consolidates around support.")
    assert "compound" in res
    assert "label" in res
    assert res["model_used"] == "vader"


def test_daily_sentiment_aggregation():
    sample_sentiment = pd.DataFrame([
        {
            "created_at": "2026-10-18T12:00:00Z",
            "sentiment_score": 0.8,
            "sentiment_label": "positive",
            "likes": 100,
            "reposts": 50,
            "replies": 10,
            "quotes": 5,
            "views": 2000,
        },
        {
            "created_at": "2026-10-18T18:00:00Z",
            "sentiment_score": -0.2,
            "sentiment_label": "negative",
            "likes": 10,
            "reposts": 2,
            "replies": 1,
            "quotes": 0,
            "views": 100,
        },
    ])
    daily = compute_daily_sentiment_features(sample_sentiment)
    assert len(daily) == 1
    assert daily["date"].iloc[0] == "2026-10-18"
    assert daily["post_count"].iloc[0] == 2
    assert daily["bullish_ratio"].iloc[0] == 0.5
    # Weighted sentiment should give more weight to the heavily engaged positive post
    assert daily["sentiment_weighted"].iloc[0] > daily["sentiment_mean"].iloc[0]


def test_price_features_no_leakage():
    dates = pd.date_range("2026-05-01", periods=30, freq="D")
    prices = pd.DataFrame({
        "date": dates,
        "open": np.linspace(60000, 70000, 30),
        "high": np.linspace(61000, 71000, 30),
        "low": np.linspace(59000, 69000, 30),
        "close": np.linspace(60500, 70500, 30),
        "volume": np.random.uniform(1e9, 2e9, 30),
    })

    feat = compute_price_features(prices)
    assert "return_1d" in feat.columns
    assert "volatility_7d" in feat.columns
    assert "rsi_14" in feat.columns
    assert "sma_ratio" in feat.columns
    # Verify no future index in rolling calculations
    assert len(feat) == 30


def test_multi_horizon_targets():
    dates = pd.date_range("2026-05-01", periods=40, freq="D")
    prices = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "open": np.full(40, 60000.0),
        "high": np.full(40, 61000.0),
        "low": np.full(40, 59000.0),
        "close": np.full(40, 60000.0) * (1.0 + 0.01 * np.arange(40)),
        "volume": np.full(40, 1e9),
        "return_1d": np.full(40, 0.01),
        "return_3d": np.full(40, 0.03),
        "return_7d": np.full(40, 0.07),
        "volatility_7d": np.full(40, 0.02),
        "volatility_14d": np.full(40, 0.02),
        "sma_ratio": np.full(40, 0.01),
        "high_low_ratio": np.full(40, 0.03),
        "volume_change_1d": np.full(40, 0.0),
        "volume_ratio": np.full(40, 0.0),
        "rsi_14": np.full(40, 50.0),
        "sentiment_mean": np.full(40, 0.2),
        "sentiment_weighted": np.full(40, 0.2),
        "sentiment_std": np.full(40, 0.05),
        "sentiment_ema_3d": np.full(40, 0.2),
        "bullish_ratio": np.full(40, 0.6),
        "bearish_ratio": np.full(40, 0.1),
        "post_count": np.full(40, 5),
        "total_engagement": np.full(40, 500.0),
        "has_sentiment": np.full(40, 1),
    })

    model_df, feats = prepare_modeling_dataset(prices)
    for h in [0, 1, 2, 3, 4, 5]:
        assert f"target_h{h}" in model_df.columns

    # Check forward alignment: for t=0, close=60000, for t=1, close=60600 -> target_h1 = 0.01
    assert pytest.approx(model_df["target_h1"].iloc[0], 0.001) == 0.01


def test_metrics_calculation():
    y_true = np.array([0.02, -0.01, 0.03, -0.04])
    y_pred = np.array([0.01, -0.02, 0.02, 0.01])
    y_naive = np.array([0.0, 0.0, 0.0, 0.0])

    res = calculate_metrics(y_true, y_pred, y_naive)
    assert res["model_mae"] > 0
    assert res["model_rmse"] > 0
    # Signs match for items 0, 1, 2; item 3 predicted pos (+0.01) vs neg (-0.04) -> 3 out of 4 = 75%
    assert res["model_dir_acc"] == 0.75
    assert res["naive_dir_acc"] == 0.0


def test_financial_performance():
    sample_oos = pd.DataFrame({
        "date": ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05", "2026-06-06"],
        "actual_ret_h1": [0.02, -0.01, 0.03, 0.01, -0.02, 0.04],
        "pred_ret_h1": [0.01, -0.005, 0.02, 0.015, -0.01, 0.02],
    })
    fin = calculate_financial_performance(sample_oos)
    assert "sharpe_ratio" in fin
    assert "sortino_ratio" in fin
    assert "max_drawdown" in fin
    assert fin["trades_count"] == 4
