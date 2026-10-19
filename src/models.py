"""Predictive modeling and chronological walk-forward validation module.

Supports multi-horizon forecasting (n+0 to n+5), naive baseline comparison,
leakage-free walk-forward validation, and financial backtest metrics
(Sharpe, Sortino, Maximum Drawdown).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

from src.config import PROCESSED_DIR, ROOT

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    # Technical & price momentum
    "return_1d",
    "return_3d",
    "return_7d",
    "volatility_7d",
    "volatility_14d",
    "sma_ratio",
    "high_low_ratio",
    "volume_change_1d",
    "volume_ratio",
    "rsi_14",
    # Sentiment signals
    "sentiment_mean",
    "sentiment_weighted",
    "sentiment_std",
    "sentiment_ema_3d",
    "bullish_ratio",
    "bearish_ratio",
    "post_count",
    "total_engagement",
    "has_sentiment",
]

HORIZONS = [0, 1, 2, 3, 4, 5]


def prepare_modeling_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Prepares feature matrix and multi-horizon target returns without forward leakage.

    Horizons:
      h=0: Intraday/current day return ((close - open) / open)
      h=1..5: Cumulative forward return ((close[t+h] - close[t]) / close[t])
    """
    data = df.sort_values("date").copy().reset_index(drop=True)

    # Validate feature presence
    present_features = [col for col in FEATURE_COLUMNS if col in data.columns]
    if len(present_features) < 5:
        raise ValueError(f"Insufficient features in dataset. Found: {present_features}")

    # Horizon targets
    # n+0: today's return (close vs open)
    data["target_h0"] = (data["close"] - data["open"]) / data["open"]

    # n+1 to n+5: forward return relative to today's close
    for h in [1, 2, 3, 4, 5]:
        data[f"target_h{h}"] = (data["close"].shift(-h) - data["close"]) / data["close"]
        data[f"future_close_h{h}"] = data["close"].shift(-h)

    data["target_price_h0"] = data["close"]
    for h in [1, 2, 3, 4, 5]:
        data[f"target_price_h{h}"] = data[f"future_close_h{h}"]

    return data, present_features


@dataclass
class HorizonMetrics:
    horizon: int
    model_mae: float
    model_rmse: float
    model_dir_acc: float
    naive_mae: float
    naive_rmse: float
    naive_dir_acc: float
    num_eval_samples: int


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray) -> Dict[str, float]:
    """Calculates MAE, RMSE, and Directional Accuracy for model and naive baseline."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isnan(y_naive)
    if not np.any(mask):
        return {
            "model_mae": 0.0, "model_rmse": 0.0, "model_dir_acc": 0.5,
            "naive_mae": 0.0, "naive_rmse": 0.0, "naive_dir_acc": 0.5,
            "samples": 0,
        }

    yt = y_true[mask]
    yp = y_pred[mask]
    yn = y_naive[mask]

    model_mae = float(np.mean(np.abs(yt - yp)))
    model_rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    # Directional accuracy: sign agreement (positive return predicted when actual is positive)
    model_dir_acc = float(np.mean((np.sign(yp) == np.sign(yt))))

    naive_mae = float(np.mean(np.abs(yt - yn)))
    naive_rmse = float(np.sqrt(np.mean((yt - yn) ** 2)))
    naive_dir_acc = float(np.mean((np.sign(yn) == np.sign(yt))))

    return {
        "model_mae": model_mae,
        "model_rmse": model_rmse,
        "model_dir_acc": model_dir_acc,
        "naive_mae": naive_mae,
        "naive_rmse": naive_rmse,
        "naive_dir_acc": naive_dir_acc,
        "samples": int(len(yt)),
    }


def walk_forward_validation(
    df: pd.DataFrame,
    features: List[str],
    horizons: List[int] = HORIZONS,
    min_train_size: int = 60,
    step_size: int = 1,
) -> Tuple[Dict[int, HorizonMetrics], pd.DataFrame]:
    """Performs strictly chronological walk-forward out-of-sample validation.

    Ensures zero forward data leakage:
      - Scaler fit ONLY on the training split of each step.
      - Training targets at step t only include samples strictly <= t - h.
      - Out-of-sample predictions are aggregated for full audit.
    """
    n_rows = len(df)
    if n_rows < min_train_size + 10:
        logger.warning(
            f"Dataset length ({n_rows}) is small for min_train_size={min_train_size}. Adjusting min_train_size."
        )
        min_train_size = max(20, n_rows // 2)

    oos_records = []

    # Model definition
    def create_model():
        return XGBRegressor(
            n_estimators=60,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
        )

    logger.info(
        f"Starting walk-forward validation on {n_rows} days (initial train: {min_train_size}, step: {step_size})..."
    )

    for current_idx in range(min_train_size, n_rows, step_size):
        train_slice = df.iloc[:current_idx]
        eval_row = df.iloc[current_idx]
        eval_date = eval_row["date"]
        eval_close = eval_row["close"]

        step_record = {
            "date": eval_date,
            "close": eval_close,
        }

        for h in horizons:
            target_col = f"target_h{h}"

            # Ensure training target has no lookahead:
            # When predicting at step current_idx for horizon h,
            # training data can only include observations where target_h{h} was fully realized:
            valid_train = train_slice.iloc[: len(train_slice) - h] if h > 0 else train_slice
            valid_train = valid_train.dropna(subset=[target_col] + features)

            if len(valid_train) < 15:
                continue

            X_train = valid_train[features].values
            y_train = valid_train[target_col].values

            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            model = create_model()
            model.fit(X_train_scaled, y_train)

            X_eval = eval_row[features].values.reshape(1, -1)
            X_eval_scaled = scaler.transform(X_eval)

            pred_return = float(model.predict(X_eval_scaled)[0])
            actual_return = eval_row[target_col] if target_col in eval_row and not pd.isna(eval_row[target_col]) else np.nan

            # Naive baseline: 0.0 return persistence (Random Walk price)
            naive_return = 0.0

            step_record[f"pred_ret_h{h}"] = pred_return
            step_record[f"actual_ret_h{h}"] = actual_return
            step_record[f"naive_ret_h{h}"] = naive_return
            step_record[f"pred_price_h{h}"] = eval_close * (1.0 + pred_return)
            step_record[f"actual_price_h{h}"] = (
                eval_close * (1.0 + actual_return) if not np.isnan(actual_return) else np.nan
            )
            step_record[f"naive_price_h{h}"] = eval_close

        oos_records.append(step_record)

    oos_df = pd.DataFrame(oos_records)

    # Compute metrics per horizon
    horizon_metrics: Dict[int, HorizonMetrics] = {}
    for h in horizons:
        y_true = oos_df[f"actual_ret_h{h}"].values
        y_pred = oos_df[f"pred_ret_h{h}"].values
        y_naive = oos_df[f"naive_ret_h{h}"].values

        res = calculate_metrics(y_true, y_pred, y_naive)
        horizon_metrics[h] = HorizonMetrics(
            horizon=h,
            model_mae=res["model_mae"],
            model_rmse=res["model_rmse"],
            model_dir_acc=res["model_dir_acc"],
            naive_mae=res["naive_mae"],
            naive_rmse=res["naive_rmse"],
            naive_dir_acc=res["naive_dir_acc"],
            num_eval_samples=res["samples"],
        )

    # Save OOS predictions to reports
    oos_df.to_csv(REPORTS_DIR / "walk_forward_oos_predictions.csv", index=False)
    return horizon_metrics, oos_df


def calculate_financial_performance(
    oos_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    """Computes Sharpe ratio, Sortino ratio, and Maximum Drawdown from out-of-sample predictions.

    Trading rule: Long BTC when out-of-sample predicted 1-day return > 0.0; otherwise Cash.
    Compares against Buy & Hold BTC benchmark over the exact same period.
    """
    if oos_df.empty or "actual_ret_h1" not in oos_df.columns or "pred_ret_h1" not in oos_df.columns:
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "strategy_total_return": 0.0,
            "benchmark_sharpe": 0.0,
            "benchmark_sortino": 0.0,
            "benchmark_mdd": 0.0,
            "benchmark_total_return": 0.0,
            "trades_count": 0,
            "equity_df": pd.DataFrame(),
        }

    clean_df = oos_df.dropna(subset=["actual_ret_h1", "pred_ret_h1"]).copy()
    if len(clean_df) < 5:
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "strategy_total_return": 0.0,
            "benchmark_sharpe": 0.0,
            "benchmark_sortino": 0.0,
            "benchmark_mdd": 0.0,
            "benchmark_total_return": 0.0,
            "trades_count": 0,
            "equity_df": pd.DataFrame(),
        }

    # Signal on day t dictates return on day t+1
    clean_df["signal"] = (clean_df["pred_ret_h1"] > 0.0).astype(float)
    clean_df["btc_return"] = clean_df["actual_ret_h1"]
    clean_df["strategy_return"] = clean_df["signal"] * clean_df["btc_return"]

    # Equity curves (start at 1.0)
    clean_df["strategy_equity"] = (1.0 + clean_df["strategy_return"]).cumprod()
    clean_df["benchmark_equity"] = (1.0 + clean_df["btc_return"]).cumprod()

    # Ratios
    def get_metrics_for_series(returns: pd.Series, equity: pd.Series) -> Tuple[float, float, float]:
        mean_r = float(returns.mean())
        std_r = float(returns.std())
        annualized_return = mean_r * 365.0
        annualized_vol = std_r * np.sqrt(365.0) if std_r > 1e-8 else 1e-8

        sharpe = (annualized_return - risk_free_rate) / annualized_vol

        downside_returns = returns[returns < 0.0]
        downside_std = float(downside_returns.std()) if len(downside_returns) > 1 else std_r
        annualized_downside_vol = downside_std * np.sqrt(365.0) if downside_std > 1e-8 else 1e-8
        sortino = (annualized_return - risk_free_rate) / annualized_downside_vol

        # Max Drawdown
        running_max = equity.cummax()
        drawdowns = (running_max - equity) / running_max
        max_dd = float(drawdowns.max())

        return sharpe, sortino, max_dd

    strat_sharpe, strat_sortino, strat_mdd = get_metrics_for_series(
        clean_df["strategy_return"], clean_df["strategy_equity"]
    )
    bench_sharpe, bench_sortino, bench_mdd = get_metrics_for_series(
        clean_df["btc_return"], clean_df["benchmark_equity"]
    )

    strat_tot_ret = float(clean_df["strategy_equity"].iloc[-1] - 1.0)
    bench_tot_ret = float(clean_df["benchmark_equity"].iloc[-1] - 1.0)

    results = {
        "sharpe_ratio": strat_sharpe,
        "sortino_ratio": strat_sortino,
        "max_drawdown": strat_mdd,
        "strategy_total_return": strat_tot_ret,
        "benchmark_sharpe": bench_sharpe,
        "benchmark_sortino": bench_sortino,
        "benchmark_mdd": bench_mdd,
        "benchmark_total_return": bench_tot_ret,
        "trades_count": int(clean_df["signal"].sum()),
        "eval_days": len(clean_df),
        "equity_df": clean_df[["date", "strategy_equity", "benchmark_equity", "signal"]],
    }

    return results


def train_production_models_and_predict_matrix(
    df: pd.DataFrame,
    features: List[str],
    horizons: List[int] = HORIZONS,
) -> Dict[str, Any]:
    """Trains final models using all available historical data and generates live n+0 through n+5 forecast matrix."""
    latest_row = df.iloc[-1]
    latest_date = latest_row["date"]
    latest_close = float(latest_row["close"])

    forecast_matrix = []

    for h in horizons:
        target_col = f"target_h{h}"
        valid_train = df.iloc[: len(df) - h] if h > 0 else df
        valid_train = valid_train.dropna(subset=[target_col] + features)

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(valid_train[features].values)
        y_train = valid_train[target_col].values

        model = XGBRegressor(
            n_estimators=75,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)

        X_latest = latest_row[features].values.reshape(1, -1)
        X_latest_scaled = scaler.transform(X_latest)
        pred_return = float(model.predict(X_latest_scaled)[0])
        pred_price = latest_close * (1.0 + pred_return)

        # Feature importances
        importances = dict(zip(features, [float(x) for x in model.feature_importances_]))

        # Uncertainty estimate based on training residuals
        train_preds = model.predict(X_train_scaled)
        residual_std = float(np.std(y_train - train_preds))
        lower_bound = latest_close * (1.0 + pred_return - 1.96 * residual_std)
        upper_bound = latest_close * (1.0 + pred_return + 1.96 * residual_std)

        forecast_matrix.append({
            "horizon": f"n+{h}",
            "horizon_days": h,
            "predicted_return_pct": pred_return * 100.0,
            "predicted_price": pred_price,
            "price_lower_95": lower_bound,
            "price_upper_95": upper_bound,
            "signal": "BULLISH" if pred_return > 0.005 else ("BEARISH" if pred_return < -0.005 else "NEUTRAL"),
            "residual_volatility": residual_std,
            "top_features": sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5],
        })

    return {
        "base_date": latest_date,
        "base_close": latest_close,
        "forecast_matrix": forecast_matrix,
    }


def run_modeling_pipeline() -> Dict[str, Any]:
    """Runs complete modeling pipeline: dataset prep, walk-forward CV, financial metrics, and live forecasts."""
    from src.features import load_or_create_features

    raw_features = load_or_create_features()
    model_df, feature_cols = prepare_modeling_dataset(raw_features)

    # Walk-forward validation
    metrics_by_h, oos_df = walk_forward_validation(model_df, feature_cols)

    # Financial evaluation
    financial_results = calculate_financial_performance(oos_df)

    # Live forecasts
    live_forecasts = train_production_models_and_predict_matrix(model_df, feature_cols)

    summary = {
        "metrics_by_horizon": {h: vars(m) for h, m in metrics_by_h.items()},
        "financial_performance": {
            k: v for k, v in financial_results.items() if k != "equity_df"
        },
        "live_forecasts": live_forecasts,
    }

    import json
    report_file = REPORTS_DIR / "modeling_summary.json"
    report_file.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(f"Complete modeling summary saved to {report_file}")

    return {
        "metrics_by_horizon": metrics_by_h,
        "oos_df": oos_df,
        "financial_results": financial_results,
        "live_forecasts": live_forecasts,
        "features_used": feature_cols,
    }


if __name__ == "__main__":
    run_modeling_pipeline()
