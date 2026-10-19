"""CrowdWisdomTrading (CWT) Sentiment Agent Dashboard.

Interactive web application displaying:
1. Ingestion & Pipeline Orchestration
2. Forecast Matrix (n+0 through n+5 horizons)
3. Sentiment Outlook & Market Intelligence (FinBERT & VADER)
4. Chronological Walk-Forward Validation Audit (vs Naive Baseline)
5. Portfolio Financial Backtest (Sharpe, Sortino, Max Drawdown)
6. Data Quality & Regulatory Disclosures
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import ASSET, LOOKBACK_DAYS, PROCESSED_DIR, RAW_DIR, ROOT
from src.features import load_or_create_features
from src.ingestion import fetch_news, fetch_prices, fetch_x_posts, ingestion_status
from src.models import run_modeling_pipeline
from src.normalize import normalize_x_posts
from src.sentiment import SentimentAnalyzer

# App configuration
st.set_page_config(
    page_title="CWT Sentiment Agent | BTC Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

REPORTS_DIR = ROOT / "reports"


def get_secret(key: str, default: str = "") -> str:
    try:
        val = str(st.secrets.get(key, default))
        return val if val and "YOUR_" not in val else default
    except Exception:
        return default


apify_token = get_secret("APIFY_API_TOKEN")
tavily_key = get_secret("TAVILY_API_KEY")
actor_id = get_secret("APIFY_ACTOR_ID", "61RPP7dywgiy0JPD0")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/cryptocurrency.png", width=64)
    st.title("CWT Sentiment Agent")
    st.caption("AI-Powered Sentiment & Price Prediction Engine")

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Controls")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_all = st.button("🚀 Run Pipeline", width="stretch")
    with col_btn2:
        fetch_fresh_prices = st.button("🔄 Sync Prices", width="stretch")

    st.markdown("---")
    st.markdown("### 📡 API Credentials")
    st.write(f"• **Asset**: `{ASSET}`")
    st.write(f"• **Lookback**: `{LOOKBACK_DAYS} days`")
    st.write(f"• **Apify API**: {'🟢 Active' if apify_token else '🔴 Missing'}")
    st.write(f"• **Tavily API**: {'🟢 Active' if tavily_key else '🔴 Missing'}")
    st.write(f"• **Apify Actor**: `{actor_id}`")

    max_items_input = st.number_input(
        "Max X scrape items",
        min_value=10,
        max_value=2000,
        value=500,
        step=50,
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size: 0.8rem; color: #888;'>"
        "CrowdWisdomTrading Agent v2.0<br/>"
        "FinBERT NLP + XGBoost Multi-Horizon Forecasting<br/>"
        "Strict temporal validation (Zero Leakage)</div>",
        unsafe_allow_html=True,
    )


# --- ORCHESTRATION TRIGGERS ---
if fetch_fresh_prices:
    with st.spinner("Fetching latest BTC-USD prices from Yahoo Finance..."):
        try:
            p_df = fetch_prices()
            st.toast(f"Successfully refreshed {len(p_df)} BTC prices!", icon="✅")
        except Exception as exc:
            st.sidebar.error(f"Price fetch failed: {exc}")

if run_all:
    with st.status("Executing End-to-End Sentiment & Modeling Pipeline...", expanded=True) as status_box:
        st.write("1/5 Ingesting market data...")
        fetch_prices()
        if tavily_key:
            try:
                fetch_news(tavily_key)
            except Exception as e:
                st.write(f"News fetch skipped ({e})")
        st.write("2/5 Normalizing X post datasets...")
        normalize_x_posts()
        st.write("3/5 Scoring sentiment via FinBERT (with VADER fallback)...")
        from src.sentiment import score_clean_posts
        score_clean_posts()
        st.write("4/5 Merging daily features and technical indicators...")
        load_or_create_features()
        st.write("5/5 Running multi-horizon walk-forward validation and models...")
        run_modeling_pipeline()
        status_box.update(label="Pipeline Execution Complete!", state="complete", expanded=False)
        st.rerun()


# --- HEADER & STATUS BANNER ---
st.title("CrowdWisdomTrading: BTC Sentiment & Multi-Horizon Forecast Agent")
st.markdown(
    "Automated intelligence system that ingests X (Twitter) social sentiment and market data, "
    "scores text with **FinBERT** (domain-adapted Transformer), merges leakage-free technical indicators, "
    "and evaluates multi-horizon forecasts ($n+0$ to $n+5$) via **chronological walk-forward validation**."
)

status = ingestion_status()
prices_exist = status["btc_prices.csv"]["exists"]
posts_exist = status["btc_x_posts.json"]["exists"]
news_exist = status["btc_news.json"]["exists"]
features_file = PROCESSED_DIR / "merged_features.csv"
features_exist = features_file.exists()
report_file = REPORTS_DIR / "modeling_summary.json"
models_exist = report_file.exists()

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
col_m1.metric("Historical Prices", f"{LOOKBACK_DAYS}d" if prices_exist else "Missing", "Yahoo Finance")
col_m2.metric("X Posts Collected", "Available" if posts_exist else "Missing", "Apify Scraper")
col_m3.metric("News Articles", "Available" if news_exist else "Missing", "Tavily API")
col_m4.metric("Sentiment Features", "Processed" if features_exist else "Pending", "FinBERT + VADER")
col_m5.metric("Walk-Forward Models", "Trained" if models_exist else "Pending", "XGBoost (n+0 to n+5)")

st.divider()

# Load modeling report if exists, or auto-run if features exist
modeling_data = {}
if report_file.exists():
    try:
        modeling_data = json.loads(report_file.read_text(encoding="utf-8"))
    except Exception:
        modeling_data = {}

if not modeling_data and features_exist:
    with st.spinner("Generating initial model predictions and walk-forward audit..."):
        run_modeling_pipeline()
        if report_file.exists():
            modeling_data = json.loads(report_file.read_text(encoding="utf-8"))


# --- MAIN TABS ---
tab_forecast, tab_sentiment, tab_validation, tab_portfolio, tab_data, tab_ingest = st.tabs([
    "🔮 Forecast Matrix (n+0 to n+5)",
    "🧠 Sentiment & Market Intelligence",
    "📊 Walk-Forward Validation Audit",
    "💼 Portfolio & Financial Backtest",
    "⚠️ Data Quality & Methodology",
    "📥 Ingestion Console",
])

# ==========================================
# TAB 1: FORECAST MATRIX
# ==========================================
with tab_forecast:
    st.header("Multi-Horizon Price & Return Forecasts")
    st.caption("Point predictions, confidence intervals, and directional signals strictly using features available up to today.")

    if modeling_data and "live_forecasts" in modeling_data:
        live = modeling_data["live_forecasts"]
        base_close = live.get("base_close", 0.0)
        base_date = live.get("base_date", "")
        forecasts = live.get("forecast_matrix", [])

        st.info(f"**Baseline Anchor**: Most recent close price is **${base_close:,.2f}** as of **{base_date}**.")

        # Metric cards for each horizon
        cols = st.columns(len(forecasts))
        for idx, fc in enumerate(forecasts):
            with cols[idx]:
                h_name = fc["horizon"]
                ret_pct = fc["predicted_return_pct"]
                price_pred = fc["predicted_price"]
                sig = fc["signal"]

                sig_color = "🟢" if sig == "BULLISH" else ("🔴" if sig == "BEARISH" else "⚪")
                delta_str = f"{ret_pct:+.2f}%"

                st.markdown(f"### {h_name}")
                st.metric(
                    label=f"Predicted Close",
                    value=f"${price_pred:,.0f}",
                    delta=delta_str,
                )
                st.markdown(f"**Signal**: {sig_color} `{sig}`")
                st.caption(f"95% CI: ${fc['price_lower_95']:,.0f} - ${fc['price_upper_95']:,.0f}")

        st.markdown("---")

        # Interactive Forecast Trajectory Chart
        col_chart, col_table = st.columns([2, 1])

        with col_chart:
            st.subheader("Price Projection Trajectory")

            # Load recent prices
            price_csv = RAW_DIR / "btc_prices.csv"
            if price_csv.exists():
                p_df = pd.read_csv(price_csv).tail(20)
                p_df["date"] = pd.to_datetime(p_df["date"]).dt.strftime("%Y-%m-%d")

                fig_proj = go.Figure()

                # Historical line
                fig_proj.add_trace(go.Scatter(
                    x=p_df["date"],
                    y=p_df["close"],
                    mode="lines+markers",
                    name="Historical Close",
                    line=dict(color="#1f77b4", width=3),
                ))

                # Build forecast coordinates
                fc_dates = [base_date]
                fc_prices = [base_close]
                fc_lowers = [base_close]
                fc_uppers = [base_close]

                base_dt = pd.to_datetime(base_date)
                for fc in forecasts:
                    if fc["horizon_days"] == 0:
                        continue
                    fut_dt = (base_dt + pd.Timedelta(days=fc["horizon_days"])).strftime("%Y-%m-%d")
                    fc_dates.append(fut_dt)
                    fc_prices.append(fc["predicted_price"])
                    fc_lowers.append(fc["price_lower_95"])
                    fc_uppers.append(fc["price_upper_95"])

                # Forecast line
                fig_proj.add_trace(go.Scatter(
                    x=fc_dates,
                    y=fc_prices,
                    mode="lines+markers",
                    name="Predicted Trajectory",
                    line=dict(color="#ff7f0e", width=3, dash="dash"),
                ))

                # Upper Bound
                fig_proj.add_trace(go.Scatter(
                    x=fc_dates,
                    y=fc_uppers,
                    mode="lines",
                    name="95% Upper Bound",
                    line=dict(width=0),
                    showlegend=False,
                ))

                # Lower Bound with shaded area
                fig_proj.add_trace(go.Scatter(
                    x=fc_dates,
                    y=fc_lowers,
                    mode="lines",
                    name="95% Confidence Band",
                    fill="tonexty",
                    fillcolor="rgba(255, 127, 14, 0.15)",
                    line=dict(width=0),
                ))

                fig_proj.update_layout(
                    title="BTC-USD 5-Day Forward Outlook",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode="x unified",
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=450,
                )
                st.plotly_chart(fig_proj, width="stretch")

        with col_table:
            st.subheader("Forecast Breakdown")
            fc_display_df = pd.DataFrame([
                {
                    "Horizon": fc["horizon"],
                    "Pred Return": f"{fc['predicted_return_pct']:+.2f}%",
                    "Target Price": f"${fc['predicted_price']:,.2f}",
                    "Lower Bound": f"${fc['price_lower_95']:,.2f}",
                    "Upper Bound": f"${fc['price_upper_95']:,.2f}",
                    "Signal": fc["signal"],
                }
                for fc in forecasts
            ])
            st.dataframe(fc_display_df, width="stretch", hide_index=True)

            with st.expander("🔍 Top Driver Features by Horizon"):
                for fc in forecasts:
                    st.markdown(f"**{fc['horizon']} Key Drivers:**")
                    top_feats = fc.get("top_features", [])
                    st.write(", ".join([f"`{f[0]}` ({f[1]:.1%})" for f in top_feats]))
    else:
        st.warning("No model forecasts available. Click '🚀 Run Pipeline' in the sidebar to generate.")


# ==========================================
# TAB 2: SENTIMENT & MARKET INTELLIGENCE
# ==========================================
with tab_sentiment:
    st.header("Social & Market Intelligence (FinBERT & VADER)")
    st.caption("Natural Language Processing applied to crypto community discourse and financial news.")

    sent_file = PROCESSED_DIR / "btc_x_posts_sentiment.csv"
    if sent_file.exists():
        s_df = pd.read_csv(sent_file)

        # Overview Metrics
        avg_score = float(s_df["sentiment_score"].mean())
        pos_ratio = float((s_df["sentiment_label"] == "positive").mean() * 100.0)
        neg_ratio = float((s_df["sentiment_label"] == "negative").mean() * 100.0)
        neu_ratio = float((s_df["sentiment_label"] == "neutral").mean() * 100.0)
        model_name = s_df["sentiment_model"].iloc[0] if "sentiment_model" in s_df.columns else "FinBERT"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Polarity Score", f"{avg_score:+.3f}", "Scale: -1.0 to +1.0")
        c2.metric("Bullish Share", f"{pos_ratio:.1f}%", f"{int(pos_ratio*len(s_df)/100)} posts")
        c3.metric("Bearish Share", f"{neg_ratio:.1f}%", f"{int(neg_ratio*len(s_df)/100)} posts")
        c4.metric("NLP Engine", f"{model_name.upper()}", "Domain Fine-Tuned")

        st.markdown("---")

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.subheader("Sentiment Distribution")
            fig_pie = px.pie(
                s_df,
                names="sentiment_label",
                color="sentiment_label",
                color_discrete_map={
                    "positive": "#2ca02c",
                    "neutral": "#7f7f7f",
                    "negative": "#d62728",
                },
                hole=0.45,
            )
            fig_pie.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig_pie, width="stretch")

        with col_s2:
            st.subheader("Engagement vs Polarity")
            s_df["engagement_sum"] = (
                s_df.get("likes", 0) + s_df.get("reposts", 0) + s_df.get("replies", 0)
            )
            fig_scatter = px.scatter(
                s_df,
                x="sentiment_score",
                y="engagement_sum",
                size=s_df["engagement_sum"].clip(lower=1),
                color="sentiment_label",
                color_discrete_map={
                    "positive": "#2ca02c",
                    "neutral": "#7f7f7f",
                    "negative": "#d62728",
                },
                hover_data=["text"],
                labels={"sentiment_score": "Sentiment Polarity", "engagement_sum": "Social Interactions"},
            )
            fig_scatter.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig_scatter, width="stretch")

        st.subheader("Inspected Posts & NLP Classifications")
        disp_posts = s_df[["created_at", "text", "sentiment_score", "sentiment_label", "likes", "reposts"]].copy()
        disp_posts["created_at"] = pd.to_datetime(disp_posts["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(disp_posts, width="stretch", hide_index=True)
    else:
        st.info("No sentiment data generated yet. Click '🚀 Run Pipeline' in the sidebar.")

    # News Section
    news_file = RAW_DIR / "btc_news.json"
    if news_file.exists():
        st.markdown("---")
        st.subheader("📰 Recent Financial News Coverage")
        news_items = json.loads(news_file.read_text(encoding="utf-8"))
        for item in news_items[:5]:
            title = item.get("title", "Market Update")
            snippet = item.get("content", item.get("snippet", ""))
            url = item.get("url", "#")
            st.markdown(f"**[{title}]({url})**")
            st.caption(snippet)


# ==========================================
# TAB 3: WALK-FORWARD VALIDATION AUDIT
# ==========================================
with tab_validation:
    st.header("Chronological Walk-Forward Validation")
    st.caption(
        "Strict out-of-sample evaluation across 119 historical days. Models are retrained sequentially "
        "without lookahead, comparing predictive error against a naive random-walk persistence baseline."
    )

    if modeling_data and "metrics_by_horizon" in modeling_data:
        metrics_dict = modeling_data["metrics_by_horizon"]

        # Audit Table
        audit_rows = []
        for h_str, m in metrics_dict.items():
            h_int = int(h_str)
            m_mae = m["model_mae"]
            n_mae = m["naive_mae"]
            mae_improvement = ((n_mae - m_mae) / n_mae * 100.0) if n_mae > 0 else 0.0

            m_rmse = m["model_rmse"]
            n_rmse = m["naive_rmse"]
            rmse_improvement = ((n_rmse - m_rmse) / n_rmse * 100.0) if n_rmse > 0 else 0.0

            m_dir = m["model_dir_acc"] * 100.0
            n_dir = m["naive_dir_acc"] * 100.0

            audit_rows.append({
                "Horizon": f"n+{h_int}",
                "Model MAE": f"{m_mae:.4f}",
                "Naive MAE": f"{n_mae:.4f}",
                "MAE Delta": f"{mae_improvement:+.1f}%",
                "Model RMSE": f"{m_rmse:.4f}",
                "Naive RMSE": f"{n_rmse:.4f}",
                "RMSE Delta": f"{rmse_improvement:+.1f}%",
                "Model Dir Acc": f"{m_dir:.1f}%",
                "Naive Dir Acc": f"{n_dir:.1f}%",
                "Samples": m["num_eval_samples"],
            })

        audit_df = pd.DataFrame(audit_rows)
        st.dataframe(audit_df, width="stretch", hide_index=True)

        st.markdown("---")

        col_v1, col_v2 = st.columns(2)

        with col_v1:
            st.subheader("Directional Accuracy by Horizon")
            h_labels = [f"n+{h}" for h in metrics_dict.keys()]
            m_dirs = [metrics_dict[h]["model_dir_acc"] * 100.0 for h in metrics_dict.keys()]

            fig_dir = go.Figure()
            fig_dir.add_trace(go.Bar(
                x=h_labels,
                y=m_dirs,
                name="Model Accuracy",
                marker_color="#2ca02c",
            ))
            fig_dir.add_hline(
                y=50.0,
                line_dash="dash",
                line_color="red",
                annotation_text="Random Guessing (50%)",
            )
            fig_dir.update_layout(
                yaxis_title="Directional Accuracy (%)",
                yaxis_range=[0, 105],
                template="plotly_white",
                height=350,
            )
            st.plotly_chart(fig_dir, width="stretch")

        with col_v2:
            st.subheader("Model MAE vs Naive Baseline")
            m_maes = [metrics_dict[h]["model_mae"] for h in metrics_dict.keys()]
            n_maes = [metrics_dict[h]["naive_mae"] for h in metrics_dict.keys()]

            fig_mae = go.Figure()
            fig_mae.add_trace(go.Bar(x=h_labels, y=m_maes, name="XGBoost Model MAE", marker_color="#1f77b4"))
            fig_mae.add_trace(go.Bar(x=h_labels, y=n_maes, name="Naive Baseline MAE", marker_color="#aec7e8"))
            fig_mae.update_layout(
                barmode="group",
                yaxis_title="Mean Absolute Error",
                template="plotly_white",
                height=350,
            )
            st.plotly_chart(fig_mae, width="stretch")

        # OOS Scatter Plot
        oos_file = REPORTS_DIR / "walk_forward_oos_predictions.csv"
        if oos_file.exists():
            st.markdown("---")
            st.subheader("Out-of-Sample Actual vs Predicted Returns (Horizon n+1)")
            oos_data = pd.read_csv(oos_file).dropna(subset=["actual_ret_h1", "pred_ret_h1"])
            fig_oos = px.scatter(
                oos_data,
                x="pred_ret_h1",
                y="actual_ret_h1",
                labels={"pred_ret_h1": "Out-of-Sample Predicted 1-Day Return", "actual_ret_h1": "Actual 1-Day Return"},
                trendline="ols",
                template="plotly_white",
            )
            fig_oos.update_layout(height=400)
            st.plotly_chart(fig_oos, width="stretch")
    else:
        st.info("Validation audit not yet generated. Click '🚀 Run Pipeline' in the sidebar.")


# ==========================================
# TAB 4: PORTFOLIO & FINANCIAL BACKTEST
# ==========================================
with tab_portfolio:
    st.header("Financial Performance & Out-of-Sample Backtest")
    st.caption(
        "Trading simulation driven purely by out-of-sample forward signals (Long when predicted 1-day return > 0; "
        "otherwise Cash). Compares risk-adjusted metrics against BTC Buy & Hold benchmark over the exact same period."
    )

    if modeling_data and "financial_performance" in modeling_data:
        fin = modeling_data["financial_performance"]

        c_f1, c_f2, c_f3, c_f4 = st.columns(4)
        c_f1.metric(
            "Sharpe Ratio (Annualized)",
            f"{fin.get('sharpe_ratio', 0.0):.2f}",
            f"Benchmark: {fin.get('benchmark_sharpe', 0.0):.2f}",
        )
        c_f2.metric(
            "Sortino Ratio (Annualized)",
            f"{fin.get('sortino_ratio', 0.0):.2f}",
            f"Benchmark: {fin.get('benchmark_sortino', 0.0):.2f}",
        )
        c_f3.metric(
            "Maximum Drawdown",
            f"{fin.get('max_drawdown', 0.0):.1%}",
            f"Benchmark: {fin.get('benchmark_mdd', 0.0):.1%}",
            delta_color="inverse",
        )
        c_f4.metric(
            "Strategy Total Return",
            f"{fin.get('strategy_total_return', 0.0):+.1%}",
            f"Benchmark: {fin.get('benchmark_total_return', 0.0):+.1%}",
        )

        st.markdown("---")

        # Equity Curve Chart
        oos_file = REPORTS_DIR / "walk_forward_oos_predictions.csv"
        if oos_file.exists():
            st.subheader("Out-of-Sample Cumulative Wealth Trajectory")
            oos_data = pd.read_csv(oos_file).dropna(subset=["actual_ret_h1", "pred_ret_h1"]).copy()
            oos_data["signal"] = (oos_data["pred_ret_h1"] > 0.0).astype(float)
            oos_data["strat_ret"] = oos_data["signal"] * oos_data["actual_ret_h1"]
            oos_data["strat_equity"] = (1.0 + oos_data["strat_ret"]).cumprod()
            oos_data["bench_equity"] = (1.0 + oos_data["actual_ret_h1"]).cumprod()

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=oos_data["date"],
                y=oos_data["strat_equity"],
                mode="lines",
                name="Sentiment-Guided Strategy",
                line=dict(color="#2ca02c", width=2.5),
            ))
            fig_eq.add_trace(go.Scatter(
                x=oos_data["date"],
                y=oos_data["bench_equity"],
                mode="lines",
                name="BTC Buy & Hold Benchmark",
                line=dict(color="#1f77b4", width=2, dash="dot"),
            ))
            fig_eq.update_layout(
                yaxis_title="Growth of $1.00",
                template="plotly_white",
                hovermode="x unified",
                height=420,
            )
            st.plotly_chart(fig_eq, width="stretch")
    else:
        st.info("Financial backtest metrics not yet available. Click '🚀 Run Pipeline'.")


# ==========================================
# TAB 5: DATA QUALITY & METHODOLOGY
# ==========================================
with tab_data:
    st.header("Data Quality, Integrity & Regulatory Disclosures")

    st.warning(
        r"⚠️ **Data Coverage & Free-Tier Limitation Notice**:" "\n\n"
        r"1. **Apify Scraping Cap**: The Apify Twitter/X Actor (`apidojo/tweet-scraper` / `61RPP7dywgiy0JPD0`) "
        r"enforces a strict limit of 10 items per run on the Free Plan. To respect the constraint 'never fabricate data', "
        r"days without empirical X posts are marked with `has_sentiment=0` and imputed with neutral sentiment (0.0)." "\n\n"
        r"2. **Accumulation Architecture**: Each scraping run preserves and appends non-duplicate historical tweets, "
        r"enabling progressive dataset expansion across repeated runs without data loss." "\n\n"
        r"3. **Zero-Leakage Guarantee**: Rolling technical indicators, sentiment exponentially weighted averages, "
        r"standardization scaling, and forward prediction targets are strictly calculated on causal history ($t \le \tau$). "
        r"No future data is ever observed during walk-forward training."
    )

    # Coverage stats
    feat_file = PROCESSED_DIR / "merged_features.csv"
    if feat_file.exists():
        f_df = pd.read_csv(feat_file)
        total_days = len(f_df)
        days_with_sent = int(f_df["has_sentiment"].sum())
        coverage_pct = (days_with_sent / total_days) * 100.0 if total_days > 0 else 0.0

        q1, q2, q3 = st.columns(3)
        q1.metric("Total Historical Days", f"{total_days} days", "Yahoo Finance (180d window)")
        q2.metric("Days with X Sentiment", f"{days_with_sent} days", f"{coverage_pct:.1f}% coverage")
        q3.metric("Imputed Neutral Days", f"{total_days - days_with_sent} days", "Flagged has_sentiment=0")

        st.subheader("Data Matrix Preview")
        st.dataframe(f_df.tail(10), width="stretch", hide_index=True)


# ==========================================
# TAB 6: INGESTION CONSOLE
# ==========================================
with tab_ingest:
    st.header("Raw Ingestion Console")
    st.caption("Inspect and trigger individual ingestion pipelines.")

    c_in1, c_in2, c_in3 = st.columns(3)

    with c_in1:
        st.markdown("### 1. BTC Historical Prices")
        if st.button("Fetch Yahoo Finance Prices", width="stretch"):
            with st.spinner("Downloading..."):
                try:
                    p = fetch_prices()
                    st.success(f"Saved {len(p)} daily price rows.")
                except Exception as exc:
                    st.error(f"Price ingestion failed: {exc}")

    with c_in2:
        st.markdown("### 2. Financial News")
        if st.button("Search Crypto News (Tavily)", width="stretch"):
            if not tavily_key:
                st.error("TAVILY_API_KEY is not configured.")
            else:
                with st.spinner("Searching news..."):
                    try:
                        n = fetch_news(tavily_key)
                        st.success(f"Saved {len(n)} news articles.")
                    except Exception as exc:
                        st.error(f"News ingestion failed: {exc}")

    with c_in3:
        st.markdown("### 3. Social Media (X)")
        if st.button("Scrape X (Apify)", width="stretch"):
            if not apify_token:
                st.error("APIFY_API_TOKEN is not configured.")
            else:
                with st.spinner("Executing Apify Actor..."):
                    try:
                        x = fetch_x_posts(apify_token, actor_id, int(max_items_input))
                        st.success(f"Saved {len(x)} total unique X posts.")
                    except Exception as exc:
                        st.error(f"X scraping failed: {exc}")

    st.markdown("---")
    st.subheader("Current Dataset Artifacts on Disk")
    st.dataframe(pd.DataFrame([
        {
            "Dataset File": name,
            "Exists": details["exists"],
            "Last Modified (UTC)": details["modified_utc"] or "Not Collected",
        }
        for name, details in ingestion_status().items()
    ]), width="stretch", hide_index=True)