# CrowdWisdomTrading (CWT): BTC Sentiment & Multi-Horizon Forecast Agent

An autonomous cryptocurrency market intelligence system that integrates social media sentiment from X (Twitter) and financial news with historical market data to generate leakage-free multi-horizon Bitcoin (BTC-USD) price and return forecasts.

---

## Architecture Overview

```
                      ┌──────────────────────┐
                      │ Yahoo Finance API    │─── 180-Day BTC-USD Daily OHLCV
                      └──────────────────────┘
                                 │
                      ┌──────────────────────┐
                      │ Apify Twitter Scraper│─── Targeted Search Query Slices
                      └──────────────────────┘    (since:YYYY-MM-DD until:YYYY-MM-DD)
                                 │
                      ┌──────────────────────┐
                      │  Tavily News Search  │─── Crypto Market Sentiment & Articles
                      └──────────────────────┘
                                 │
                                 ▼
         ┌────────────────────────────────────────────────────────┐
         │             src/normalize.py (Normalization)           │
         │  Standardize UTC timestamps, deduplicate post IDs,    │
         │  clean social engagement metrics (likes, reposts, etc.)│
         └────────────────────────────────────────────────────────┘
                                 │
                                 ▼
         ┌────────────────────────────────────────────────────────┐
         │             src/sentiment.py (FinBERT + VADER)         │
         │  Primary: ProsusAI/finbert (domain-adapted Transformer)│
         │  Fallback: Enhanced VADER with crypto market lexicon   │
         │  Outputs: Polarity [-1, 1], pos/neg/neu probabilities  │
         └────────────────────────────────────────────────────────┘
                                 │
                                 ▼
         ┌────────────────────────────────────────────────────────┐
         │             src/features.py (Zero-Leakage Features)    │
         │  • Daily Engagement-Weighted Sentiment Signals        │
         │  • Technical Indicators: 1d/3d/7d Returns, 7d/14d Vol, │
         │    SMA_7, SMA_20, SMA Ratio, RSI-14, Volume Ratio     │
         │  • Strict causality: all row t features strictly <= t  │
         └────────────────────────────────────────────────────────┘
                                 │
                                 ▼
         ┌────────────────────────────────────────────────────────┐
         │             src/models.py (Multi-Horizon Forecasting)  │
         │  • Horizons n+0 (intraday) through n+5 (forward days)  │
         │  • Regularized Gradient Boosting (XGBoost)             │
         │  • Chronological Walk-Forward CV (Expanding Window)    │
         │  • Naive Random Walk Benchmark Comparison              │
         │  • Financial Simulation: Sharpe, Sortino, Max Drawdown │
         └────────────────────────────────────────────────────────┘
                                 │
                                 ▼
         ┌────────────────────────────────────────────────────────┐
         │             Streamlit Web Dashboard (app.py)           │
         │  Forecast Matrix, Sentiment Intelligence, OOS Audits,  │
         │  Interactive Equity Curves, and Data Quality Warnings  │
         └────────────────────────────────────────────────────────┘
```

---

## Key Capabilities

1. **Multi-Horizon Forecast Matrix ($n+0$ to $n+5$)**:
   - $n+0$: Intraday close vs. open directional return.
   - $n+1$ through $n+5$: Forward multi-day cumulative returns and price targets with 95% uncertainty confidence intervals.
2. **FinBERT NLP with Automated VADER Fallback**:
   - Uses `ProsusAI/finbert` for domain-specific financial sentiment analysis.
   - Automatically falls back to an enhanced `VADERSentimentScorer` containing crypto trading lexicon (`bullish`, `bearish`, `rally`, `breakout`, `crash`, `rugpull`) if offline or resource-constrained.
3. **Engagement-Weighted Sentiment Aggregation**:
   - Computes daily social sentiment weighted by virality ($1 + \ln(1 + \text{likes} + 2\cdot\text{reposts} + 1.5\cdot\text{replies} + 0.1\cdot\text{views})$).
4. **Strict Zero-Leakage Walk-Forward Validation**:
   - Expanding-window evaluation across 119 out-of-sample days.
   - Preprocessing scaling (`RobustScaler`) is strictly fit within each temporal training fold.
   - Training targets only include fully realized horizons $\le t - h$.
5. **Out-of-Sample Financial Backtesting**:
   - Evaluates a realistic long/cash trading strategy driven exclusively by forward predictions.
   - Computes Annualized Sharpe, Sortino, and Maximum Drawdown vs. BTC Buy & Hold.

---

## Walk-Forward Validation Results (Out-of-Sample Audit)

*Evaluated over 119 chronological walk-forward steps:*

| Horizon | Model MAE | Naive Baseline MAE | Model RMSE | Naive Baseline RMSE | Model Directional Acc | Naive Directional Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **n+0** (Today) | **0.0047** | 0.0132 | **0.0081** | 0.0195 | **95.8%** | 0.0% |
| **n+1** (1-Day Ahead) | 0.0139 | 0.0132 | 0.0205 | 0.0196 | **54.2%** | 0.0% |
| **n+2** (2-Days Ahead) | 0.0214 | 0.0195 | 0.0311 | 0.0289 | **48.7%** | 0.0% |
| **n+3** (3-Days Ahead) | 0.0294 | 0.0244 | 0.0419 | 0.0373 | **38.8%** | 0.0% |
| **n+4** (4-Days Ahead) | 0.0334 | 0.0283 | 0.0506 | 0.0451 | **40.9%** | 0.0% |
| **n+5** (5-Days Ahead) | 0.0389 | 0.0321 | 0.0595 | 0.0526 | **43.0%** | 0.0% |

---

## Out-of-Sample Financial Performance

| Metric | Sentiment-Guided Strategy | BTC Buy & Hold Benchmark | Description |
| :--- | :---: | :---: | :--- |
| **Annualized Sharpe Ratio** | **1.56** | 2.29 | Risk-adjusted return over annual volatility |
| **Annualized Sortino Ratio** | **1.78** | 4.72 | Return penalizing only downside volatility |
| **Maximum Drawdown (MDD)** | **9.18%** | 6.96% | Maximum peak-to-trough capital decline |
| **Strategy Total Return** | **+11.12%** | +28.86% | Cumulative return across 52 active trades |
| **Evaluation Period** | 118 days | 118 days | Out-of-sample walk-forward test horizon |

---

## Data Quality & Regulatory Disclosures

> [!WARNING]
> **Apify Actor Free-Tier Limitation**:
> 1. **Scraping Cap**: The Apify Twitter/X Actor (`apidojo/tweet-scraper` / Actor ID `61RPP7dywgiy0JPD0`) enforces a strict limitation of **10 items per run** on Apify's Free Plan.
> 2. **Actor Input Schema**: Top-level `start` and `end` JSON keys are not accepted by the actor. Date filtering requires inline Twitter advanced search operators (`since:YYYY-MM-DD until:YYYY-MM-DD`) inside `searchTerms`.
> 3. **Accumulation Architecture**: Each execution of `fetch_x_posts` preserves existing posts and deduplicates incoming posts by unique ID, progressively growing the dataset across runs without data loss.
> 4. **No Synthetic Data**: To adhere strictly to integrity rules, missing sentiment dates are marked with `has_sentiment = 0` and neutral baseline values ($0.0$). Synthetic posts are never fabricated.

---

## Setup & Execution

### 1. Environment Setup

```powershell
# Create virtual environment
python -m venv .venv

# Activate environment (PowerShell)
.venv\Scripts\Activate.ps1

# Install all project dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials

Ensure `.streamlit/secrets.toml` contains your API tokens (never commit this file):

```toml
APIFY_API_TOKEN = "your_apify_token_here"
TAVILY_API_KEY = "your_tavily_key_here"
APIFY_ACTOR_ID = "61RPP7dywgiy0JPD0"
```

### 3. Run Pipeline Components

```powershell
# Ingest prices, news, and social data
python -m src.ingestion

# Clean and normalize raw X posts
python -m src.normalize

# Run FinBERT & VADER sentiment scoring
python -m src.sentiment

# Generate technical indicators & merge features
python -m src.features

# Execute walk-forward validation and produce forecasts
python -m src.models
```

### 4. Launch the Streamlit Dashboard

```powershell
streamlit run app.py
```
*Or directly via venv:*
```powershell
.venv\Scripts\streamlit.exe run app.py
```

### 5. Run Test Suite

```powershell
pytest tests/ -v
```

---

## Project Structure

```
CWT-Sentiment-Agent/
├── .streamlit/
│   └── secrets.toml              # API credentials (git-ignored)
├── data/
│   ├── raw/                      # Raw downloaded prices, news, and X posts
│   └── processed/                # Normalized, sentiment-scored, and merged features
├── models/                       # Model artifacts and feature configs
├── reports/
│   ├── modeling_summary.json     # Walk-forward metrics, financial KPIs, and forecasts
│   └── walk_forward_oos_predictions.csv  # Granular out-of-sample prediction records
├── src/
│   ├── __init__.py
│   ├── config.py                 # Project paths, asset symbols, lookback windows
│   ├── ingestion.py              # Ingestion from Yahoo Finance, Apify, and Tavily
│   ├── normalize.py              # Text cleaning and datetime normalization
│   ├── sentiment.py              # FinBERT model & VADER sentiment scoring
│   ├── features.py               # Engagement-weighted aggregation & technical indicators
│   └── models.py                 # Multi-horizon modeling, walk-forward CV, and backtesting
├── tests/
│   └── test_pipeline.py          # Unit & integration tests for all components
├── app.py                        # Streamlit interactive intelligence dashboard
├── requirements.txt              # Frozen dependency specifications (UTF-8)
├── .gitignore                    # Secrets, caches, and dataset exclusion rules
└── README.md                     # Comprehensive project documentation
```
