import json

import pandas as pd
import streamlit as st

from src.config import ASSET, LOOKBACK_DAYS, RAW_DIR
from src.ingestion import (
    fetch_news,
    fetch_prices,
    fetch_x_posts,
    ingestion_status,
)

st.set_page_config(
    page_title="CWT Sentiment Agent",
    page_icon="📈",
    layout="wide",
)

st.title("CrowdWisdomTrading")
st.subheader("X Sentiment & BTC Price Prediction Agent")

st.caption(
    "Phase 1: Data ingestion | "
    f"Asset: {ASSET} | Lookback: {LOOKBACK_DAYS} days"
)

def get_secret(key, default=""):
    try:
        return str(st.secrets.get(key, default))
    except Exception:
        return default


apify_token = get_secret("APIFY_API_TOKEN")
tavily_key = get_secret("TAVILY_API_KEY")
actor_id = get_secret("APIFY_ACTOR_ID")

with st.sidebar:
    st.header("Data configuration")
    st.write(f"Asset: `{ASSET}`")
    st.write(f"Lookback: `{LOOKBACK_DAYS} days`")
    st.write(f"Apify token: {'Configured' if apify_token and 'YOUR_' not in apify_token else 'Missing'}")
    st.write(f"Tavily key: {'Configured' if tavily_key and 'YOUR_' not in tavily_key else 'Missing'}")
    st.write(f"Actor ID: `{actor_id}`")

    max_items = st.number_input(
        "Maximum X posts",
        min_value=10,
        max_value=2000,
        value=500,
        step=100,
    )

st.header("1. Collect data")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Fetch BTC prices", use_container_width=True):
        with st.spinner("Downloading historical prices..."):
            try:
                prices = fetch_prices()
                st.success(f"Saved {len(prices)} price rows.")
            except Exception as exc:
                st.error(f"Price ingestion failed: {exc}")

with col2:
    if st.button("Fetch recent news", use_container_width=True):
        if not tavily_key or "YOUR_" in tavily_key:
            st.error("Configure TAVILY_API_KEY in secrets.toml first.")
        else:
            with st.spinner("Searching Bitcoin news..."):
                try:
                    news = fetch_news(tavily_key)
                    st.success(f"Saved {len(news)} news results.")
                except Exception as exc:
                    st.error(f"News ingestion failed: {exc}")

with col3:
    if st.button("Scrape X posts", use_container_width=True):
        if not apify_token or "YOUR_" in apify_token:
            st.error("Configure APIFY_API_TOKEN first.")
        elif not actor_id or "YOUR_" in actor_id:
            st.error("Configure APIFY_ACTOR_ID first.")
        else:
            with st.spinner("Running the Apify Actor..."):
                try:
                    posts = fetch_x_posts(
                        apify_token,
                        actor_id,
                        int(max_items),
                    )
                    st.success(f"Saved {len(posts)} X posts.")
                except Exception as exc:
                    st.error(f"X ingestion failed: {exc}")

st.divider()
st.header("2. Ingestion status")

status = ingestion_status()
status_df = pd.DataFrame(
    [
        {
            "Dataset": name,
            "Available": details["exists"],
            "Last modified (UTC)": details["modified_utc"] or "Not collected",
        }
        for name, details in status.items()
    ]
)
st.dataframe(status_df, use_container_width=True, hide_index=True)

st.divider()
st.header("3. Latest collected data")

price_file = RAW_DIR / "btc_prices.csv"
news_file = RAW_DIR / "btc_news.json"
posts_file = RAW_DIR / "btc_x_posts.json"

if price_file.exists():
    prices = pd.read_csv(price_file)
    st.subheader("BTC-USD historical prices")
    st.line_chart(prices.set_index("date")["close"])
    st.dataframe(prices.tail(10), use_container_width=True, hide_index=True)

if news_file.exists():
    with st.expander("View collected news"):
        news = json.loads(news_file.read_text(encoding="utf-8"))
        for item in news:
            st.markdown(f"**{item.get('title', 'Untitled')}**")
            st.write(item.get("content", item.get("snippet", "")))
            if item.get("url"):
                st.markdown(f"[Read source]({item['url']})")

if posts_file.exists():
    with st.expander("Inspect raw X posts"):
        posts = json.loads(posts_file.read_text(encoding="utf-8"))
        st.write(f"Total stored posts: {len(posts)}")
        st.json(posts[:3])

st.info(
    "No forecasts are generated yet. First verify price history, "
    "news results, and X post fields. Modeling comes after data validation."
)