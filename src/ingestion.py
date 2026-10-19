import json
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from apify_client import ApifyClient
from tavily import TavilyClient

from src.config import (
    ASSET,
    LOOKBACK_DAYS,
    RAW_DIR,
    START_DATE,
    END_DATE,
)


def fetch_prices():
    df = yf.download(
        ASSET,
        period=f"{LOOKBACK_DAYS}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise RuntimeError("Yahoo Finance returned no BTC-USD price data.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    if "date" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})

    if "date" not in df.columns:
        raise RuntimeError(f"Unexpected price columns: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").drop_duplicates("date")

    output = RAW_DIR / "btc_prices.csv"
    df.to_csv(output, index=False)

    return df


def fetch_news(api_key):
    if not api_key:
        raise ValueError("TAVILY_API_KEY is missing.")

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query="Bitcoin BTC market news price outlook",
        topic="news",
        days=7,
        max_results=10,
        search_depth="basic",
    )

    results = response.get("results", [])

    output = RAW_DIR / "btc_news.json"
    output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return results


def fetch_x_posts(api_token, actor_id, max_items=500):
    if not api_token:
        raise ValueError("APIFY_API_TOKEN is missing.")

    if not actor_id:
        raise ValueError("APIFY_ACTOR_ID is missing.")

    client = ApifyClient(api_token)

    actor_input = {
        "searchTerms": [
            "$BTC",
            "Bitcoin",
            "BTC price",
        ],
        "start": START_DATE.isoformat(),
        "end": END_DATE.isoformat(),
        "maxItems": max_items,
        "sort": "Latest",
        "tweetLanguage": "en",
    }

    run = client.actor(actor_id).call(
        run_input=actor_input,
        timeout_secs=300,
    )

    if not run:
        raise RuntimeError("The Apify Actor did not return a completed run.")

    dataset_id = run.get("defaultDatasetId")
    if not dataset_id:
        raise RuntimeError("Apify run has no default dataset.")

    items = list(
        client.dataset(dataset_id).iterate_items()
    )

    output = RAW_DIR / "btc_x_posts.json"
    output.write_text(
        json.dumps(items, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return items


def ingestion_status():
    files = [
        RAW_DIR / "btc_prices.csv",
        RAW_DIR / "btc_news.json",
        RAW_DIR / "btc_x_posts.json",
    ]

    return {
        path.name: {
            "exists": path.exists(),
            "modified_utc": (
                datetime.fromtimestamp(
                    path.stat().st_mtime,
                    tz=timezone.utc,
                ).isoformat()
                if path.exists()
                else None
            ),
        }
        for path in files
    }