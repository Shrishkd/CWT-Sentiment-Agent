
import json
import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR


def normalize_x_posts():
    source = RAW_DIR / "btc_x_posts.json"
    output = PROCESSED_DIR / "btc_x_posts_clean.csv"

    if not source.exists():
        raise FileNotFoundError(f"Missing source file: {source}")

    with source.open("r", encoding="utf-8") as f:
        posts = json.load(f)

    if not posts:
        raise ValueError("The X posts dataset is empty.")

    rows = []

    for post in posts:
        author = post.get("author") or {}

        rows.append({
            "post_id": post.get("id"),
            "created_at": post.get("createdAt"),
            "text": post.get("fullText") or post.get("text") or "",
            "likes": post.get("likeCount", 0),
            "reposts": post.get("retweetCount", 0),
            "replies": post.get("replyCount", 0),
            "quotes": post.get("quoteCount", 0),
            "views": post.get("viewCount", 0),
            "followers": author.get("followers", 0),
            "language": post.get("lang"),
            "url": post.get("url") or post.get("twitterUrl"),
        })

    df = pd.DataFrame(rows)

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        utc=True,
        format="mixed",
        errors="coerce",
    )

    numeric_cols = [
        "likes", "reposts", "replies",
        "quotes", "views", "followers",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        ).fillna(0)

    df = df.dropna(subset=["created_at"])
    df = df.drop_duplicates(subset=["post_id"])
    df = df.sort_values("created_at").reset_index(drop=True)

    df.to_csv(output, index=False)

    print(f"Normalized posts: {len(df)}")
    print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    print(f"Saved: {output}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nSample preview ready.")
    try:
        sample_str = df[["post_id", "created_at", "likes", "reposts"]].head(3).to_string(index=False)
        print(sample_str)
    except Exception:
        pass

    return df


if __name__ == "__main__":
    normalize_x_posts()
