"""Sentiment analysis module implementing FinBERT with automated VADER fallback.

Computes sentiment scores (compound/polarity, positive, negative, neutral)
for cryptocurrency news and social media posts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from src.config import PROCESSED_DIR

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class VADERSentimentScorer:
    """Fast, lexicon-based sentiment analysis using VADER with crypto market enhancements."""

    CRYPTO_LEXICON = {
        "bull": 2.0,
        "bullish": 2.5,
        "rally": 2.5,
        "moon": 2.5,
        "breakout": 2.0,
        "ath": 2.5,
        "highs": 1.5,
        "surge": 2.0,
        "pump": 1.8,
        "bear": -2.0,
        "bearish": -2.5,
        "dump": -2.5,
        "crash": -3.0,
        "collapse": -3.0,
        "scam": -3.0,
        "rugpull": -3.5,
        "plunge": -2.5,
        "dip": -1.0,
        "buy": 1.5,
        "sell": -1.5,
        "profit": 2.0,
        "loss": -2.0,
    }

    def __init__(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
            self.analyzer.lexicon.update(self.CRYPTO_LEXICON)
            self.available = True
        except Exception as exc:
            logger.warning(f"Failed to initialize VADER: {exc}")
            self.analyzer = None
            self.available = False

    def score_text(self, text: str) -> Dict[str, Any]:
        if not self.available or not self.analyzer or not isinstance(text, str) or not text.strip():
            return {
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "label": "neutral",
                "model_used": "vader_default",
            }

        scores = self.analyzer.polarity_scores(text)
        compound = float(scores.get("compound", 0.0))
        pos = float(scores.get("pos", 0.0))
        neg = float(scores.get("neg", 0.0))
        neu = float(scores.get("neu", 1.0))

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return {
            "compound": compound,
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "label": label,
            "model_used": "vader",
        }


class FinBERTSentimentScorer:
    """Domain-specific financial sentiment analysis using ProsusAI/finbert."""

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.available = False
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info(f"Loading FinBERT model '{self.model_name}' on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.available = True
            logger.info("FinBERT successfully loaded.")
        except Exception as exc:
            logger.warning(f"FinBERT could not be initialized ({exc}). Will fall back to VADER.")
            self.tokenizer = None
            self.model = None
            self.available = False

    def score_text(self, text: str) -> Dict[str, Any]:
        if not self.available or not self.model or not self.tokenizer:
            raise RuntimeError("FinBERT model is not available.")

        if not isinstance(text, str) or not text.strip():
            return {
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "label": "neutral",
                "model_used": "finbert",
            }

        import torch

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)

        # Inspect model id2label mapping for reliability
        id2label = getattr(self.model.config, "id2label", {0: "positive", 1: "negative", 2: "neutral"})
        prob_dict = {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}

        pos = prob_dict.get("positive", 0.0)
        neg = prob_dict.get("negative", 0.0)
        neu = prob_dict.get("neutral", 0.0)

        # Continuous polarity metric bounded in [-1.0, 1.0]
        compound = float(pos - neg)

        labels = ["positive", "negative", "neutral"]
        max_label = max(labels, key=lambda lbl: prob_dict.get(lbl, 0.0))

        return {
            "compound": compound,
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "label": max_label,
            "model_used": "finbert",
        }


class SentimentAnalyzer:
    """Unified sentiment analyzer that coordinates FinBERT with automated VADER fallback."""

    def __init__(self, prefer_finbert: bool = True) -> None:
        self.vader = VADERSentimentScorer()
        self.finbert = None
        if prefer_finbert:
            self.finbert = FinBERTSentimentScorer()

    @property
    def is_finbert_active(self) -> bool:
        return self.finbert is not None and self.finbert.available

    def analyze_text(self, text: str) -> Dict[str, Any]:
        if self.is_finbert_active:
            try:
                return self.finbert.score_text(text)
            except Exception as exc:
                logger.debug(f"FinBERT scoring failed on text snippet ({exc}), falling back to VADER.")
        return self.vader.score_text(text)

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        output_file: Optional[str] = None,
    ) -> pd.DataFrame:
        if df.empty or text_column not in df.columns:
            logger.warning(f"Dataframe is empty or missing '{text_column}' column.")
            return df

        results: List[Dict[str, Any]] = []
        for text in df[text_column].fillna("").astype(str):
            results.append(self.analyze_text(text))

        res_df = pd.DataFrame(results)
        annotated_df = df.copy()
        annotated_df["sentiment_score"] = res_df["compound"]
        annotated_df["sentiment_pos"] = res_df["positive"]
        annotated_df["sentiment_neg"] = res_df["negative"]
        annotated_df["sentiment_neu"] = res_df["neutral"]
        annotated_df["sentiment_label"] = res_df["label"]
        annotated_df["sentiment_model"] = res_df["model_used"]

        out_path = PROCESSED_DIR / (output_file or "btc_x_posts_sentiment.csv")
        annotated_df.to_csv(out_path, index=False)
        logger.info(f"Saved sentiment-scored dataset ({len(annotated_df)} rows) to {out_path}")

        return annotated_df


def score_clean_posts() -> pd.DataFrame:
    """Loads cleaned posts, scores sentiment, and saves processed results."""
    clean_path = PROCESSED_DIR / "btc_x_posts_clean.csv"
    if not clean_path.exists():
        from src.normalize import normalize_x_posts
        df = normalize_x_posts()
    else:
        df = pd.read_csv(clean_path)

    analyzer = SentimentAnalyzer(prefer_finbert=True)
    return analyzer.analyze_dataframe(df, text_column="text")


if __name__ == "__main__":
    score_clean_posts()
