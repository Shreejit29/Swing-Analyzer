"""
Market Sentiment Engine

Provides a lightweight, production-friendly sentiment layer for
stock analysis.

Design goals:
- No heavy transformer models
- Works well with Streamlit Cloud
- Robust handling of missing/invalid news
- Produces transparent sentiment metrics
- Can later be connected to historical backtesting
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import math
import re

import numpy as np
import pandas as pd


# =====================================================================
# SENTIMENT VOCABULARY
# =====================================================================

POSITIVE_WORDS = {
    "beat",
    "beats",
    "beaten",
    "bullish",
    "bullishness",
    "buy",
    "bought",
    "buying",
    "breakout",
    "breakouts",
    "growth",
    "growing",
    "gain",
    "gains",
    "gained",
    "profit",
    "profits",
    "profitable",
    "surge",
    "surges",
    "surged",
    "strong",
    "strength",
    "positive",
    "positively",
    "upgrade",
    "upgraded",
    "upside",
    "outperform",
    "outperformed",
    "outperformance",
    "record",
    "records",
    "recovery",
    "recover",
    "expansion",
    "expand",
    "expanding",
    "robust",
    "optimistic",
    "optimism",
    "improve",
    "improved",
    "improvement",
    "improving",
    "rally",
    "rallied",
    "rising",
    "rise",
    "raises",
    "raised",
    "dividend",
    "partnership",
    "contract",
    "order",
    "orders",
    "approval",
    "approved",
    "launch",
    "launched",
    "milestone",
    "innovation",
    "investment",
    "invest",
    "investing",
}

NEGATIVE_WORDS = {
    "bearish",
    "bearishness",
    "sell",
    "selling",
    "sold",
    "selloff",
    "breakdown",
    "breakdowns",
    "fall",
    "falls",
    "fell",
    "falling",
    "drop",
    "drops",
    "dropped",
    "decline",
    "declines",
    "declined",
    "loss",
    "losses",
    "lost",
    "weak",
    "weakness",
    "negative",
    "negatively",
    "downgrade",
    "downgraded",
    "downside",
    "underperform",
    "underperformed",
    "debt",
    "default",
    "lawsuit",
    "penalty",
    "fraud",
    "scandal",
    "investigation",
    "probe",
    "warning",
    "warns",
    "warning",
    "risk",
    "risks",
    "concern",
    "concerns",
    "concerned",
    "crisis",
    "crash",
    "recession",
    "inflation",
    "layoff",
    "layoffs",
    "cut",
    "cuts",
    "slump",
    "slumped",
    "plunge",
    "plunged",
    "plunging",
    "disappoint",
    "disappointed",
    "disappointing",
    "miss",
    "missed",
    "misses",
    "fraud",
    "fraudulent",
}


# Stronger terms receive larger weights.
STRONG_POSITIVE = {
    "breakout",
    "surge",
    "surged",
    "record",
    "outperformance",
    "approved",
    "milestone",
}

STRONG_NEGATIVE = {
    "crash",
    "plunge",
    "plunged",
    "fraud",
    "default",
    "scandal",
    "crisis",
}


# =====================================================================
# TEXT PROCESSING
# =====================================================================


def _clean_text(text: Any) -> str:
    """Normalize news text."""

    if text is None:
        return ""

    try:
        if pd.isna(text):
            return ""
    except Exception:
        pass

    text = str(text)

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-zA-Z0-9\s'-]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip().lower()


def _tokenize(text: str) -> List[str]:
    """Tokenize cleaned text."""

    return re.findall(
        r"[a-zA-Z]+(?:'[a-zA-Z]+)?",
        text.lower(),
    )


# =====================================================================
# SENTIMENT SCORING
# =====================================================================


def sentiment_score(
    text: Any,
) -> float:
    """
    Calculate a transparent lexicon-based sentiment score.

    Range:
        -1 = strongly negative
         0 = neutral
        +1 = strongly positive
    """

    cleaned = _clean_text(text)

    if not cleaned:
        return 0.0

    tokens = _tokenize(cleaned)

    if not tokens:
        return 0.0

    score = 0.0

    for token in tokens:

        if token in STRONG_POSITIVE:
            score += 2.0

        elif token in STRONG_NEGATIVE:
            score -= 2.0

        elif token in POSITIVE_WORDS:
            score += 1.0

        elif token in NEGATIVE_WORDS:
            score -= 1.0

    # Normalize by the number of tokens so long headlines
    # don't automatically receive larger scores.
    normalized = score / math.sqrt(
        max(len(tokens), 1)
    )

    # Convert to a stable [-1, +1] range.
    normalized = normalized / 2.0

    return float(
        np.clip(
            normalized,
            -1.0,
            1.0,
        )
    )


def sentiment_label(
    score: float,
) -> str:

    score = float(score)

    if score >= 0.20:
        return "POSITIVE"

    if score <= -0.20:
        return "NEGATIVE"

    return "NEUTRAL"


# =====================================================================
# NEWS ITEM ANALYSIS
# =====================================================================


def analyze_headline(
    headline: Any,
    summary: Any = "",
    published: Any = None,
) -> Dict[str, Any]:
    """
    Analyze a single news item.
    """

    headline_text = _clean_text(
        headline
    )

    summary_text = _clean_text(
        summary
    )

    combined = (
        f"{headline_text} {summary_text}"
    ).strip()

    score = sentiment_score(
        combined
    )

    return {
        "headline": (
            str(headline)
            if headline is not None
            else ""
        ),
        "summary": (
            str(summary)
            if summary is not None
            else ""
        ),
        "sentiment_score": float(score),
        "sentiment": sentiment_label(
            score
        ),
        "published": published,
    }


# =====================================================================
# NEWS DATAFRAME
# =====================================================================


def analyze_news(
    news: Optional[Iterable[Any]],
) -> pd.DataFrame:
    """
    Convert common news formats into a sentiment DataFrame.

    Supported:
    - list of dictionaries
    - pandas DataFrame
    - Yahoo Finance-style news dictionaries
    """

    if news is None:
        return pd.DataFrame(
            columns=[
                "headline",
                "summary",
                "published",
                "sentiment_score",
                "sentiment",
            ]
        )

    if isinstance(news, pd.DataFrame):
        records = news.to_dict(
            orient="records"
        )

    else:
        try:
            records = list(news)
        except Exception:
            records = []

    results = []

    for item in records:

        if not isinstance(item, dict):
            continue

        # ------------------------------------------------------------
        # Yahoo Finance variants
        # ------------------------------------------------------------

        content = item.get(
            "content",
            {},
        )

        if not isinstance(
            content,
            dict,
        ):
            content = {}

        headline = (
            item.get("title")
            or item.get("headline")
            or content.get("title")
            or ""
        )

        summary = (
            item.get("summary")
            or item.get("description")
            or content.get("summary")
            or content.get("description")
            or ""
        )

        published = (
            item.get("providerPublishTime")
            or item.get("pubDate")
            or item.get("published")
            or content.get("pubDate")
            or content.get("displayTime")
        )

        result = analyze_headline(
            headline=headline,
            summary=summary,
            published=published,
        )

        # Keep provider/source information when available.
        result["publisher"] = (
            item.get("publisher")
            or content.get("provider")
            or ""
        )

        result["link"] = (
            item.get("link")
            or item.get("url")
            or content.get("canonicalUrl", {}).get(
                "url",
                ""
            )
            if isinstance(
                content.get(
                    "canonicalUrl",
                    {},
                ),
                dict,
            )
            else item.get("link")
            or item.get("url")
            or ""
        )

        results.append(result)

    if not results:
        return pd.DataFrame(
            columns=[
                "headline",
                "summary",
                "published",
                "publisher",
                "link",
                "sentiment_score",
                "sentiment",
            ]
        )

    result_df = pd.DataFrame(
        results
    )

    return result_df


# =====================================================================
# SENTIMENT SUMMARY
# =====================================================================


def sentiment_summary(
    news_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Create a compact market sentiment summary.
    """

    if (
        news_df is None
        or news_df.empty
        or "sentiment_score" not in news_df.columns
    ):
        return {
            "score": 0.0,
            "label": "NO DATA",
            "articles": 0,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "sentiment_strength": 0.0,
        }

    scores = pd.to_numeric(
        news_df["sentiment_score"],
        errors="coerce",
    ).dropna()

    if scores.empty:
        return {
            "score": 0.0,
            "label": "NO DATA",
            "articles": 0,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "sentiment_strength": 0.0,
        }

    labels = (
        news_df["sentiment"]
        .astype(str)
        .str.upper()
    )

    positive = int(
        (labels == "POSITIVE").sum()
    )

    neutral = int(
        (labels == "NEUTRAL").sum()
    )

    negative = int(
        (labels == "NEGATIVE").sum()
    )

    articles = len(scores)

    score = float(
        scores.mean()
    )

    return {
        "score": score,
        "label": sentiment_label(
            score
        ),
        "articles": int(articles),
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "positive_ratio": float(
            positive / articles
        ),
        "negative_ratio": float(
            negative / articles
        ),
        "sentiment_strength": float(
            abs(score)
        ),
    }


# =====================================================================
# RECENT SENTIMENT
# =====================================================================


def recent_sentiment_score(
    news_df: pd.DataFrame,
    max_articles: int = 10,
) -> float:
    """
    Give greater importance to recent headlines.

    The function uses the ordering supplied by the news provider
    when publication dates cannot be parsed reliably.
    """

    if (
        news_df is None
        or news_df.empty
        or "sentiment_score" not in news_df.columns
    ):
        return 0.0

    data = news_df.copy()

    data["sentiment_score"] = pd.to_numeric(
        data["sentiment_score"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["sentiment_score"]
    )

    if data.empty:
        return 0.0

    # Try to parse publication timestamps.
    if "published" in data.columns:

        parsed = pd.to_datetime(
            data["published"],
            errors="coerce",
            utc=True,
        )

        data["_published_dt"] = parsed

        if parsed.notna().any():
            data = data.sort_values(
                "_published_dt",
                ascending=False,
            )

    data = data.head(
        max_articles
    )

    scores = data[
        "sentiment_score"
    ].to_numpy(
        dtype=float
    )

    if len(scores) == 0:
        return 0.0

    # More recent entries receive larger weights.
    weights = np.arange(
        len(scores),
        0,
        -1,
        dtype=float,
    )

    weighted_score = np.average(
        scores,
        weights=weights,
    )

    return float(
        np.clip(
            weighted_score,
            -1.0,
            1.0,
        )
    )


# =====================================================================
# NEWS FETCHING
# =====================================================================


def fetch_stock_news(
    ticker: str,
    max_items: int = 20,
) -> List[Dict[str, Any]]:
    """
    Fetch recent stock news through yfinance.

    Failure is handled gracefully so news availability never
    prevents the price-analysis application from running.
    """

    ticker = str(
        ticker
    ).strip()

    if not ticker:
        return []

    try:
        import yfinance as yf

        stock = yf.Ticker(
            ticker
        )

        news = stock.news

        if not news:
            return []

        if not isinstance(
            news,
            list,
        ):
            return []

        return news[:max_items]

    except Exception:
        return []


# =====================================================================
# COMPLETE SENTIMENT PIPELINE
# =====================================================================


def get_stock_sentiment(
    ticker: str,
    max_items: int = 20,
) -> Dict[str, Any]:
    """
    Fetch and analyze recent stock news.
    """

    news = fetch_stock_news(
        ticker=ticker,
        max_items=max_items,
    )

    news_df = analyze_news(
        news
    )

    summary = sentiment_summary(
        news_df
    )

    recent_score = (
        recent_sentiment_score(
            news_df
        )
    )

    return {
        "ticker": ticker,
        "news": news_df,
        "summary": summary,
        "recent_score": float(
            recent_score
        ),
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# =====================================================================
# EXPORTS
# =====================================================================


__all__ = [
    "sentiment_score",
    "sentiment_label",
    "analyze_headline",
    "analyze_news",
    "sentiment_summary",
    "recent_sentiment_score",
    "fetch_stock_news",
    "get_stock_sentiment",
]
