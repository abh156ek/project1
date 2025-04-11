import os
import logging
import requests
from typing import List, Dict
from dotenv import load_dotenv
import json
from datetime import datetime

# Load env vars
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HumanMessage:
    def __init__(self, content, name):
        self.content = content
        self.name = name

def fetch_recent_news(ticker: str, limit: int = 5) -> List[str]:
    url = f"https://data.alpaca.markets/v1beta1/news?symbols={ticker}&limit={limit}"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        news_data = response.json()
        return [article.get("headline", "") for article in news_data.get("news", [])]
    except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {e}")
        return []

def analyze_sentiment(text: str) -> float:
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

    try:
        response = requests.post(url, headers=headers, json={"inputs": text}, timeout=10)
        response.raise_for_status()
        result = response.json()
        if not result or not isinstance(result[0], list):
            return 0.0

        top = result[0][0]
        label, score = top["label"], top["score"]

        if label == "positive":
            return score
        elif label == "negative":
            return -score
        return 0.0
    except Exception as e:
        logger.warning(f"Sentiment analysis failed for text: {text[:50]}... — {e}")
        return 0.0

def get_sentiment_score(ticker: str) -> float:
    headlines = fetch_recent_news(ticker)
    if not headlines:
        return 0.0
    scores = [analyze_sentiment(h) for h in headlines]
    avg = sum(scores) / len(scores) if scores else 0.0
    return round(avg, 3)

def sentiment_agent(state: Dict) -> Dict:
    data = state.get("data", {})
    tickers = data.get("tickers", [])

    sentiment_analysis = {}

    for ticker in tickers:
        score = get_sentiment_score(ticker)
        sentiment = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"

        sentiment_analysis[ticker] = {
            "sentiment_score": score,
            "signal": sentiment,
            "reasoning": f"Average sentiment score from news headlines: {score}"
        }

    message = HumanMessage(
        content=json.dumps(sentiment_analysis, indent=2),
        name="sentiment_agent"
    )
    data.setdefault("analyst_signals", {})["sentiment_agent"] = sentiment_analysis

    return {
        "messages": state.get("messages", []) + [message],
        "data": data,
    }
