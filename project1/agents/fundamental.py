import os
import json
from datetime import datetime, timedelta
import yfinance as yf
import alpaca_trade_api as tradeapi

# Load keys securely from environment
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"

# Initialize Alpaca client
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Ideally place this class in utils/state.py
class HumanMessage:
    def __init__(self, content, name):
        self.content = content
        self.name = name

def get_historical_data(ticker, start_date, end_date):
    try:
        bars = api.get_bars(
            ticker,
            tradeapi.rest.TimeFrame.Day,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        ).df

        if bars.empty:
            return None

        price_changes = [
            (bars.iloc[i].close - bars.iloc[i - 1].close) / bars.iloc[i - 1].close
            for i in range(1, len(bars))
        ]
        avg_price_change = sum(price_changes) / len(price_changes) if price_changes else 0
        return {"price_change_avg": avg_price_change}
    except Exception as e:
        print(f"[Error] Failed to get historical data for {ticker}: {e}")
        return None

def get_financial_ratios(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "price_to_earnings_ratio": info.get("trailingPE"),
            "price_to_book_ratio": info.get("priceToBook"),
            "earnings_per_share": info.get("epsTrailingTwelveMonths"),
            "debt_to_equity": info.get("debtToEquity"),
        }
    except Exception as e:
        print(f"[Error] Failed to get financial ratios for {ticker}: {e}")
        return None

def fundamentals_agent(state):
    data = state["data"]
    tickers = data.get("tickers", [])
    today = datetime.now()
    start_date = today.replace(day=1)
    end_date = today - timedelta(days=7)

    fundamental_analysis = {}

    for ticker in tickers:
        financial_metrics = get_historical_data(ticker, start_date, end_date)
        financial_ratios = get_financial_ratios(ticker)

        if not financial_metrics or not financial_ratios:
            continue

        price_change_avg = financial_metrics["price_change_avg"]
        pe_ratio = financial_ratios["price_to_earnings_ratio"]

        signal = (
            "bullish" if price_change_avg > 0
            else "bearish" if price_change_avg < 0
            else "neutral"
        )

        reasoning = {
            "profitability_signal": {
                "signal": signal,
                "details": f"Avg Price Change: {price_change_avg:.2%}, P/E: {pe_ratio or 'N/A'}",
            }
        }

        confidence = 100  # Always 100% for now since it's single-signal based

        fundamental_analysis[ticker] = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    message = HumanMessage(content=json.dumps(fundamental_analysis, indent=2), name="fundamentals_agent")
    data.setdefault("analyst_signals", {})["fundamentals_agent"] = fundamental_analysis

    return {
        "messages": state.get("messages", []) + [message],
        "data": data,
    }
