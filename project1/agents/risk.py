import os
import requests
from typing import Dict
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = "https://data.alpaca.markets/v2/stocks"

def get_prices(ticker: str, start_date: str, end_date: str):
    url = f"{BASE_URL}/{ticker}/bars?start={start_date}&end={end_date}&timeframe=1Day"
    headers = {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return [{"close": bar["c"]} for bar in data.get("bars", [])]
    except Exception as e:
        print(f"[Error] Failed to fetch prices for {ticker}: {e}")
        return []

def prices_to_df(prices):
    return {"close": [p["close"] for p in prices]}

def risk_management_agent(portfolio: Dict) -> Dict:
    tickers = portfolio.get("tickers", [])
    start_date = portfolio.get("start_date")
    end_date = portfolio.get("end_date")

    if isinstance(start_date, datetime):
        start_date = start_date.strftime("%Y-%m-%d")
    if isinstance(end_date, datetime):
        end_date = end_date.strftime("%Y-%m-%d")

    cost_basis = portfolio.get("cost_basis", {})
    cash = portfolio.get("cash", 0)
    total_value = cash + sum(cost_basis.values())
    position_limit_pct = 0.20  # 20% of total portfolio value

    risk_analysis = {}

    for ticker in tickers:
        prices = get_prices(ticker, start_date, end_date)
        if not prices:
            continue

        df = prices_to_df(prices)
        current_price = df["close"][-1]
        current_position_value = cost_basis.get(ticker, 0)

        position_limit = total_value * position_limit_pct
        remaining_limit = position_limit - current_position_value
        max_position_size = min(remaining_limit, cash)

        risk_analysis[ticker] = {
            "remaining_position_limit": float(max_position_size),
            "current_price": float(current_price),
            "reasoning": {
                "portfolio_value": float(total_value),
                "current_position": float(current_position_value),
                "position_limit": float(position_limit),
                "remaining_limit": float(remaining_limit),
                "available_cash": float(cash),
            },
        }

    return {"risk_analysis": risk_analysis}
