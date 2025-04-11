# Hedge Fund Portfolio Analysis System (Modular)

# --- 📦 Install Required Libraries (Uncomment if needed) ---
# %pip install pandas matplotlib seaborn plotly langchain langgraph pydantic

# --- 📁 Import Agent Functions ---
from agents.sentiment import get_sentiment_score
from agents.risk import get_risk_score
from agents.fundamentals import get_fundamentals
from agents.technical import get_technical_signal

# --- 🔧 Core Imports ---
from typing import List, Dict
from pydantic import BaseModel
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- 📊 Input Model ---
class PortfolioInput(BaseModel):
    tickers: List[str]

# --- 🧠 Agent Composition Layer ---
def run_all_agents(portfolio: PortfolioInput) -> Dict:
    result = {}
    for ticker in portfolio.tickers:
        result[ticker] = {
            "sentiment": get_sentiment_score(ticker),
            "risk": get_risk_score(ticker),
            "fundamentals": get_fundamentals(ticker),
            "technical": get_technical_signal(ticker),
        }
    return result

# --- ⚙️ Execution Layer ---
user_input = input("Enter a list of tickers (comma-separated): ")
tickers = [ticker.strip().upper() for ticker in user_input.split(",") if ticker.strip()]
portfolio_input = PortfolioInput(tickers=tickers)

agent_output = run_all_agents(portfolio_input)
agent_output

# --- 📊 Aggregation Layer ---
def aggregate_scores(agent_output: Dict) -> Dict:
    total_sentiment = 0
    total_risk = 0
    total_earnings = 0
    count = len(agent_output)

    for data in agent_output.values():
        total_sentiment += data['sentiment']
        total_risk += data['risk']
        total_earnings += data['fundamentals']['earnings']

    return {
        "avg_sentiment": round(total_sentiment / count, 2),
        "avg_risk": round(total_risk / count, 2),
        "total_earnings": round(total_earnings, 2),
    }

aggregated_results = aggregate_scores(agent_output)
aggregated_results

# --- 📈 Visualization Layer ---
def plot_sentiment_risk(agent_output: Dict):
    df = pd.DataFrame([
        {
            "Ticker": ticker,
            "Sentiment": data["sentiment"],
            "Risk": data["risk"]
        }
        for ticker, data in agent_output.items()
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Ticker", y="Sentiment", data=df, color="skyblue", ax=ax, label="Sentiment")
    sns.barplot(x="Ticker", y="Risk", data=df, color="salmon", ax=ax, alpha=0.6, label="Risk")
    plt.title("Sentiment and Risk per Asset")
    plt.legend(title="Metric")
    plt.show()

plot_sentiment_risk(agent_output)

# --- 🧾 Report Generation Layer ---
def generate_summary(agent_output: Dict, aggregated: Dict):
    print("\n📋 Portfolio Summary Report")
    print("=" * 40)
    print(f"Average Sentiment: {aggregated['avg_sentiment']}")
    print(f"Average Risk: {aggregated['avg_risk']}")
    print(f"Total Earnings Estimate: {aggregated['total_earnings']}")

    print("\nIndividual Ticker Summary:")
    for ticker, data in agent_output.items():
        print(f"\n🔹 {ticker}")
        print(f"  Sentiment: {data['sentiment']}")
        print(f"  Risk: {data['risk']}")
        print(f"  Technical Signal: {data['technical']}")
        print(f"  Fundamentals: {data['fundamentals']}")

# Display report
generate_summary(agent_output, aggregated_results)
