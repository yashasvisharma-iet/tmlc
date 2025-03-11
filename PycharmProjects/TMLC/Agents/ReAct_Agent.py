import yfinance as yf
from googlesearch import search
from crewai import Agent, Task, Crew
import ollama

# Function to fetch stock data
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "current_price": info.get("currentPrice"),
        "pe_ratio": info.get("trailingPE"),
        "debt_to_equity": info.get("debtToEquity"),
        "roe": info.get("returnOnEquity"),
    }

# Function to fetch financial news
def fetch_news(ticker):
    query = f"{ticker} stock news"
    return [url for url in search(query, num_results=3)]

# Function to analyze investment risk using Ollama
def analyze_investment(stock_data, news_urls):
    prompt = f"""
    Analyze the following financial metrics and news for investment advice:

    - Current Price: {stock_data['current_price']}
    - PE Ratio: {stock_data['pe_ratio']}
    - Debt-to-Equity Ratio: {stock_data['debt_to_equity']}
    - ROE: {stock_data['roe']}
    - Recent News: {news_urls}

    Should an investor buy, hold, or sell this stock? Provide a detailed explanation.
    """

    response = ollama.chat(model="phi3", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Define the Agent
financial_agent = Agent(
    role="Investment Analyst",
    goal="Analyze stock trends, calculate financial ratios, fetch market news, and provide investment recommendations.",
    backstory="A finance expert with a deep understanding of stock markets, risk analysis, and investment strategies.",
    llm={"model": "mistral"},
)

# Define the Task
analyze_stock_task = Task(
    description="Fetch stock data, analyze risk metrics, retrieve financial news, and generate investment advice.",
    agent=financial_agent,
    expected_output="A detailed investment recommendation with a Buy/Hold/Sell decision and reasoning."
)


# Crew to execute the task
crew = Crew(agents=[financial_agent], tasks=[analyze_stock_task])

# Main Execution
if __name__ == "__main__":
    ticker = input("Enter a stock ticker (e.g., TSLA, AAPL): ").upper()
    stock_data = fetch_stock_data(ticker)
    news_urls = fetch_news(ticker)
    recommendation = analyze_investment(stock_data, news_urls)
    print("\n### Investment Recommendation ###")
    print(recommendation)
