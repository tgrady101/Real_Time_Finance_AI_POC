"""Market Data Agent using Yahoo Finance MCP server.

This agent provides real-time stock market data through the Yahoo Finance MCP server.
It handles:
- Real-time stock prices and metrics
- Historical price data (OHLCV)
- Stock news and analyst recommendations
- Options data (chains, expirations)
- Financial statements
- Holder information (institutional, insider)

The agent validates S&P 500 tickers before making requests and provides
user-friendly responses with relevant market context.
"""

from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path, override=True)


# Agent instruction prompt for the Market Data sub-agent
MARKET_DATA_INSTRUCTION = """You are the Market Data Agent specializing in real-time stock market information.

Your primary responsibilities:
1. **Stock Prices**: Get current prices, daily changes, volume, and key metrics for stocks
2. **Historical Data**: Retrieve historical OHLCV data for technical analysis
3. **Market News**: Fetch latest news articles for specific stocks
4. **Analyst Opinions**: Get recommendations and upgrade/downgrade history
5. **Options Data**: Retrieve options chains, expiration dates, calls/puts
6. **Financial Statements**: Access income statements, balance sheets, cash flow
7. **Holder Information**: Show institutional holders, insider transactions

Guidelines:
- Focus on S&P 500 stocks. If a ticker seems invalid, suggest corrections.
- Always include relevant context (market cap, sector, P/E ratio) when providing prices.
- For historical data, recommend appropriate periods based on the analysis need.
- Format large numbers readably (e.g., $4.1T for trillion, $150.5B for billion).
- When showing price changes, include both absolute and percentage values.
- Provide concise summaries - don't dump raw data unless specifically requested.

Available Tools (via Yahoo Finance MCP):
- get_stock_info: Comprehensive stock data (price, metrics, company details)
- get_historical_stock_prices: OHLCV data with period/interval parameters
- get_yahoo_finance_news: Latest news articles for a stock
- get_stock_actions: Dividends and stock splits history  
- get_financial_statement: Income statement, balance sheet, cash flow
- get_holder_info: Institutional/insider holdings
- get_option_expiration_dates: Available options expirations
- get_option_chain: Calls/puts options data
- get_recommendations: Analyst recommendations and upgrades/downgrades

Period options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
Interval options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo

Example responses:
- "AAPL is trading at $278.85, up 1.2% today. Market cap: $4.1T, P/E: 37.3"
- "Here's MSFT's 5-day price history showing a 3.5% gain..."
- "Recent NVDA news: 3 articles about AI chip demand..."
"""


def create_market_data_agent(model: Optional[str] = None, use_dynamic_routing: bool = True):
    """Create the Market Data sub-agent with Yahoo Finance MCP tools.
    
    This creates an LlmAgent configured as a sub-agent for the root orchestrator.
    The agent uses Yahoo Finance MCP server for real-time market data.
    
    Args:
        model: Gemini model to use (defaults to config GEMINI_MODEL)
        use_dynamic_routing: If True, use before_model_callback for per-query routing
        
    Returns:
        LlmAgent configured with Yahoo Finance MCP tools
    """
    from google.adk.agents import LlmAgent
    from ..config import Config
    from ..mcp_clients.yahoo_finance import create_yahoo_finance_toolset
    
    agent_model = model or Config.GEMINI_MODEL
    
    # Create MCP toolset (connects to Yahoo Finance MCP server)
    mcp_toolset = create_yahoo_finance_toolset()
    
    # Set up dynamic routing callback if enabled
    dynamic_callback = None
    if use_dynamic_routing:
        from ..utils.dynamic_model_callback import create_dynamic_model_callback
        dynamic_callback = create_dynamic_model_callback(
            fast_model=Config.MODEL_FAST,
            complex_model=Config.MODEL_COMPLEX,
        )
    
    # Create the sub-agent with MCP tools
    market_data_agent = LlmAgent(
        model=agent_model,
        name="market_data_agent",
        description="Handles all stock market data requests: prices, historical data, news, financials, options, and analyst recommendations via Yahoo Finance.",
        instruction=MARKET_DATA_INSTRUCTION,
        tools=[mcp_toolset],
        before_model_callback=dynamic_callback,
    )
    
    return market_data_agent


if __name__ == "__main__":
    print("\n=== Market Data Agent (Sub-Agent) ===\n")
    print("This agent is designed to be used as a sub-agent of the root orchestrator.")
    print("Use the root_agent.py to run the full finance assistant.")
    print("\nTo create the agent:")
    print("  from finance_agent.workflow_1.agents.market_data_agent import create_market_data_agent")
    print("  market_agent = create_market_data_agent()")
    print("\nAvailable MCP Tools:")
    tools = [
        "get_stock_info",
        "get_historical_stock_prices", 
        "get_yahoo_finance_news",
        "get_stock_actions",
        "get_financial_statement",
        "get_holder_info",
        "get_option_expiration_dates",
        "get_option_chain",
        "get_recommendations",
    ]
    for tool in tools:
        print(f"  • {tool}")
