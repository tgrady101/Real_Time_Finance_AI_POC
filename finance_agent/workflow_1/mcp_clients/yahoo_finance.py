"""Yahoo Finance MCP client for real-time market data.

Uses the Yahoo Finance MCP server as an intermediary - NO yfinance installation needed
in the main application. The MCP server handles all Yahoo Finance dependencies.

The MCP server provides these tools:
- get_stock_info: Comprehensive stock data (price, metrics, company details)
- get_historical_stock_prices: OHLCV data with customizable period/interval
- get_yahoo_finance_news: Latest news articles for a stock
- get_stock_actions: Dividends and stock splits history
- get_financial_statement: Income statement, balance sheet, cash flow
- get_holder_info: Institutional holders, insider transactions
- get_option_expiration_dates: Available options expiration dates
- get_option_chain: Options chain for calls/puts
- get_recommendations: Analyst recommendations and upgrades/downgrades

For Cloud Run deployment:
- The MCP server is bundled in finance_agent/mcp_servers/
- Uses Python directly (no uv) in the container
"""

import os
import sys
from pathlib import Path


def _get_mcp_server_path() -> Path:
    """Get the path to the MCP server, handling both local dev and Cloud Run.
    
    Path resolution order:
    1. Bundled: Look for mcp_servers/ relative to this file (Cloud Run deployment)
    2. Check sys.path for finance_agent context
    3. Check current working directory
    """
    # This file is at: finance_agent/workflow_1/mcp_clients/yahoo_finance.py
    # MCP server is at: finance_agent/mcp_servers/
    bundled_path = Path(__file__).parent.parent.parent / "mcp_servers"
    if bundled_path.exists() and (bundled_path / "yahoo_finance_server.py").exists():
        return bundled_path
    
    # Check relative to current working directory (Cloud Run)
    cwd_bundled = Path.cwd() / "mcp_servers"
    if cwd_bundled.exists() and (cwd_bundled / "yahoo_finance_server.py").exists():
        return cwd_bundled
    
    # Fall back - shouldn't happen in Cloud Run
    return bundled_path


def _is_cloud_run() -> bool:
    """Detect if running in Cloud Run environment."""
    # Cloud Run sets K_SERVICE and K_REVISION environment variables
    return bool(os.getenv("K_SERVICE") or os.getenv("K_REVISION"))


# Get the absolute path to the MCP server
MCP_SERVER_PATH = _get_mcp_server_path()


def get_mcp_toolset_config():
    """Get the MCPToolset configuration for Yahoo Finance.
    
    Returns a configuration dict that can be used with Google ADK's McpToolset.
    
    Usage in an agent:
        from google.adk.agents import LlmAgent
        from google.adk.tools import McpToolset
        from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
        from mcp.client.stdio import StdioServerParameters
        from finance_agent.workflow_1.mcp_clients.yahoo_finance import get_mcp_toolset_config
        
        config = get_mcp_toolset_config()
        agent = LlmAgent(
            model='gemini-2.0-flash',
            name='market_data_agent',
            instruction='Help users get stock market data.',
            tools=[
                McpToolset(
                    connection_params=StdioConnectionParams(
                        server_params=StdioServerParameters(
                            command=config['command'],
                            args=config['args'],
                        )
                    )
                )
            ]
        )
    """
    return {
        "server_path": str(MCP_SERVER_PATH),
        "command": "uv",
        "args": ["run", "--directory", str(MCP_SERVER_PATH), "python", "server.py"],
    }


def get_mcp_server_params():
    """Get StdioConnectionParams for the Yahoo Finance MCP server.
    
    Returns:
        StdioConnectionParams configured for the Yahoo Finance MCP server
        
    Usage:
        from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
        from mcp.client.stdio import StdioServerParameters
        from google.adk.tools import McpToolset
        
        toolset = McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="uv",
                    args=["--directory", str(MCP_SERVER_PATH), "run", "server.py"],
                )
            )
        )
    """
    # Import here to avoid requiring packages at module load time
    from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
    from mcp.client.stdio import StdioServerParameters
    
    return StdioConnectionParams(
        server_params=StdioServerParameters(
            command="uv",
            args=["run", "--directory", str(MCP_SERVER_PATH), "python", "server.py"],
        ),
        timeout=30.0,  # Increase timeout for initial connection
    )


def create_yahoo_finance_toolset():
    """Create an McpToolset configured for Yahoo Finance.
    
    Returns:
        McpToolset ready to be added to an agent's tools list
        
    Usage:
        from google.adk.agents import LlmAgent
        from workflow_1.mcp_clients.yahoo_finance import create_yahoo_finance_toolset
        
        agent = LlmAgent(
            model='gemini-2.0-flash',
            name='market_data_agent',
            instruction='Help users get stock market data.',
            tools=[create_yahoo_finance_toolset()]
        )
    """
    from google.adk.tools import McpToolset
    from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
    from mcp.client.stdio import StdioServerParameters
    
    # In Cloud Run deployment, always use Python directly with bundled server
    server_script = str(MCP_SERVER_PATH / "yahoo_finance_server.py")
    command = sys.executable  # Use the current Python interpreter
    args = [server_script]
    
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=command,
                args=args,
            ),
            timeout=30.0,  # Increase timeout for initial connection
        ),
    )


# Available MCP tools documentation
MCP_TOOLS = {
    "get_stock_info": {
        "description": "Get comprehensive stock data including price, metrics, and company details",
        "parameters": {"ticker": "Stock ticker symbol (e.g., 'AAPL')"},
    },
    "get_historical_stock_prices": {
        "description": "Get historical OHLCV data for a stock",
        "parameters": {
            "ticker": "Stock ticker symbol",
            "period": "Time period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max",
            "interval": "Data interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo",
        },
    },
    "get_yahoo_finance_news": {
        "description": "Get latest news articles for a stock",
        "parameters": {"ticker": "Stock ticker symbol"},
    },
    "get_stock_actions": {
        "description": "Get stock dividends and splits history",
        "parameters": {"ticker": "Stock ticker symbol"},
    },
    "get_financial_statement": {
        "description": "Get financial statements (income, balance sheet, cash flow)",
        "parameters": {
            "ticker": "Stock ticker symbol",
            "financial_type": "One of: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow",
        },
    },
    "get_holder_info": {
        "description": "Get holder information (institutional, insider, etc.)",
        "parameters": {
            "ticker": "Stock ticker symbol",
            "holder_type": "One of: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders",
        },
    },
    "get_option_expiration_dates": {
        "description": "Get available options expiration dates",
        "parameters": {"ticker": "Stock ticker symbol"},
    },
    "get_option_chain": {
        "description": "Get options chain for a specific expiration",
        "parameters": {
            "ticker": "Stock ticker symbol",
            "expiration_date": "Expiration date (YYYY-MM-DD)",
            "option_type": "Either 'calls' or 'puts'",
        },
    },
    "get_recommendations": {
        "description": "Get analyst recommendations or upgrades/downgrades",
        "parameters": {
            "ticker": "Stock ticker symbol",
            "recommendation_type": "Either 'recommendations' or 'upgrades_downgrades'",
            "months_back": "Number of months back for upgrades/downgrades (default: 12)",
        },
    },
}


if __name__ == "__main__":
    print("\n=== Yahoo Finance MCP Client Configuration ===\n")
    
    config = get_mcp_toolset_config()
    print(f"MCP Server Path: {config['server_path']}")
    print(f"Command: {config['command']} {' '.join(config['args'])}")
    
    print("\n=== Available MCP Tools ===\n")
    for tool_name, info in MCP_TOOLS.items():
        print(f"📌 {tool_name}")
        print(f"   {info['description']}")
        print(f"   Parameters: {info['parameters']}")
        print()
    
    print("\n=== Usage Example ===\n")
    print("""
from google.adk.agents import LlmAgent
from finance_agent.workflow_1.mcp_clients.yahoo_finance import create_yahoo_finance_toolset

# Create an agent with Yahoo Finance MCP tools
market_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='MarketDataAgent',
    instruction='Help users get real-time stock market data.',
    tools=[create_yahoo_finance_toolset()]
)
""")

