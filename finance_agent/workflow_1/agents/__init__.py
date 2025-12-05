"""Finance AI Agents.

This module contains specialized agents for financial data analysis:
- root_agent: Main orchestrator that coordinates all sub-agents
- market_data_agent: Real-time stock prices and market data via Yahoo Finance MCP
- economic_agent: Economic indicators via FRED API (GDP, inflation, unemployment, rates)
- headlines_agent: Recent news and headlines via Google Search
- portfolio_agent: Portfolio analysis, risk metrics, and rebalancing
- vector_store_agent: Vertex AI Vector Search for Q2/Q3 2025 earnings call transcripts
- utility_agent: S&P 500 validation (ticker lookup, company info)
"""

from .market_data_agent import create_market_data_agent
from .economic_agent import create_economic_agent
from .headlines_agent import create_headlines_agent
from .portfolio_agent import create_portfolio_agent
from .vector_store_agent import create_vector_store_agent
from .utility_agent import create_utility_agent
from .root_agent import (
    create_root_agent,
    run_finance_assistant,
    get_root_agent,
    chat_loop,
)

__all__ = [
    # Root orchestrator
    "create_root_agent",
    "run_finance_assistant",
    "get_root_agent",
    "chat_loop",
    # Sub-agents
    "create_market_data_agent",
    "create_economic_agent",
    "create_headlines_agent",
    "create_portfolio_agent",
    "create_vector_store_agent",
    "create_utility_agent",
]
